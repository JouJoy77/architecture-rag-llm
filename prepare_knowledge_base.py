"""Download wiki articles and build a renamed, text-only knowledge base.

Run once online, then use --offline to rebuild from data/source_snapshot.json.
Only knowledge_base/*.md is intended for the future vector index.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
API = "https://rickandmorty.fandom.com/api.php"
WIKI = "https://rickandmorty.fandom.com/wiki/"
SNAPSHOT = ROOT / "data/source_snapshot.json"
MAP_PATH = ROOT / "terms_map.json"
OUTPUT = ROOT / "knowledge_base"
SKIP_SECTIONS = {
    "appearances", "episode appearances", "gallery", "trivia", "references",
    "external links", "behind the scenes", "quotes", "quote", "notes",
    "see also", "navigation", "in other media", "videos", "video", "references and notes",
    "pocket mortys", "story train", "other media", "in games", "video games",
}
META_SENTENCE = re.compile(
    r"\b(?:episodes?|seasons?|series|protagonists?|antagonists?|viewers?|audience|"
    r"voiced|voice actors?|portrayed|writers?|showrunners?|comics?|manga|"
    r"Justin Roiland|Dan Harmon|Adult Swim|Fandom|Wikia|cameos?|non-canon|"
    r"screen time|on screen|first appearance|first appeared|debuted|show|canon|complete list|"
    r"deuteragonists?|tritagonists?|recurring character|character development|"
    r"potential for more development|scenarios showing|movie characters?|legally[- ]safe|iPhone)\b", re.I,
)
# Independent of terms_map.json: a missing alias must not make the corpus check pass.
SOURCE_MARKERS = re.compile(
    r"\b(?:rick\w*|morty\w*|meeseek\w*|squanch\w*|gromflom\w*|gazorp\w*|"
    r"cromulon\w*|plumbus\w*|froopy\w*|birdperson|poopybutthole|portal[- ]guns?|brainalyzer|"
    r"C[- ]?\d{2,}[A-Z]?|GFYs?)\b", re.I,
)
AMBIGUOUS_PROPER_NAMES = {"unity", "summer", "prime", "defiance", "resistance"}
WORD = re.compile(r"\b[\w'-]+\b")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text.translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "\u200b": "",
    }))).strip()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def request_json(url):
    """Retry transient failures; a forbidden request is not retried or bypassed."""
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "RAG-coursework/1.0 (educational research)"})
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def split_sentences(text):
    # Protect abbreviations; keep sentence-ending quotation marks with the sentence.
    protected = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|[A-Z])\.",
        lambda match: match.group().replace(".", "\ue000"), text,
    )
    return [part.strip().replace("\ue000", ".") for part in re.findall(
        r".+?(?:[.!?][\"']?(?=\s+[A-Z\"']|$)|$)", protected,
    ) if part.strip()]


def clean_sentence_text(text):
    """Keep in-world prose; omit sentences referring to the television show."""
    text = normalize(text)
    # Keep the corpus focused on entities and technology, without explicit side plots
    # or game mechanics. Remove whole paragraphs rather than orphan their context.
    if re.search(
        r"\b(?:sex|masturb\w*|copulat\w*|commentary)\b|\bin the game\b|Rick and Morty:", text, re.I,
    ):
        return ""
    text = re.sub(r"\s+and a recurring character in [^.]+(?=\.)", "", text, flags=re.I)
    text = re.sub(r"\btitular\s+", "", text, flags=re.I)
    text = re.sub(r"\bmain characters\b", "travelers", text, flags=re.I)
    text = re.sub(r"\bIn (?:a|the) previous scene,\s*", "", text, flags=re.I)
    text = re.sub(r'\bIn the ending credits for "[^"]+",\s*', "", text, flags=re.I)
    text = re.sub(r'\bIn the post-credits scene for [^,]+,\s*', "", text, flags=re.I)
    text = re.sub(r'\bThis hint is reinforced in the post credits scene of [^.]+\.', "", text, flags=re.I)
    text = re.sub(r"\bwe (?:learn|see|know)(?: that)?\s+", "", text, flags=re.I)
    sentences = split_sentences(text)
    return " ".join(sentence for sentence in sentences if not META_SENTENCE.search(sentence))


def entity_alias(label, title):
    """A link can label a person, a whole group, or just a grammatical reference."""
    if "(disambiguation)" in title.casefold():
        return None
    # Keep 's outside the replacement so the possessive survives.
    label = label.removesuffix("'s")
    if not label or not label[0].isupper() or label.casefold() not in title.casefold():
        return None
    return label


def extract_article(html, max_words):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output")
    if root is None:
        raise ValueError("No article body in API response")
    for node in root.select(
        "aside, table, figure, img, script, style, noscript, dl, blockquote, "
        ".toc, .navbox, .mw-editsection, .reference, .references, .reflist, "
        ".thumb, .gallery, .noprint, .hatnote, .metadata, [role='navigation'], .mw-collapsible, sup"
    ):
        node.decompose()

    paragraphs, terms = [], {}
    section, skipped_level, count = "Overview", None, 0
    for node in root.find_all(["h2", "h3", "h4", "p", "li"]):
        if node.name.startswith("h"):
            level = int(node.name[1])
            section = normalize(node.get_text(" ", strip=True)).strip(" []")
            if skipped_level is not None and level <= skipped_level:
                skipped_level = None
            if section.lower() in SKIP_SECTIONS:
                skipped_level = level
            continue
        if skipped_level is not None or node.find_parent(["li", "p"]) is not None:
            continue
        # Inline tags must not introduce spaces before punctuation or within words.
        text = clean_sentence_text(node.get_text())
        if node.name == "li":
            label = node.find("a", recursive=False)
            name = normalize(label.get_text()) if label else ""
            original = normalize(node.get_text())
            # Removing an episode reference must not leave a subjectless list item.
            if name and original.startswith(name + " (") and text.startswith("It "):
                text = name + text[2:]
                if text.endswith(").") and text.count(")") > text.count("("):
                    text = text[:-2] + "."
        if len(WORD.findall(text)) < 12 or text in {p["text"] for p in paragraphs}:
            continue
        if count >= max_words:
            break
        # Never cut a document mid-sentence, even if a paragraph exceeds the budget.
        kept = []
        for sentence in split_sentences(text):
            if count >= max_words:
                break
            kept.append(sentence)
            count += len(WORD.findall(sentence))
        text = " ".join(kept)
        if not text:
            continue
        for link in node.select('a[href]'):
            href = link.get("href", "")
            path = urlparse(href).path
            if not path.startswith("/wiki/") or ":" in unquote(path[6:]):
                continue
            label = normalize(link.get_text())
            title = normalize(unquote(path[6:]).replace("_", " ").split("#")[0])
            for candidate in (entity_alias(label, title), entity_alias(title, title)):
                if candidate is None:
                    continue
                if (3 <= len(candidate) <= 100 and any(c.isupper() for c in candidate)
                        and candidate in text and not candidate.startswith(("List of ", "Category:"))):
                    terms[candidate] = title
        # Episode/book names can be plain italic text, without a wiki link.
        for item in node.select("i, em"):
            title = normalize(item.get_text()).strip("\"'")
            if 3 <= len(title) <= 100 and title in text and any(c.isupper() for c in title):
                terms.setdefault(title, title)
        paragraphs.append({"section": section, "text": text})
    if not paragraphs:
        raise ValueError("No useful paragraphs after cleaning")
    return paragraphs, dict(sorted(terms.items()))


def fetch_page(page, delay):
    cache = ROOT / ".cache/wiki"
    cache.mkdir(parents=True, exist_ok=True)
    visited = set()
    for _ in range(5):
        if page in visited:
            raise ValueError(f"Redirect loop: {page}")
        visited.add(page)
        path = cache / (slug(page) + ".json")
        if path.exists():
            payload = read_json(path)
        else:
            url = f"{API}?action=parse&page={quote(page.replace(' ', '_'), safe='')}&prop=text&format=json"
            payload = request_json(url)
            if "parse" not in payload:
                raise ValueError(f"{page}: {payload}")
            payload["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(path, payload)
            time.sleep(delay)
        parsed = dict(payload["parse"])
        parsed["fetched_at_utc"] = payload.get(
            "fetched_at_utc", datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        )
        redirect = BeautifulSoup(parsed["text"]["*"], "html.parser").select_one(".redirectText a")
        if redirect is None:
            return parsed
        target = urlparse(redirect.get("href", "")).path
        if not target.startswith("/wiki/"):
            raise ValueError(f"Invalid redirect: {page}")
        page = unquote(target[6:]).replace("_", " ")
    raise ValueError(f"Too many redirects: {page}")


def download(sources, max_words, delay):
    records = []
    for index, source in enumerate(sources, 1):
        page = source["page"]
        parsed = fetch_page(page, delay)
        if "(episode)" in parsed["title"].lower() or "/transcript" in parsed["title"].lower():
            raise ValueError(f"Expected an entity article, got episode material: {parsed['title']}")
        html = parsed["text"]["*"]
        paragraphs, terms = extract_article(html, max_words)
        revision = re.search(r"revision id (\d+)", html)
        record = {
            **source,
            "resolved_title": parsed["title"],
            "page_id": parsed["pageid"],
            "revision_id": int(revision.group(1)) if revision else None,
            "fetched_at_utc": parsed["fetched_at_utc"],
            "url": WIKI + quote(parsed["title"].replace(" ", "_"), safe=""),
            "history_url": WIKI + quote(parsed["title"].replace(" ", "_"), safe="") + "?action=history",
            "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
            "paragraphs": paragraphs,
            "linked_terms": terms,
        }
        records.append(record)
        print(f"[{index}/{len(sources)}] {page}: {sum(len(WORD.findall(p['text'])) for p in paragraphs)} words", flush=True)
    if len({record["page_id"] for record in records}) != len(records):
        raise ValueError("Several requested titles resolve to the same article")
    return {
        "schema_version": 1,
        "snapshot_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Rick and Morty Wiki contributors",
        "license": "CC-BY-SA; see DATA_SOURCES.md",
        "max_words": max_words,
        "documents": records,
    }


def fictional_name(term):
    """Stable names: no network, LLM calls, global RNG, or Python hash()."""
    syllables = ("va", "ne", "lo", "ri", "sa", "te", "ko", "mi", "zu", "fa", "de", "ra", "vi", "no", "se", "lu")
    digest = hashlib.sha256(("veylan-world-v1:" + term.casefold()).encode()).digest()
    first = "".join(syllables[b % len(syllables)] for b in digest[:3]).capitalize()
    last = "".join(syllables[b % len(syllables)] for b in digest[3:6]).capitalize()
    return f"{first} {last}"


def make_replacer(mapping):
    lookup = {}
    for original, replacement in mapping.items():
        key = normalize(original).casefold()
        if key in lookup and lookup[key] != replacement:
            raise ValueError(f"Conflicting case-insensitive term: {original}")
        lookup[key] = replacement
    # One substitution pass: a generated name never becomes input to another rule.
    alternatives = []
    for key in sorted(lookup, key=lambda key: (-len(key), key)):
        if key in AMBIGUOUS_PROPER_NAMES:
            # Unity (a person) differs from unity (an ordinary noun).
            alternatives.append("(?-i:" + re.escape(key[0].upper()) + ")" + re.escape(key[1:]))
        else:
            alternatives.append(re.escape(key))
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)
    return pattern, lambda match: lookup[match.group().casefold()]


def complete_mapping(snapshot, mapping):
    lookup = {normalize(key).casefold(): value for key, value in mapping.items()}
    for record in snapshot["documents"]:
        for original in (record["page"], record["resolved_title"]):
            current = lookup.get(normalize(original).casefold())
            if current is not None and current != record["name"]:
                raise ValueError(f"Entity title and term map disagree: {original}: {current} != {record['name']}")
            mapping.setdefault(original, record["name"])
            lookup[normalize(original).casefold()] = record["name"]
        for paragraph in record["paragraphs"]:
            # Fandom uses both C-137 and C137 for the same reality.
            for match in re.finditer(r"\bC[- ]?(\d{2,}[A-Z]?)\b", paragraph["text"], re.I):
                canonical = "C-" + match.group(1).upper()
                replacement = lookup.get(canonical.casefold())
                if replacement is None:
                    replacement = "N-" + hashlib.sha256(canonical.encode()).hexdigest()[:6].upper()
                    mapping[canonical] = replacement
                    lookup[canonical.casefold()] = replacement
                mapping.setdefault(match.group(), replacement)
    aliases = {}
    for record in snapshot["documents"]:
        for alias, title in record["linked_terms"].items():
            alias = entity_alias(alias, title)
            if alias is None:
                continue
            aliases.setdefault(alias, set()).add(title)
    lookup = {normalize(key).casefold(): value for key, value in mapping.items()}
    for title in sorted({title for titles in aliases.values() for title in titles}):
        if normalize(title).casefold() not in lookup:
            mapping[title] = fictional_name(title)
            lookup[normalize(title).casefold()] = mapping[title]
    for alias, titles in sorted(aliases.items()):
        if normalize(alias).casefold() in lookup:
            continue
        if len(titles) != 1:
            raise ValueError(f"Ambiguous wiki alias; add an explicit term rule: {alias}: {sorted(titles)}")
        title = next(iter(titles))
        replacement = lookup[normalize(title).casefold()]
        # A linked short name follows the same entity as its full name.
        if " " not in alias and title.startswith(alias + " "):
            replacement = replacement.split()[0]
        mapping[alias] = replacement
        lookup[normalize(alias).casefold()] = replacement
    return dict(sorted(mapping.items(), key=lambda item: (item[0].casefold(), item[0])))


def build(snapshot, mapping):
    pattern, replace = make_replacer(mapping)
    documents = {}
    for record in snapshot["documents"]:
        paragraphs = []
        previous_section = None
        for item in record["paragraphs"]:
            section = pattern.sub(replace, normalize(item["section"]))
            if section != previous_section:
                paragraphs.append("## " + section)
                previous_section = section
            paragraphs.append(pattern.sub(replace, normalize(item["text"])))
        text = "# " + record["name"] + "\n\n" + "\n\n".join(paragraphs) + "\n"
        filename = slug(record["name"]) + ".md"
        if filename in documents:
            raise ValueError(f"Duplicate filename: {filename}")
        documents[filename] = text
    return documents


def validate(documents, mapping):
    if len(documents) < 30:
        raise ValueError("At least 30 documents are required")
    pattern, _ = make_replacer(mapping)
    seen, counts = set(), []
    for filename, text in documents.items():
        if pattern.search(text):
            raise ValueError(f"Original term remains in {filename}: {pattern.search(text).group()}")
        if re.search(r"https?://|<\s*/?\s*[a-zA-Z][^>]*>|\[edit\]|\[\s*\]", text):
            raise ValueError(f"HTML, URL or wiki markup remains in {filename}")
        if META_SENTENCE.search(text):
            raise ValueError(f"Out-of-world text remains in {filename}: {META_SENTENCE.search(text).group()}")
        if SOURCE_MARKERS.search(text):
            raise ValueError(f"Source-universe marker remains in {filename}: {SOURCE_MARKERS.search(text).group()}")
        if re.search(r"\bDr\.\s+Dr\.", text):
            raise ValueError(f"Duplicated title in {filename}")
        body = normalize(" ".join(line for line in text.splitlines() if not line.startswith("#")))
        digest = hashlib.sha256(body.casefold().encode()).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate document: {filename}")
        seen.add(digest)
        count = len(WORD.findall(text))
        if count < 80:
            raise ValueError(f"Too little useful text in {filename}: {count} words")
        counts.append(count)
    return {"documents": len(documents), "terms": len(mapping), "words": sum(counts),
            "min_words": min(counts), "max_words": max(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the tracked text snapshot; no HTTP calls")
    parser.add_argument("--check", action="store_true", help="Validate committed artifacts against the snapshot; no HTTP calls or writes")
    parser.add_argument("--max-words", type=int, default=700, help="Approximate prose budget per article")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay after each network request")
    args = parser.parse_args()
    if args.max_words < 100 or args.delay < 0:
        parser.error("--max-words must be >= 100 and --delay must be nonnegative")
    sources = read_json(ROOT / "data/kb_sources.json")
    snapshot = read_json(SNAPSHOT) if args.offline or args.check else download(sources, args.max_words, args.delay)
    if [(r["page"], r["name"], r["category"]) for r in snapshot["documents"]] != [
        (r["page"], r["name"], r["category"]) for r in sources
    ]:
        raise ValueError("Snapshot does not match data/kb_sources.json; rebuild it online")
    mapping = complete_mapping(snapshot, read_json(MAP_PATH))
    documents = build(snapshot, mapping)
    report = validate(documents, mapping)
    report["categories"] = dict(Counter(r["category"] for r in snapshot["documents"]))
    if args.check:
        if mapping != read_json(MAP_PATH):
            raise ValueError("terms_map.json is incomplete")
        actual = {p.name: p.read_text(encoding="utf-8") for p in OUTPUT.glob("*.md")}
        if actual != documents:
            raise ValueError("knowledge_base differs from the snapshot and term map")
        if read_json(ROOT / "data/kb_build_report.json") != report:
            raise ValueError("data/kb_build_report.json does not match the knowledge base")
    else:
        OUTPUT.mkdir(exist_ok=True)
        expected = set(documents)
        stale = {path.name for path in OUTPUT.glob("*.md")} - expected
        if stale:
            raise ValueError(f"Unexpected documents in knowledge_base; review manually: {sorted(stale)}")
        write_json(SNAPSHOT, snapshot)
        write_json(MAP_PATH, mapping)
        for filename, text in documents.items():
            (OUTPUT / filename).write_text(text, encoding="utf-8")
        write_json(ROOT / "data/kb_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
