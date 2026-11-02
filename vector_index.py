"""Chunking, local E5 embeddings and FAISS storage shared by build and search."""

from bisect import bisect_left
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "vector_index"
MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
DIMENSION = 768
MODEL_TOKENS = 512
BODY_TOKENS = 450
TARGET_WORDS = 200
OVERLAP_WORDS = 30
WORD = re.compile(r"\b[\w'-]+\b")
SEPARATORS = [r"\n\n", r"\n", r"(?<=[.!?])\s+", r"(?<=;)\s+", " ", ""]

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def word_count(text):
    return len(WORD.findall(text))


def passage(title, text):
    return f"passage: {title}\n{text}"


def document_paths(kb_dir):
    return sorted([*kb_dir.glob("*.md"), *kb_dir.glob("*.txt")])


class E5Encoder:
    def __init__(self, *, revision=MODEL_REVISION, offline=False, threads=4):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model = SentenceTransformer(
            MODEL_NAME, revision=revision, cache_folder=str(ROOT / ".cache/models"),
            device="cpu", local_files_only=offline, trust_remote_code=False, token=False,
            model_kwargs={"use_safetensors": True},
        )
        self.model.max_seq_length = MODEL_TOKENS
        self.tokenizer = self.model.tokenizer
        if self.model.get_sentence_embedding_dimension() != DIMENSION:
            raise ValueError("Unexpected embedding dimension")

    def count_tokens(self, text, *, special=True):
        return len(self.tokenizer.encode(text, add_special_tokens=special, truncation=False, verbose=False))

    def encode(self, texts, *, batch_size=8, progress=False):
        if not texts or any(self.count_tokens(text) > MODEL_TOKENS for text in texts):
            raise ValueError("Empty input or E5 input exceeds 512 tokens; truncation is forbidden")
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=progress, prompt="",
        )
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape != (len(texts), DIMENSION) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embeddings")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Embeddings are not L2-normalized")
        return vectors


def located_splits(text, splitter):
    """Locate exact substrings; LangChain's start-index overlap uses character units."""
    previous_start = -1
    for chunk in splitter.split_text(text):
        start = text.find(chunk, previous_start + 1)
        if start < 0:
            raise ValueError("Splitter changed the source text; cannot cite it exactly")
        yield start, start + len(chunk)
        previous_start = start


def chunk_spans(text, title, encoder):
    @lru_cache(maxsize=8192)
    def token_length(value):
        return encoder.count_tokens(value, special=False)

    def fits(start, end):
        value = text[start:end]
        return (word_count(value) <= 300 and token_length(value) <= BODY_TOKENS
                and encoder.count_tokens(passage(title, value)) <= MODEL_TOKENS)

    if word_count(text) < 100:
        if not fits(0, len(text)):
            raise ValueError("Short document exceeds E5 token limit")
        return [(0, len(text))]

    words = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_WORDS, chunk_overlap=OVERLAP_WORDS, length_function=word_count,
        separators=SEPARATORS, is_separator_regex=True, keep_separator=True,
    )
    tokens = RecursiveCharacterTextSplitter(
        chunk_size=BODY_TOKENS, chunk_overlap=50, length_function=token_length,
        separators=SEPARATORS, is_separator_regex=True, keep_separator=True,
    )
    spans = []
    for parent_start, parent_end in located_splits(text, words):
        parent = text[parent_start:parent_end]
        for start, end in located_splits(parent, tokens):
            spans.append((parent_start + start, parent_start + end))
    spans = sorted(set(spans))

    # Merge short tails if budgets allow; otherwise extend with adjacent source words.
    positions = list(WORD.finditer(text))
    starts = [match.start() for match in positions]
    result = []
    for i, (start, end) in enumerate(spans):
        if word_count(text[start:end]) < 100:
            if result and fits(result[-1][0], end):
                start = result.pop()[0]
            elif i + 1 < len(spans) and fits(start, spans[i + 1][1]):
                spans[i + 1] = (start, spans[i + 1][1])
                continue
            else:
                last = bisect_left(starts, end)
                extended_start = starts[max(0, last - 100)]
                if extended_start <= start and fits(extended_start, end):
                    start = extended_start
                if word_count(text[start:end]) < 100:
                    first = bisect_left(starts, start)
                    extended_end = positions[min(len(positions) - 1, first + 99)].end()
                    if extended_end >= end and fits(start, extended_end):
                        end = extended_end
        if word_count(text[start:end]) < 100 or not fits(start, end):
            raise ValueError(f"Cannot make a 100–300-word chunk within token budget: {title}")
        result.append((start, end))

    # Extension can make an earlier chunk fully contained in a later one.
    result = [span for span in sorted(set(result)) if not any(
        other != span and other[0] <= span[0] and span[1] <= other[1] for other in result
    )]
    cursor = 0
    for start, end in result:
        if start > cursor and text[cursor:start].strip():
            raise ValueError(f"Chunking lost source text in {title}")
        cursor = max(cursor, end)
    if text[cursor:].strip():
        raise ValueError(f"Chunking lost the end of {title}")
    return result


def make_chunks(encoder, kb_dir=KB_DIR):
    chunks, sources = [], []
    # Deliberately non-recursive: no source_snapshot, terms_map, reports or code.
    paths = document_paths(kb_dir)
    if not paths:
        raise ValueError("The knowledge base is empty")
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Empty source document: {path.name}")
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else path.stem
        source = f"knowledge_base/{path.name}"
        digest = sha256(raw)
        sources.append({"source": source, "sha256": digest})
        headings = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^#{2,6}\s+(.+)$", text, re.M)]
        spans = chunk_spans(text, title, encoder)
        for number, (start, end) in enumerate(spans):
            content = text[start:end]
            previous = [name for pos, name in headings if pos <= start]
            sections = list(dict.fromkeys((previous[-1:] if previous else []) + [
                name for pos, name in headings if start < pos < end
            ]))
            chunk = {
                "chunk_id": f"{path.stem}:{number:03d}:" + sha256(f"{source}:{digest}:{start}:{end}".encode())[:12],
                "row_id": len(chunks), "source": source, "title": title, "chunk_index": number,
                "source_sha256": digest, "char_start": start, "char_end": end,
                "line_start": text.count("\n", 0, start) + 1,
                "line_end": text.count("\n", 0, end - 1) + 1, "sections": sections,
                "word_count": word_count(content), "input_tokens": encoder.count_tokens(passage(title, content)),
                "text": content,
            }
            if chunk["word_count"] < 100:
                chunk["short_reason"] = "source_shorter_than_100_words"
            chunks.append(chunk)
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    return chunks, sources


def load_index(directory=INDEX_DIR, *, root=ROOT):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["artifacts"]) != {"faiss.index", "chunks.jsonl"}:
        raise ValueError("Index artifact list is incomplete")
    for name, expected in manifest["artifacts"].items():
        if name not in {"faiss.index", "chunks.jsonl"} or sha256((directory / name).read_bytes()) != expected:
            raise ValueError(f"Index artifact checksum mismatch: {name}")
    chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    index = faiss.read_index(str(directory / "faiss.index"))
    model = manifest["embedding_model"]
    if (manifest["schema_version"] != 1 or model["name"] != MODEL_NAME
            or model["dimension"] != DIMENSION or model["passage_prefix"] != "passage: "
            or model["query_prefix"] != "query: " or model["max_input_tokens"] != MODEL_TOKENS
            or not model["normalized"]):
        raise ValueError("Unsupported embedding configuration")
    if index.ntotal != len(chunks) or index.d != DIMENSION or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS and metadata are incompatible")
    if manifest["chunks"] != len(chunks) or manifest["faiss"]["vectors"] != index.ntotal:
        raise ValueError("Manifest count does not match index")
    current_sources = {f"knowledge_base/{path.name}" for path in document_paths(root / "knowledge_base")}
    indexed_sources = {source["source"] for source in manifest["sources"]}
    if current_sources != indexed_sources:
        raise ValueError("Knowledge base file list changed; rebuild index")
    texts = {}
    for source in manifest["sources"]:
        path = (root / source["source"]).resolve()
        if not path.is_relative_to((root / "knowledge_base").resolve()):
            raise ValueError("Source outside knowledge_base")
        data = path.read_bytes()
        if sha256(data) != source["sha256"]:
            raise ValueError(f"Knowledge base changed; rebuild index: {source['source']}")
        texts[source["source"]] = data.decode("utf-8")
    for row, chunk in enumerate(chunks):
        text = texts[chunk["source"]]
        start, end = chunk["char_start"], chunk["char_end"]
        if (chunk["row_id"] != row or not 0 <= start < end <= len(text)
                or text[start:end] != chunk["text"]
                or chunk["line_start"] != text.count("\n", 0, start) + 1
                or chunk["line_end"] != text.count("\n", 0, end - 1) + 1):
            raise ValueError("Chunk text or position does not match its source")
    return index, chunks, manifest


class Retriever:
    def __init__(self, directory=INDEX_DIR, *, threads=4):
        self.index, self.chunks, self.manifest = load_index(directory)
        faiss.omp_set_num_threads(threads)
        self.encoder = E5Encoder(revision=self.manifest["embedding_model"]["revision"], offline=True, threads=threads)

    def search(self, query, top_k=3):
        if not query.strip() or not 1 <= top_k <= len(self.chunks):
            raise ValueError("Query must be nonempty; top_k must be between 1 and index size")
        vector = self.encoder.encode(["query: " + query.strip()])
        scores, rows = self.index.search(vector, top_k)
        return [{**self.chunks[int(row)], "score": float(score)} for score, row in zip(scores[0], rows[0])]
