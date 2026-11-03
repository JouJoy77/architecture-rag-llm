"""Regression checks for term replacement and wiki cleanup; no network calls."""

import unittest

from prepare_knowledge_base import (
    MAP_PATH, clean_sentence_text, complete_mapping, extract_article,
    make_replacer, normalize, read_json, validate,
)


class ReplacementTests(unittest.TestCase):
    def test_longest_alias_wins_and_word_boundaries_are_respected(self):
        pattern, replace = make_replacer({
            "Morty": "Talen", "Evil Morty": "Vex Talen", "Rick": "Veyran",
        })
        self.assertEqual(
            pattern.sub(replace, "Evil Morty meets MORTY, Rick's nephew and Ricky."),
            "Vex Talen meets Talen, Veyran's nephew and Ricky.",
        )

    def test_replacements_do_not_cascade(self):
        pattern, replace = make_replacer({"Rick": "Morty", "Morty": "Talen"})
        self.assertEqual(pattern.sub(replace, "Rick and Morty"), "Morty and Talen")

    def test_unicode_apostrophes_and_case_are_normalized(self):
        pattern, replace = make_replacer({"Rick's portal gun": "Veyran's Rift Spindle"})
        self.assertEqual(
            pattern.sub(replace, normalize("RICK’S\u00a0portal gun")),
            "Veyran's Rift Spindle",
        )

    def test_conflicting_aliases_fail(self):
        with self.assertRaises(ValueError):
            make_replacer({"Rick": "Veyran", "RICK": "Another person"})

    def test_short_and_full_link_labels_refer_to_the_same_entity(self):
        snapshot = {"documents": [{
            "page": "Rick Sanchez", "resolved_title": "Rick Sanchez", "name": "Veyran Kest",
            "paragraphs": [],
            "linked_terms": {"Zeep": "Zeep Xanflorp", "Zeep Xanflorp": "Zeep Xanflorp"},
        }]}
        mapping = complete_mapping(snapshot, {"Rick Sanchez": "Veyran Kest"})
        self.assertEqual(mapping["Zeep"], mapping["Zeep Xanflorp"].split()[0])

    def test_generic_unity_and_summer_are_not_character_names(self):
        pattern, replace = make_replacer(read_json(MAP_PATH))
        self.assertEqual(
            pattern.sub(replace, "Unity promises unity. Summer visits in summer."),
            "The Confluence promises unity. Neris visits in summer.",
        )

    def test_group_references_and_possessives_are_not_replaced_by_one_person(self):
        pattern, replace = make_replacer(read_json(MAP_PATH))
        self.assertEqual(
            pattern.sub(replace, "Rick's inventions helped all other Ricks and their Beths."),
            "Veyran's inventions helped all other Veyrans and their Selras.",
        )

    def test_real_corpus_aliases_keep_titles_units_and_species_consistent(self):
        pattern, replace = make_replacer(read_json(MAP_PATH))
        self.assertEqual(pattern.sub(replace, "Dr. Helen Wong"), "Dr. Ivara")
        self.assertEqual(pattern.sub(replace, "Portal-Gun and portal gun"), "Rift Spindle and Rift Spindle")
        self.assertEqual(pattern.sub(replace, "GFY, GFYs, EY"), "VCY, VCYs, VY")
        self.assertEqual(pattern.sub(replace, "Gazorpians and Gazorpian"), "Zharuuni and Zharuuni")
        self.assertNotEqual(pattern.sub(replace, "Cyberbird"), pattern.sub(replace, "Phoenixperson"))

    def test_selected_document_name_cannot_disagree_with_dictionary(self):
        snapshot = {"documents": [{
            "page": "Rick Sanchez", "resolved_title": "Rick Sanchez", "name": "Veyran Kest",
            "paragraphs": [], "linked_terms": {},
        }]}
        with self.assertRaisesRegex(ValueError, "Entity title and term map disagree"):
            complete_mapping(snapshot, {"Rick Sanchez": "An unrelated person"})

    def test_family_counterparts_keep_their_name_and_distinct_reality(self):
        pattern, replace = make_replacer(read_json(MAP_PATH))
        self.assertEqual(pattern.sub(replace, "Beth Smith (5126)"), "Selra Voss N-526")
        self.assertEqual(pattern.sub(replace, "Beth Smith Prime"), "Selra Voss of Raralo Delone")
        self.assertEqual(pattern.sub(replace, "Rick Sanchez (C-131)"), "Veyran Kest N-731")


class CleanupTests(unittest.TestCase):
    def test_cleaned_dimension_list_item_keeps_its_subject(self):
        html = '''<div class="mw-parser-output"><ul><li>
          <a href="/wiki/Dimension_C-131">C-131</a> (the world shown in the episode Solaricks.
          It is also the dimension from which the current Beth and Summer originally came).
        </li></ul></div>'''
        paragraphs, _ = extract_article(html, max_words=700)
        text = paragraphs[0]["text"]
        self.assertTrue(text.startswith("C-131 is also"))
        self.assertEqual(text.count("("), text.count(")"))

    def test_disambiguation_and_descriptive_link_labels_are_not_entities(self):
        html = '''<div class="mw-parser-output"><p>Rick considered
          <a href="/wiki/Rick_(disambiguation)">all other Ricks</a> while planning inventions for
          <a href="/wiki/Space_Beth">One of the Beths</a> and the rest of the family.</p></div>'''
        _, terms = extract_article(html, max_words=700)
        self.assertNotIn("all other Ricks", terms)
        self.assertNotIn("Rick (disambiguation)", terms)
        self.assertNotIn("One of the Beths", terms)
    def test_commentary_paragraph_does_not_leave_orphaned_author_names(self):
        self.assertEqual(clean_sentence_text(
            'In the commentary, the creators discussed the relationship. Alice and Bob wanted to reuse it.'
        ), "")
        self.assertIn("creator", clean_sentence_text(
            "The scientist asked his creator for help because the machine had stopped working."
        ))

    def test_tv_roles_and_screen_references_are_removed(self):
        self.assertEqual(clean_sentence_text("Summer is the deuteragonist of Rick and Morty."), "")
        self.assertEqual(clean_sentence_text("Her first appearance on screen was described in detail."), "")
    def test_quoted_episode_sentence_does_not_swallow_device_definition(self):
        html = '''<div class="mw-parser-output"><p>It was introduced in the episode
          "Meeseeks and Destroy." It is a gadget which creates a Mr. Meeseeks for the
          purpose of completing one objective. Once it completes the objective, Mr. Meeseeks vanishes.
        </p></div>'''
        paragraphs, _ = extract_article(html, max_words=700)
        text = paragraphs[0]["text"]
        self.assertTrue(text.startswith("It is a gadget"))
        self.assertIn("Mr. Meeseeks", text)
        self.assertNotIn("episode", text)

    def test_italic_unlinked_titles_are_collected_for_replacement(self):
        html = '''<div class="mw-parser-output"><p>In <i>Mortynight Run</i>, the travelers
          visit a large entertainment complex with several virtual reality games and machines.</p></div>'''
        _, terms = extract_article(html, max_words=700)
        self.assertIn("Mortynight Run", terms)

    def test_removes_navigation_quotes_references_and_tv_sentences(self):
        html = '''<div class="mw-parser-output">
          <aside><p>Infobox text must not appear in the resulting corpus at all.</p></aside>
          <table><tr><td>Catchphrase quoted from the television show.</td></tr></table>
          <p>The <b>Portal Gun</b> opens passages between distant locations and allows
          <a href="/wiki/Rick_Sanchez">Rick Sanchez</a> to cross different realities.<sup>[1]</sup></p>
          <p>This episode was written for the television series by its original writers.</p>
        </div>'''
        paragraphs, terms = extract_article(html, max_words=700)
        text = " ".join(p["text"] for p in paragraphs)
        self.assertIn("The Portal Gun opens", text)
        self.assertIn("Rick Sanchez", terms)
        self.assertNotIn("Infobox", text)
        self.assertNotIn("Catchphrase", text)
        self.assertNotIn("episode", text)
        self.assertNotIn("[1]", text)

    def test_skips_nested_trivia_but_resumes_at_next_main_section(self):
        html = '''<div class="mw-parser-output">
          <h2>Trivia</h2><h3>Design</h3>
          <p>Unwanted trivia describes production details and unrelated information about the original design.</p>
          <h2>History</h2>
          <p>The settlement was founded by several travelers who decided to build a new home together.</p>
        </div>'''
        paragraphs, _ = extract_article(html, max_words=700)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0]["section"], "History")
        self.assertIn("settlement", paragraphs[0]["text"])


class CorpusValidationTests(unittest.TestCase):
    @staticmethod
    def documents():
        return {f"entity_{i}.md": f"# Entity {i}\n\nRecord {i}. " + "Useful factual detail. " * 30 for i in range(30)}

    def test_marker_missing_from_dictionary_is_still_rejected(self):
        documents = self.documents()
        documents["entity_0.md"] += " A Portal-Gun opens passages."
        with self.assertRaisesRegex(ValueError, "Source-universe marker remains"):
            validate(documents, {"Unrelated": "Nexa"})

    def test_different_titles_do_not_hide_duplicate_bodies(self):
        documents = self.documents()
        documents["entity_1.md"] = documents["entity_0.md"].replace("# Entity 0", "# Entity 1")
        with self.assertRaisesRegex(ValueError, "Duplicate document"):
            validate(documents, {"Unrelated": "Nexa"})


if __name__ == "__main__":
    unittest.main()
