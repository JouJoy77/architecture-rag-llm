"""Offline chunk/serialization checks; semantic search is tested with real E5 separately."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import faiss
import numpy as np

from vector_index import (
    BODY_TOKENS, DIMENSION, MODEL_NAME, MODEL_TOKENS, chunk_spans,
    load_index, make_chunks, sha256, word_count,
)


class TokenCounter:
    def count_tokens(self, text, *, special=True):
        return word_count(text) * 2 + (2 if special else 0)


class ChunkTests(unittest.TestCase):
    def test_sentence_punctuation_stays_with_the_previous_sentence(self):
        text = "# Instrument\n\n" + " ".join(
            f"Device number {i} " + "transfers clean energy between adjacent rooms " * 5 + "."
            for i in range(20)
        )
        spans = chunk_spans(text, "Instrument", TokenCounter())
        self.assertGreater(len(spans), 1)
        for start, _ in spans:
            self.assertNotIn(text[start], ".;,:!?")

    def test_exact_positions_coverage_limits_and_input_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            kb = root / "knowledge_base"
            kb.mkdir()
            text = "# Néra's device\n\n## History\n\n" + "\n\n".join(
                f"Record {i}. " + "This instrument transfers energy between two adjacent chambers. " * 9
                for i in range(9)
            )
            (kb / "device.md").write_text(text, encoding="utf-8")
            (root / "terms_map.json").write_text('{"secret": "must not be indexed"}')
            chunks, sources = make_chunks(TokenCounter(), kb)
            again, _ = make_chunks(TokenCounter(), kb)
            self.assertEqual(chunks, again)
            self.assertEqual(len(sources), 1)
            self.assertGreater(len(chunks), 1)
            covered = [False] * len(text)
            for chunk in chunks:
                start, end = chunk["char_start"], chunk["char_end"]
                self.assertEqual(chunk["text"], text[start:end])
                self.assertEqual(chunk["line_start"], text[:start].count("\n") + 1)
                self.assertEqual(chunk["line_end"], text[:end - 1].count("\n") + 1)
                self.assertTrue(100 <= chunk["word_count"] <= 300)
                self.assertLessEqual(chunk["input_tokens"], MODEL_TOKENS)
                self.assertNotIn("secret", chunk["text"])
                covered[start:end] = [True] * (end - start)
            self.assertTrue(all(covered[i] or char.isspace() for i, char in enumerate(text)))

    def test_short_document_is_preserved_without_padding(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            text = "# Small device\n\n" + "A factual description. " * 20
            (path / "small.md").write_text(text, encoding="utf-8")
            chunks, _ = make_chunks(TokenCounter(), path)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["text"], text)
            self.assertEqual(chunks[0]["short_reason"], "source_shorter_than_100_words")

    def test_oversized_input_fails_instead_of_silent_truncation(self):
        class ExpensiveCounter:
            def count_tokens(self, text, *, special=True):
                return len(text) * 10
        with self.assertRaises(ValueError):
            chunk_spans("# One document\n\n" + "A" * 60, "Title", ExpensiveCounter())


class StoredIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kb = self.root / "knowledge_base"
        self.kb.mkdir()
        (self.kb / "device.md").write_text("# Device\n\n" + "A factual statement. " * 20, encoding="utf-8")
        self.directory = self.root / "index"
        self.directory.mkdir()
        chunks, sources = make_chunks(TokenCounter(), self.kb)
        index = faiss.IndexFlatIP(DIMENSION)
        vectors = np.zeros((len(chunks), DIMENSION), dtype="float32")
        vectors[:, 0] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(self.directory / "faiss.index"))
        self.chunks = chunks
        (self.directory / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
        self.manifest = {
            "schema_version": 1, "sources": sources, "chunks": len(chunks), "faiss": {"vectors": len(chunks)},
            "embedding_model": {"name": MODEL_NAME, "dimension": DIMENSION, "max_input_tokens": MODEL_TOKENS,
                "passage_prefix": "passage: ", "query_prefix": "query: ", "normalized": True},
            "artifacts": {name: sha256((self.directory / name).read_bytes()) for name in ["faiss.index", "chunks.jsonl"]},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_real_faiss_round_trip_preserves_source_and_scores(self):
        index, chunks, _ = load_index(self.directory, root=self.root)
        vector = np.zeros((1, DIMENSION), dtype="float32")
        vector[0, 0] = 1.0
        scores, rows = index.search(vector, 1)
        self.assertEqual(int(rows[0, 0]), 0)
        self.assertAlmostEqual(float(scores[0, 0]), 1.0)
        self.assertEqual(chunks, self.chunks)

    def test_changed_source_requires_rebuild(self):
        with (self.kb / "device.md").open("a", encoding="utf-8") as stream:
            stream.write("New information.")
        with self.assertRaisesRegex(ValueError, "Knowledge base changed"):
            load_index(self.directory, root=self.root)

    def test_added_document_requires_rebuild(self):
        (self.kb / "new.txt").write_text("New knowledge.", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file list changed"):
            load_index(self.directory, root=self.root)

    def test_removed_document_requires_rebuild(self):
        (self.kb / "device.md").unlink()
        with self.assertRaisesRegex(ValueError, "file list changed"):
            load_index(self.directory, root=self.root)

    def test_corrupt_metadata_is_rejected_by_checksum(self):
        with (self.directory / "chunks.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_index(self.directory, root=self.root)

    def test_wrong_citation_lines_are_rejected_even_with_new_checksum(self):
        self.chunks[0]["line_start"] = 999
        file = self.directory / "chunks.jsonl"
        file.write_text("".join(json.dumps(c) + "\n" for c in self.chunks), encoding="utf-8")
        self.manifest["artifacts"]["chunks.jsonl"] = sha256(file.read_bytes())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "position does not match"):
            load_index(self.directory, root=self.root)


if __name__ == "__main__":
    unittest.main()
