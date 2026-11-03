"""Build a local multilingual-e5-base + FAISS index from knowledge_base only."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import faiss

from vector_index import (
    ROOT, INDEX_DIR, MODEL_NAME, MODEL_REVISION, DIMENSION, MODEL_TOKENS, BODY_TOKENS,
    TARGET_WORDS, OVERLAP_WORDS, E5Encoder, make_chunks, passage, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached model files only")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    start = perf_counter()
    encoder = E5Encoder(offline=args.offline, threads=args.threads)
    model_loaded = perf_counter()
    chunks, sources = make_chunks(encoder)
    chunked = perf_counter()
    print(f"{len(sources)} documents -> {len(chunks)} chunks", flush=True)
    vectors = encoder.encode([passage(c["title"], c["text"]) for c in chunks], batch_size=args.batch_size, progress=True)
    embedded = perf_counter()
    faiss.omp_set_num_threads(args.threads)
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vectors)
    (ROOT / ".cache").mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="index-build-", dir=ROOT / ".cache") as temp:
        from pathlib import Path
        staging = Path(temp)
        faiss.write_index(index, str(staging / "faiss.index"))
        (staging / "chunks.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8",
        )
        written = perf_counter()
        manifest = {
            "schema_version": 1, "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": {"name": MODEL_NAME, "url": f"https://huggingface.co/{MODEL_NAME}",
                "revision": MODEL_REVISION, "dimension": DIMENSION, "max_input_tokens": MODEL_TOKENS,
                "passage_prefix": "passage: ", "query_prefix": "query: ", "normalized": True},
            "faiss": {"type": "IndexFlatIP", "metric": "cosine similarity on L2-normalized vectors", "vectors": index.ntotal},
            "chunking": {"target_words": TARGET_WORDS, "overlap_words": OVERLAP_WORDS,
                "max_body_tokens": BODY_TOKENS, "min_words": min(c["word_count"] for c in chunks),
                "max_words": max(c["word_count"] for c in chunks), "max_input_tokens": max(c["input_tokens"] for c in chunks),
                "short_chunks": [{"source": c["source"], "reason": c["short_reason"]} for c in chunks if "short_reason" in c]},
            "sources": sources, "documents": len(sources), "chunks": len(chunks),
            "timings_seconds": {"model_load": round(model_loaded - start, 3), "chunking": round(chunked - model_loaded, 3),
                "embedding": round(embedded - chunked, 3), "index_write": round(written - embedded, 3), "total": round(written - start, 3)},
            "runtime": {"device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
                "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                "packages": {name: version(name) for name in ["torch", "sentence-transformers", "transformers", "langchain-text-splitters", "faiss-cpu", "numpy"]}},
            "artifacts": {name: sha256((staging / name).read_bytes()) for name in ["faiss.index", "chunks.jsonl"]},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Manifest is published last; readers reject partial/mixed versions by hash.
        for name in ["faiss.index", "chunks.jsonl", "manifest.json"]:
            os.replace(staging / name, INDEX_DIR / name)
    print(json.dumps({"documents": len(sources), "chunks": len(chunks), "dimension": DIMENSION,
                      "timings_seconds": manifest["timings_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
