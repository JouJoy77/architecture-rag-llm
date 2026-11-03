"""Search saved chunks; this command does not call or generate with an LLM."""

import argparse
import json

from vector_index import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    retriever = Retriever()
    try:
        results = retriever.search(args.query, args.top_k)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"Query: {args.query}\n")
        for rank, result in enumerate(results, 1):
            print(f"[{rank}] {result['score']:.4f} | {result['source']}:{result['line_start']}-{result['line_end']}")
            print(f"chunk_id: {result['chunk_id']} | sections: {', '.join(result['sections'])}")
            print(result["text"] + "\n")


if __name__ == "__main__":
    main()
