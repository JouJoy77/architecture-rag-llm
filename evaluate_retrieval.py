"""Run six local retrieval smoke checks and save actual returned chunks."""

import json

from vector_index import ROOT, Retriever, sha256


def main():
    cases = json.loads((ROOT / "data/retrieval_questions.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    rows = []
    for case in cases:
        results = retriever.search(case["query"], top_k=5)
        relevant = [r for r in results if r["source"] == case["source"]]
        evidence = "\n".join(r["text"] for r in relevant).casefold()
        rank = next((i for i, r in enumerate(results, 1) if r["source"] == case["source"]), None)
        passed = rank is not None and all(term.casefold() in evidence for term in case["evidence"])
        rows.append({**case, "first_relevant_rank": rank, "passed": passed, "results": results})
        print(f"{case['id']}: {'PASS' if passed else 'FAIL'}; first relevant rank={rank}")
    report = {"manifest_sha256": sha256((ROOT / "vector_index/manifest.json").read_bytes()),
              "cases": len(rows), "passed": sum(r["passed"] for r in rows),
              "source_hit_at_1": sum(r["first_relevant_rank"] == 1 for r in rows) / len(rows),
              "source_hit_at_5": sum(r["first_relevant_rank"] is not None for r in rows) / len(rows), "results": rows}
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "retrieval_examples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Проверка поиска", "", "Это результаты извлечения фрагментов, без генерации ответа LLM.", "",
             f"Пройдено: {report['passed']}/{report['cases']}. Source Hit@1: {report['source_hit_at_1']:.3f}; Source Hit@5: {report['source_hit_at_5']:.3f}.", ""]
    for row in rows:
        lines += ["## " + row["id"], "", row["query"], "", f"Ожидаемый источник: `{row['source']}`. Первая позиция: {row['first_relevant_rank']}.", ""]
        for i, result in enumerate(row["results"][:3], 1):
            lines += [f"### {i}. {result['title']} — {result['score']:.4f}", "",
                      f"`{result['source']}:{result['line_start']}-{result['line_end']}`; `{result['chunk_id']}`", "", result["text"], ""]
    (directory / "retrieval_examples.md").write_text("\n".join(lines), encoding="utf-8")
    if report["passed"] != report["cases"]:
        raise SystemExit("Some retrieval checks failed; inspect reports/retrieval_examples.json")


if __name__ == "__main__":
    main()
