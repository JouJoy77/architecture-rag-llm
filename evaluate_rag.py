"""Save actual local LLM dialogues: five supported answers and two unknown cases."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from local_model import CONFIG, ROOT, LLM_URL
from rag_bot import LocalLLM, RAGBot, SYSTEM_PROMPT, few_shot_messages
from vector_index import Retriever, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=LLM_URL)
    parser.add_argument("--case", action="append", help="Run selected case IDs only")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rag_dialogues")
    parser.add_argument("--note", default="", help="Hardware and run conditions for the report")
    args = parser.parse_args()
    cases = json.loads((ROOT / "data/rag_questions.json").read_text(encoding="utf-8"))
    if args.case:
        known = {c["id"] for c in cases}
        if set(args.case) - known:
            parser.error("Unknown case ID")
        cases = [c for c in cases if c["id"] in args.case]
    llm = LocalLLM(args.url)
    retriever = Retriever()
    bot = RAGBot(retriever, llm)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_note": args.note,
        "llm_config": CONFIG, "endpoint": args.url,
        "server": llm.request("/v1/models"),
        "manifest_sha256": sha256((ROOT / "vector_index/manifest.json").read_bytes()),
        "cases_sha256": sha256((ROOT / "data/rag_questions.json").read_bytes()),
        "system_prompt": SYSTEM_PROMPT, "few_shot_messages": few_shot_messages(retriever.chunks),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print(f"Running {case['id']}...", flush=True)
        row = {"case": case}
        try:
            result = bot.ask(case["question"])
            answer = result["answer"]
            passed = answer["status"] == case["status"]
            if case["status"] == "answer":
                cited = {s["source_id"] for s in answer["steps"]}
                correct_sources = [s for s in result["sources"] if s["source"] == case["source"] and s["source_id"] in cited]
                evidence = "\n".join(s["text"] for s in correct_sources).casefold()
                passed = (passed and bool(correct_sources)
                          and all(term.casefold() in evidence for term in case["evidence"])
                          and all(term.casefold() in answer["answer"].casefold() for term in case["answer_terms"]))
            row.update(result=result, passed=passed)
            print(f"{case['id']}: {'PASS' if passed else 'FAIL'}; {result['total_seconds']} s", flush=True)
        except (RuntimeError, ValueError) as error:
            row.update(passed=False, error=str(error))
            print(f"{case['id']}: ERROR: {error}", flush=True)
        report["results"].append(row)
        report["passed"] = sum(r["passed"] for r in report["results"])
        report["cases"] = len(report["results"])
        args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Примеры работы RAG-бота", "",
             f"Модель: Qwen3.5-9B Q5_K_M; llama.cpp {CONFIG['tested_llama_cpp']}; thinking выключен.",
             "Каждый пример — независимый вопрос. Ответы получены от локальной модели, без ручного редактирования.",
             "Проверка: статус, ожидаемый источник и контрольные слова. Это smoke-тесты, а не полная оценка достоверности.",
             f"Пройдено: {report['passed']}/{report['cases']}.", ""]
    if args.note:
        lines += ["Условия запуска: " + args.note, ""]
    for row in report["results"]:
        lines += ["## " + row["case"]["id"], "", "**Пользователь:** " + row["case"]["question"], ""]
        if "result" in row:
            lines += ["**Бот:**", "", row["result"]["display"], "",
                      f"Время: {row['result']['total_seconds']} с; токенов промпта: {row['result']['prompt_tokens']}.", ""]
        else:
            lines += ["Ошибка: " + row["error"], ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    if report["passed"] != len(cases):
        raise SystemExit("Some cases failed; inspect the saved report")


if __name__ == "__main__":
    main()
