"""Local RAG: E5 -> FAISS -> grounded few-shot prompt -> Qwen -> cited answer."""

import argparse
import json
import re
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from local_model import CONFIG, LLM_URL
from rag_security import PRE_PROMPT, PROTECTIONS, contains_sensitive, prepare_sources, risk_reasons
from vector_index import Retriever, sha256


BASE_PROMPT = """Ты справочный помощник по вымышленному миру базы знаний. Отвечай по-русски.
Используй только факты из CONTEXT текущего вопроса. Знания модели и контекст примеров
не являются источниками для нового ответа.
Сначала проверь, достаточно ли фактов для всех частей вопроса. Затем верни короткое
объяснение ответа: 1–3 проверяемых шага (steps), каждый с точной цитатой (quote),
идентификатором источника (source_id) и кратким описанием подтверждённого факта (fact).
Это объяснение по источникам, без внутреннего монолога и догадок. Цитата — дословный
непрерывный фрагмент документа, 12–300 символов, без многоточий и перевода.
После проверки сформулируй краткий answer. Имена сохраняй как в документах.
Не добавляй в answer ссылки: программа подставит их по steps.
Если ответа нет, источники противоречат друг другу или вопрос требует отсутствующих
данных, верни status="unknown", answer="Я не знаю.", steps=[]. Не подменяй вопрос
похожим фактом и не выдавай приблизительное предположение за ответ.
Формат — JSON: status (answer/unknown), затем steps (массив), затем answer (строка).
"""
SYSTEM_PROMPT = PRE_PROMPT + "\n" + BASE_PROMPT


def system_prompt(mode):
    return (PRE_PROMPT + "\n" if PROTECTIONS[mode].pre_prompt else "") + BASE_PROMPT

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answer", "unknown"]},
        "steps": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "source_id": {"type": "string"},
                "quote": {"type": "string", "minLength": 12, "maxLength": 300},
                "fact": {"type": "string", "minLength": 1, "maxLength": 350},
            }, "required": ["source_id", "quote", "fact"],
        }},
        "answer": {"type": "string", "minLength": 1, "maxLength": 1200},
    }, "required": ["status", "steps", "answer"],
}


class LLMError(RuntimeError):
    pass


class AnswerValidationError(ValueError):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LLMError("Redirects from local LLM are not allowed")


class LocalLLM:
    def __init__(self, url=LLM_URL, *, timeout=600):
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError(f"Use a local llama-server URL, e.g. {LLM_URL}")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirects())
        props = self.request("/props")
        self.context_size = min(CONFIG["context_size"], props["default_generation_settings"]["n_ctx"])

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            digest = sha256(error.read(2048))
            raise LLMError(f"llama-server HTTP {error.code} at {path}; response_sha256={digest}") from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise LLMError(f"Local llama-server unavailable or invalid response: {error}") from error

    def count_prompt_tokens(self, messages):
        prompt = self.request("/apply-template", {
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False},
        })["prompt"]
        result = self.request("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})
        return len(result["tokens"])

    def generate(self, messages):
        response = self.request("/v1/chat/completions", {
            "model": CONFIG["model"], "messages": messages, "stream": False,
            "max_tokens": CONFIG["max_output_tokens"], "temperature": 0.0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object", "schema": ANSWER_SCHEMA},
        })
        try:
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise LLMError(f"LLM output incomplete: {choice['finish_reason']}")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMError("LLM returned empty content")
            return content, response.get("usage", {})
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("Unexpected llama-server completion response") from error


def context_message(question, sources):
    context = [{"source_id": s["source_id"], "title": s["title"], "text": s.get("context_text", s["text"])} for s in sources]
    return json.dumps({"CONTEXT": context, "QUESTION": question}, ensure_ascii=False)


def few_shot_messages(chunks):
    # Use a verified chunk from the saved corpus, not an invented document.
    source = next((c for c in chunks if c["source"] == "knowledge_base/taskling_cradle.md"
                   and "It is a gadget which creates a Taskling for the purpose of completing one objective." in c["text"]), None)
    if source is None:
        raise ValueError("Few-shot source missing or changed; update the examples")
    source = {**source, "source_id": "E1"}
    positive = {
        "status": "answer",
        "steps": [{"source_id": "E1",
                   "quote": "It is a gadget which creates a Taskling for the purpose of completing one objective.",
                   "fact": "В описании Taskling Cradle указано создание Taskling для одной задачи."}],
        "answer": "Taskling Cradle создаёт Taskling для выполнения одной задачи.",
    }
    return [
        {"role": "user", "content": context_message("Какое устройство создаёт Taskling для одной задачи?", [source])},
        {"role": "assistant", "content": json.dumps(positive, ensure_ascii=False)},
        {"role": "user", "content": context_message("Какова масса Taskling Cradle в килограммах?", [source])},
        {"role": "assistant", "content": json.dumps({"status": "unknown", "steps": [], "answer": "Я не знаю."}, ensure_ascii=False)},
    ]


def validate_answer(raw, sources):
    def fail(message):
        raise AnswerValidationError(message)

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AnswerValidationError("LLM did not return JSON") from error
    if not isinstance(value, dict) or set(value) != {"status", "answer", "steps"}:
        fail("Unexpected answer fields")
    if not isinstance(value["answer"], str) or not 1 <= len(value["answer"].strip()) <= 1200:
        fail("Invalid answer text")
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) > 3:
        fail("Invalid explanation steps")
    if value["status"] == "unknown":
        if value["answer"] != "Я не знаю." or steps:
            fail("An unknown answer must have no unsupported claims")
        return value
    if value["status"] != "answer" or not steps:
        fail("An answer needs cited evidence")
    by_id = {s["source_id"]: s for s in sources}
    normalize = lambda text: " ".join(text.split())
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"source_id", "quote", "fact"}:
            fail("Invalid evidence fields")
        if not all(isinstance(v, str) for v in step.values()) or step["source_id"] not in by_id:
            fail("Citation is outside the current retrieved context")
        if not 12 <= len(step["quote"]) <= 300 or not 1 <= len(step["fact"].strip()) <= 350:
            fail("Invalid evidence length")
        quote = normalize(step["quote"])
        if len(quote) < 12 or quote not in normalize(by_id[step["source_id"]]["text"]):
            fail("Quoted evidence is absent from its source")
        if quote not in normalize(by_id[step["source_id"]].get("context_text", by_id[step["source_id"]]["text"])):
            fail("Quoted evidence was removed from the LLM context")
    # References are supplied by the application, not generated file paths.
    if re.search(r"\[[A-Z]\d+\]", value["answer"]):
        fail("Answer text must not invent inline citations")
    return value


def render_answer(answer, sources):
    if answer["status"] == "unknown":
        return answer["answer"]
    lines = ["Проверка по источникам:"]
    for number, step in enumerate(answer["steps"], 1):
        lines.append(f"{number}. {step['fact']} [{step['source_id']}]\n   «{step['quote']}»")
    ids = list(dict.fromkeys(step["source_id"] for step in answer["steps"]))
    lines += ["", "Ответ: " + answer["answer"] + " " + " ".join(f"[{i}]" for i in ids), "", "Источники:"]
    for source in sources:
        if source["source_id"] in ids:
            lines.append(f"[{source['source_id']}] {source['title']} — {source['source']}:{source['line_start']}-{source['line_end']}; {source['chunk_id']}")
    return "\n".join(lines)


class RAGBot:
    def __init__(self, retriever, llm, *, top_k=3, protection="full", min_score=0.75):
        if not 1 <= top_k <= len(retriever.chunks):
            raise ValueError("Invalid top_k")
        self.retriever, self.llm, self.top_k = retriever, llm, top_k
        if protection not in PROTECTIONS or (min_score is not None and not -1 <= min_score <= 1):
            raise ValueError("Invalid protection mode or minimum cosine score")
        self.protection, self.min_score = protection, min_score
        example_chunks = retriever.chunks
        if PROTECTIONS[protection].filter_chunks:
            example_chunks = [c for c in example_chunks if not risk_reasons(c["title"] + "\n" + c["text"])]
        self.examples = few_shot_messages(example_chunks)

    def ask(self, question):
        if not question.strip():
            raise ValueError("Question must not be empty")
        start = perf_counter()
        retrieved = self.retriever.search(question, self.top_k)
        retrieval_seconds = perf_counter() - start
        eligible = [c for c in retrieved if self.min_score is None or c["score"] >= self.min_score]
        sources, dropped, sanitized = prepare_sources(eligible, PROTECTIONS[self.protection])
        sources = [{**c, "source_id": f"S{i}"} for i, c in enumerate(sources, 1)]
        security = {"mode": self.protection, "retrieved": len(retrieved),
                    "below_threshold": len(retrieved) - len(eligible), "dropped": dropped,
                    "sanitized": sanitized, "llm_called": False, "output_blocked": False}

        def unknown(reason, raw_hash=None, *, usage=None, prompt_tokens=0):
            answer = {"status": "unknown", "steps": [], "answer": "Я не знаю."}
            # No rejected document bodies, raw generations or secret-bearing metadata.
            audit = {**security, "reason": reason}
            if reason == "output_blocked":
                audit["dropped"] = [{"rules": item["rules"]} for item in dropped]
                audit["sanitized"] = []
            return {"question": question, "answer": answer, "display": "Я не знаю.", "sources": [],
                    "raw_output": None, "raw_output_sha256": raw_hash, "usage": usage or {}, "prompt_tokens": prompt_tokens,
                    "retrieval_seconds": round(retrieval_seconds, 3),
                    "total_seconds": round(perf_counter() - start, 3), "security": audit}

        if not sources:
            result = unknown("empty_search" if not retrieved else "below_threshold" if not eligible else "filtered_context")
            if PROTECTIONS[self.protection].check_output and contains_sensitive(json.dumps(result["security"], ensure_ascii=False)):
                security["output_blocked"] = True
                return unknown("output_blocked")
            return result
        # Count with Qwen's tokenizer and template, not E5's tokenizer.
        while sources:
            messages = [{"role": "system", "content": system_prompt(self.protection)}, *self.examples,
                        {"role": "user", "content": context_message(question, sources)}]
            prompt_tokens = self.llm.count_prompt_tokens(messages)
            if prompt_tokens + CONFIG["max_output_tokens"] + 32 <= self.llm.context_size:
                break
            sources.pop()
        if not sources:
            raise ValueError("Question and context exceed the LLM budget; shorten the question")
        security["llm_called"] = True
        raw, usage = self.llm.generate(messages)
        if PROTECTIONS[self.protection].check_output and contains_sensitive(raw):
            security["output_blocked"] = True
            return unknown("output_blocked", sha256(raw.encode()), usage=usage, prompt_tokens=prompt_tokens)
        answer = validate_answer(raw, sources)
        security["reason"] = "answer" if answer["status"] == "answer" else "model_unknown"
        result = {
            "question": question, "answer": answer, "display": render_answer(answer, sources),
            "sources": sources, "raw_output": raw, "usage": usage, "prompt_tokens": prompt_tokens,
            "prompt_sha256": sha256(json.dumps(messages, ensure_ascii=False).encode()),
            "retrieval_seconds": round(retrieval_seconds, 3), "total_seconds": round(perf_counter() - start, 3),
            "security": security,
        }
        if PROTECTIONS[self.protection].check_output:
            public = {k: v for k, v in result.items() if k != "question"}
            if contains_sensitive(json.dumps(public, ensure_ascii=False)):
                security["output_blocked"] = True
                return unknown("output_blocked", sha256(raw.encode()), usage=usage, prompt_tokens=prompt_tokens)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Omit to start the REPL")
    parser.add_argument("--url", default=LLM_URL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--protection", choices=PROTECTIONS, default="full")
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--json", action="store_true", help="JSON output for a single question")
    args = parser.parse_args()
    if args.json and args.question is None:
        parser.error("--json requires a question")
    try:
        llm = LocalLLM(args.url)
        bot = RAGBot(Retriever(), llm, top_k=args.top_k, protection=args.protection, min_score=args.min_score)
        if args.question is not None:
            result = bot.ask(args.question)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["display"])
            return
        print("RAG-бот готов. Каждый вопрос независим. Выход: /exit или Ctrl+D.")
        while True:
            question = input("\nВы: ").strip()
            if question.lower() in {"/exit", "/quit"}:
                break
            if not question:
                continue
            try:
                print("\nБот: " + bot.ask(question)["display"])
            except (LLMError, ValueError) as error:
                print(f"Ошибка: {error}")
    except (EOFError, KeyboardInterrupt):
        print()
    except (LLMError, ValueError, OSError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
