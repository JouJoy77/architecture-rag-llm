"""Check llama.cpp readiness without loading another copy of either model."""

import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "data/llm_config.json").read_text(encoding="utf-8"))
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{config['port']}/health", timeout=3) as response:
        status = json.load(response)
    if status.get("status") != "ok":
        raise SystemExit("llama-server is not ready")


if __name__ == "__main__":
    main()
