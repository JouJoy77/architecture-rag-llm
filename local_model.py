"""Download the pinned GGUF or start an installed llama-server."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "data/llm_config.json").read_text(encoding="utf-8"))
LLM_URL = f"http://127.0.0.1:{CONFIG['port']}"


def model_path(*, offline):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=CONFIG["repository"], filename=CONFIG["filename"],
        revision=CONFIG["revision"], cache_dir=ROOT / ".cache/models",
        local_files_only=offline, token=False,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "serve"])
    parser.add_argument("--server", default="llama-server", help="Path to llama-server executable")
    parser.add_argument("--gpu-layers", type=int, default=99, help="Use 0 for CPU only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=CONFIG["port"])
    args = parser.parse_args()
    if args.threads < 1 or args.gpu_layers < 0 or not 1 <= args.port <= 65535:
        parser.error("Invalid thread count, GPU layer count or port")
    path = model_path(offline=args.action == "serve")
    if args.action == "download":
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != CONFIG["sha256"]:
            raise SystemExit("GGUF checksum mismatch")
        print(path)
        return
    command = [
        args.server, "--model", str(path), "--alias", CONFIG["model"],
        "--ctx-size", str(CONFIG["context_size"]), "--parallel", "1",
        "--threads", str(args.threads), "--threads-batch", str(args.threads),
        "--gpu-layers", str(args.gpu_layers), "--batch-size", "256", "--ubatch-size", "128",
        "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
        "--reasoning-budget", "0", "--no-context-shift",
        "--host", "127.0.0.1", "--port", str(args.port),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        parser.error("llama-server not found; install llama.cpp or pass --server /path/to/llama-server")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
