"""Download both pinned models into the persistent Docker volume."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    # A separate process releases E5's RAM before downloading/checking the GGUF.
    print("Preparing the E5 revision used by the bundled index...", flush=True)
    subprocess.run([
        sys.executable, "-c",
        "from vector_index import E5Encoder, load_index; "
        "_, _, manifest = load_index(); "
        "E5Encoder(revision=manifest['embedding_model']['revision'], offline=False)",
    ], cwd=ROOT, check=True)
    print("Preparing Qwen GGUF and checking its SHA-256...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "local_model.py"), "download"],
                   cwd=ROOT, check=True)
    print("Models are ready. Start the bot with docker compose up -d --wait bot.", flush=True)


if __name__ == "__main__":
    main()
