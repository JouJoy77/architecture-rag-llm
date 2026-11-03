# syntax=docker/dockerfile:1
# The pinned Ubuntu binary needs a recent glibc; Debian trixie provides it.
FROM python:3.11-slim-trixie AS llama
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64 || (echo 'This image requires linux/amd64' >&2; exit 1)
ADD --checksum=sha256:5a51cc921ed25f4c0c8284c6d147985f5cebcabc66721932ea2a13e983b56750 https://github.com/ggml-org/llama.cpp/releases/download/b12058/llama-b12058-bin-ubuntu-x64.tar.gz /tmp/llama.tar.gz
RUN mkdir /opt/llama && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1

FROM python:3.11-slim-trixie
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/.cache/models/hub-meta \
    HF_XET_CACHE=/app/.cache/models/xet \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libstdc++6 libssl3t64 libzstd1 zlib1g tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home rag
COPY --from=llama /opt/llama /opt/llama
WORKDIR /app
COPY requirements-data.txt requirements-index.txt ./
RUN python -m pip install -r requirements-index.txt && python -m pip check
COPY *.py terms_map.json ./
COPY data/ data/
COPY knowledge_base/ knowledge_base/
COPY vector_index/ vector_index/
COPY tests/ tests/
COPY docker/ docker/
RUN mkdir -p /app/.cache/models /app/reports \
    && chown -R rag:rag /app/.cache /app/reports \
    && /opt/llama/llama-server --version \
    && python -m unittest discover -s tests -v \
    && python prepare_knowledge_base.py --check \
    && python -c 'from vector_index import load_index; load_index()'
USER rag
# Forward SIGTERM to Python and its llama-server child, and reap child processes.
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["python", "local_model.py", "serve", "--server", "/opt/llama/llama-server", "--gpu-layers", "0"]
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=12 \
    CMD ["python", "docker/healthcheck.py"]
