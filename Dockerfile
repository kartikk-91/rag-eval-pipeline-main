# Two stages so that application changes do not trigger a nine-minute re-index.
#
# The `index` stage sees only configuration, ingestion and the corpus, so editing
# api/, generate.py or the agent leaves its layer cached. Editing rag/ingest.py
# correctly invalidates it — a change to chunking *should* rebuild the index.

# ---------------------------------------------------------------- deps ------
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app HF_HOME=/opt/hf
WORKDIR /app

# CPU-only torch. The default wheels carry CUDA and add roughly 2 GB to an image
# that will never see a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Only the dependency list, read out of pyproject — not the project itself. The
# application is run from PYTHONPATH, so this layer is invalidated by a
# dependency change and nothing else.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
      > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# Bake the embedding model in. Downloading it at cold start would add seconds and
# a network dependency to the first request of every deployment.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('BAAI/bge-small-en-v1.5')"

# --------------------------------------------------------------- index ------
FROM deps AS index
ENV QDRANT_PATH=/opt/index
COPY rag/ rag/
COPY data/ data/
RUN python -m rag.build_index && test -f /opt/index/meta.json

# ------------------------------------------------------------- runtime ------
FROM deps AS runtime
ENV QDRANT_BAKED_INDEX=/opt/index PYTHONUNBUFFERED=1
# The model is baked into /opt/hf above, but sentence-transformers still asks the Hub
# whether it changed on every load. With no network that cost 63s of retries against
# 17s offline; with a slow Hub it would push a cold start past the gateway's 30s. The
# image must not depend on a third party being up. Set here, not in deps: the build
# has to download the model before it can stop asking. See docs/decisions/017.
ENV HF_HUB_OFFLINE=1

COPY --from=index /opt/index /opt/index
COPY rag/ rag/
COPY api/ api/
COPY evaluation/ evaluation/
COPY mcp_server/ mcp_server/

# Drop privileges for serving. The index stage stays root because it has to write
# /opt/index; this stage only reads it, and copies to /tmp at startup.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
