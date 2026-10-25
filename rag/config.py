"""Configuration shared by the indexer and the retriever.

These constants used to live in rag/build_index.py, which meant rag/retriever.py
imported the *indexer* just to learn a hostname — dragging in sentence_transformers
(~13s) and forcing a local import inside build_index.main() to dodge the resulting
cycle. Config has no heavyweight dependencies, so both sides can import it freely.

Values are overridable by environment variable so the same image can run against
a Qdrant container locally and a different endpoint in deployment.
"""

import os

COLLECTION = os.environ.get("RAG_COLLECTION", "hindi_indic_documents")
# The previous English-only embedding default performed poorly for Hindi/Devanagari.
# This multilingual model supports Hindi and has the same 384-dimensional output;
# the collection name changes to prevent accidental reuse of the old vectors.
MODEL_NAME = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VECTOR_SIZE = 384  # multilingual MiniLM output size; update with the model if changed

# Region for Bedrock and S3. Lambda sets AWS_REGION in its runtime environment, so
# the deployed function follows wherever Terraform put it; locally this falls back
# to the deploy region.
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")

# Generation model. Terraform passes this as BEDROCK_MODEL_ID (infra/main.tf) AND
# scopes the execution role's bedrock:InvokeModel to the matching inference-profile
# ARN. Reading the same variable keeps the grant and the call in step. While this was
# hardcoded the environment variable was ignored, so changing var.bedrock_model_id
# would have granted one model and called another — an AccessDenied naming neither.
#
# The "eu." prefix makes it a cross-region inference profile and is required in this
# region; without it Bedrock reports "on-demand throughput isn't supported".
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.amazon.nova-lite-v1:0")

# Cross-encoder used by CrossEncoderRetriever. Scores (query, passage) pairs jointly
# rather than comparing two independent vectors, which is why it can reorder what the
# bi-encoder ranked badly — and why it costs a forward pass per candidate.
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")

# How many candidates the reranker is given before it returns k. Not a free choice:
# recall@POOL is the ceiling on what reranking can reach, since it only reorders what
# dense retrieval already returned. Measured, not assumed — see docs/decisions/008.
RERANK_POOL = int(os.environ.get("RAG_RERANK_POOL", "50"))

# Bedrock timeouts and retries, budgeted against the 30s HTTP API integration ceiling
# (docs/decisions/009) rather than Lambda's 90s, because the gateway gives up first.
# If the budget overruns, the visitor gets the gateway's 504 instead of this service's
# 503 — losing the Retry-After and the explanation, which is the whole point of
# classifying the failure.
#
# The arithmetic has to include the backoff BETWEEN attempts, not just the reads:
#   connect + attempts x read + sum(2^i for i in 1..attempts-1)
#   2 + 2x6 + 2 = 16s, against a 30s ceiling, leaving room for a cold retrieval.
# An earlier 3/8/3 looked reasonable and came to 33s, over the ceiling it cited.
#
# One retry, not two: the budget buys either a second attempt or a longer read, and
# a second attempt is worth more against throttling. Anything beyond that becomes a
# 503 the caller retries with its own fresh budget.
BEDROCK_CONNECT_TIMEOUT = float(os.environ.get("RAG_BEDROCK_CONNECT_TIMEOUT", "2"))
BEDROCK_READ_TIMEOUT = float(os.environ.get("RAG_BEDROCK_READ_TIMEOUT", "6"))
BEDROCK_MAX_ATTEMPTS = int(os.environ.get("RAG_BEDROCK_MAX_ATTEMPTS", "2"))

# USD per 1,000 tokens (input, output), for the cost estimate on each request.
# Source: the AWS Price List API, AmazonBedrock offer for eu-central-1, usage types
# EUC1-NovaLite-input-tokens and EUC1-NovaLite-output-tokens, read 2026-10-19.
# Third-party trackers quoted three different input prices; this is the one AWS bills.
#
# Keyed by model ID on purpose: a model without an entry gets no estimate rather than
# a wrong one, so changing BEDROCK_MODEL_ID cannot silently keep reporting Nova Lite's
# price. The "eu." profile may serve from another EU region; that region's rate is
# not checked here, which is why this is called an estimate.
BEDROCK_PRICE_PER_1K = {
    "eu.amazon.nova-lite-v1:0": (0.000078, 0.000312),
}

# Langfuse tracing is off unless keys are present: LANGFUSE_PUBLIC_KEY and
# LANGFUSE_SECRET_KEY directly (local), or LANGFUSE_SSM_PREFIX naming where they live
# in SSM Parameter Store (deployed). The deployed keys are never in the environment,
# Terraform state or this repository — the function reads them once, at cold start.
LANGFUSE_SSM_PREFIX = os.environ.get("LANGFUSE_SSM_PREFIX")

# How long a request may wait for its trace to leave before Lambda freezes the
# process. Counted in the gateway budget (tests/test_failure_modes.py): a slow
# Langfuse must cost a trace, never a visitor's answer. The SDK's own export timeout
# is 5s, which on a 30s ceiling is not a number to inherit by default.
TRACE_FLUSH_TIMEOUT = float(os.environ.get("RAG_TRACE_FLUSH_TIMEOUT", "1"))

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# If set, Qdrant runs embedded against this directory instead of talking to a
# server. That is how the deployed image works: the index (9.9 MB for 1,688
# chunks) is baked in at build time, so there is no database to run or pay for.
QDRANT_PATH = os.environ.get("QDRANT_PATH")

# Embedded Qdrant writes a .lock beside its data, so the directory must be
# writable. The image bakes the index into a read-only location and names it
# here; it is copied once, on demand, to somewhere writable.
QDRANT_BAKED_INDEX = os.environ.get("QDRANT_BAKED_INDEX")
QDRANT_WRITABLE_INDEX = os.environ.get("QDRANT_WRITABLE_INDEX", "/tmp/index")


def index_path() -> str | None:
    """Where the embedded index lives, or None when a Qdrant server is in use.

    Deliberately resolved in Python rather than by a shell entrypoint exporting a
    variable: an entrypoint only sets the environment of the process it execs, so
    `docker exec`, a debugging shell, the MCP server and a Lambda handler would
    all silently fall back to looking for a server on localhost.
    """
    if QDRANT_PATH:
        return QDRANT_PATH
    if not QDRANT_BAKED_INDEX:
        return None

    import shutil

    if not os.path.isdir(QDRANT_WRITABLE_INDEX):
        staging = f"{QDRANT_WRITABLE_INDEX}.{os.getpid()}"
        # A crash mid-copy leaves this behind, and PIDs recycle — a reused Lambda
        # container keeps the same one, so the failure would repeat forever.
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(QDRANT_BAKED_INDEX, staging)
        try:
            os.rename(staging, QDRANT_WRITABLE_INDEX)
        except OSError:
            # Another process got there first; its copy is just as good.
            shutil.rmtree(staging, ignore_errors=True)
    return QDRANT_WRITABLE_INDEX


def make_client():
    """The one place that decides between a Qdrant server and an embedded index.

    qdrant_client is imported here rather than at module scope so that reading a
    hostname stays as cheap as it should be (see docs/decisions/005).
    """
    from qdrant_client import QdrantClient

    path = index_path()
    if path:
        return QdrantClient(path=path)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
