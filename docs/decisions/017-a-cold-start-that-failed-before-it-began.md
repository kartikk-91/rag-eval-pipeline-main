> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 017 — A cold start that failed before it began

*R4, Step 2 — 2026-10-20*

## Observation

Running the new post-deploy smoke test against the live service, one request's log contained a
line nobody had written:

> Warning: You are sending unauthenticated requests to the HF Hub.

The embedding model is baked into the image at build time. The deployed function had no reason
to talk to Hugging Face, and it was doing so on every cold start.

Looking for the cold-start cost of that led to a second finding. Every cold start in two days of
logs reported `Init Duration` within a millisecond of **10,000ms**: 39 `INIT_REPORT` lines, every
one `Status: timeout`.

## Two causes

**Lambda's init phase is capped at 10 seconds.** If it overruns, Lambda abandons it and retries
the whole initialisation inside the first invocation, under the function timeout. The 10 seconds
are lost, not banked. Importing `api.lambda_handler` took **15.5s**, of which **13.5s** was
`rag/retriever.py` importing `sentence_transformers` (and so torch) at module scope. So every cold
start spent 10s, discarded them, and started again.

CLAUDE.md already said *"imports that cost seconds belong inside the function that needs them"*.
`rag/retriever.py` was written before that rule, and nothing enforced it.

**sentence-transformers asks the Hub whether a model changed, even when it is cached.** Measured
in the deployed image with networking disabled:

| | model load |
|---|---|
| default | **63.4s**, five retries with backoff, then falls back to the cached files |
| `HF_HUB_OFFLINE=1` | **16.9s** |

On Lambda the network is up, so this was a few round trips per cold start rather than a minute.
But it made the demo depend on a third party being reachable and fast, and a slow Hub would push
a cold start past the gateway's 30s ceiling for no reason the service could see.

## Decision

- **`HF_HUB_OFFLINE=1` in the runtime stage of the Dockerfile**, not the dependency stage, which
  has to download the model before it can stop asking. Every process in the image gets it,
  including `docker run` locally (decision 006 on where runtime behaviour belongs).
- **`sentence_transformers` imported inside `DenseRetriever.__init__`.** Importing the handler
  now costs 0.77s locally and 2.5s in the image, inside the limit. The model loads on the first
  retrieval — which the warm-up ping triggers anyway.
- **A test enforces it**: importing `api.lambda_handler` in a fresh interpreter must not load
  torch, sentence-transformers or transformers. It fails against the old import.

## Measured validation

Each change deployed and measured on its own, on the first request to a fresh deployment, which
is guaranteed to hit a cold container:

| | Lambda init | first `/ask` |
|---|---|---|
| offline only | 9,999ms, `Status: timeout` | 200 in **25.1s** |
| offline + lazy import | **2,053ms**, no timeout | 200 in **19.8s** |

The Hugging Face warning is gone from the logs. Each figure is a single cold start, since forcing
one takes a deployment; the init-phase change is not noise, as the timeout status is either there
or not. The README's earlier ~47s was measured when the service was first deployed and could not
be reproduced today without reverting both fixes, so it is quoted as history, not as a controlled
before.

## What this does not fix

A cold `/ask` is still ~20s, most of it torch and the model. A second simultaneous visitor still
waits for that, and the page's retry covers them. Taking torch out of the request path, by
embedding queries through Bedrock, is the real fix; it means re-embedding the corpus and
re-recording every baseline, and it is in the backlog rather than in v1.0.

## How it was found

Not by a test or a dashboard. By running a checklist written that morning and reading a warning
line instead of skipping it. The smoke test's step 4 asks for every log line of one request, and
the warning was in that list.

