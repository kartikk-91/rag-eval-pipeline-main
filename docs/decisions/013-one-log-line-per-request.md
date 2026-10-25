> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 013 — One log line per request, before any tracing

*R3, Step 3 — 2026-10-18*

## Observation

011 classified failures correctly and left them invisible. A throttled request reached the visitor
as a 503 with `Retry-After`, and reached the operator as nothing at all: no line said which Bedrock
error caused it, how long the call took, or whether the one retry had been spent. There was no
`import logging` anywhere in `rag/`, `api/`, `agent/` or `mcp_server/`.

The only way to investigate a production failure was to reproduce it locally. For a throttle or a
timeout that is usually impossible: the condition has passed by the time anyone looks.

## Decision: one JSON line per request, built up by the stages

The API opens a record when a request arrives. Retrieval and generation add what they know with
`telemetry.note()`, and the middleware writes the record as **one** line when the response leaves:

```json
{"request_id": "…", "path": "/ask", "status": 200, "cold_start": false,
 "retrieve_ms": 69.5, "generate_ms": 433.0, "total_ms": 505.3, "bedrock_latency_ms": 341,
 "chunk_ids": ["726749c739fe", "…"], "top_score": 0.863, "input_tokens": 2209,
 "output_tokens": 10, "retries": 0, "model_id": "eu.amazon.nova-lite-v1:0", "question": "…"}
```

On failure the same line carries `bedrock_error` (the AWS code, or `ReadTimeoutError` when no
response came back to carry one) or the traceback.

**One line rather than one per stage**, so answering "what happened to request X" is a single
filter, not a join across lines that may have arrived out of order.

**The record travels in a `ContextVar`, not as an argument.** `generate_from_chunks()` keeps its
return type, and every caller outside a request (the eval harness, the batch scripts, the MCP
tools) gets a no-op without knowing telemetry exists. Starlette runs sync endpoints in a worker
thread with a *copy* of the context; the copy refers to the same dict, so what the thread notes is
what the middleware writes. A test asserts the stage timings arrive, which would fail if that
were not so.

**The request ID is Lambda's own** when there is one, and is returned to the caller as
`X-Request-Id`. A visitor's "it failed" can then arrive with the one string that finds the line,
and the same string finds Lambda's `REPORT` line with billed duration and memory. Tested through
Mangum with an HTTP API v2 event, the path production actually takes.

**`cold_start` is recorded explicitly**, as whether this request loaded the embedding model. A
5-second retrieval is healthy if it did and a fault if it did not, and the duration alone cannot
say which.

## Trade-offs

- **The question is logged; the client address is not.** The question is what makes a bad answer
  diagnosable, and this is a public demo over published papers. The address identifies a person
  and diagnoses nothing the rate limiter does not already handle. A test asserts it is absent.
  Retention is 14 days (`infra/main.tf`).
- **Chunk IDs, not chunk text.** Enough to look up exactly what the model was shown, because IDs
  are stable (invariant 2), without copying ~8 KB of corpus into the logs on every request.
- **stderr, not stdout.** The MCP server speaks its protocol over stdout, where a log line is a
  corrupted message. Lambda forwards both streams to CloudWatch, so nothing is lost by choosing the
  safe one. Tested in a subprocess, since pytest's capture would hide the difference.
- **Own handler, `propagate=False`.** The Lambda runtime installs its own handler on the root
  logger; inheriting it would print every line twice in two formats.
- **No Langfuse yet.** Tracing is the next step and builds on these same `note()` / `stage()` call
  sites. Doing it first would have meant a hosted dependency before the codebase could say, in its
  own logs, why a request failed.

## Measured validation

Ten tests in `tests/test_request_logging.py`, replaying each failure mode from
`tests/test_failure_modes.py` and asserting on the line rather than the status: the throttle names
`ThrottlingException` and its retry count, a timeout is told apart from a throttle, a permanent
failure logs at ERROR, an uncaught retrieval failure still gets its line with the traceback.
That last test passed against a bug in the first version, because it checked the log and not the
response. Review found it: the middleware re-raised, Starlette
answered with a bare `Internal Server Error`, and the header was lost — so the one failure that
most needs reporting was the one a visitor could not quote. The middleware now returns the 500
itself, ID in the header and the body.
Deleting the retry-count `note()` turns exactly the throttle test red.

Two real requests against local Qdrant and Bedrock:

| | cold | warm |
|---|---|---|
| `retrieve_ms` | 5,388 | 70 |
| `generate_ms` | 1,254 | 433 |
| `bedrock_latency_ms` | 421 | 341 |
| `total_ms` | 6,648 | 505 |

The lines answered a question nobody had asked yet. On the cold request, generation took 1,254ms
while Bedrock reports spending 421ms: **~830ms of the first call is client construction and the
TLS handshake**, not the model. On a warm request the gap is ~90ms. Neither number existed before
this change.

Not yet verified: the `filter-log-events` queries in `infra/DEPLOY.md` against the deployed
function's log group. That waits for the next deploy.

