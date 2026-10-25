> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 011 — Failures that say whether retrying helps

*R3, Step 1 — 2026-10-16*

## Observation

`api/main.py:ask()` called `answer_question()` with no `try`/`except`, and `rag/generate.py` built
its Bedrock client with no timeout and no retry policy. Any throttle, transient fault or malformed
response became an unhandled exception and `Internal Server Error` on the public URL.

Every other part of this project measures what it claims. The request path asserted nothing, and
it is the only part a visitor ever sees.

## Decision: split failures by the one thing a caller can act on

Not by HTTP status, not by AWS error code — by **whether trying again could work**.
`rag/errors.py` has exactly two types, and the AWS vocabulary stops there: nothing above `rag/`
imports botocore to decide whether a request is worth repeating.

| Failure | Response | Why |
|---|---|---|
| Throttling, timeout, service fault | **503** + `Retry-After` | The demo page already retries on 503 behind "waking the model up", built for cold starts; a throttled request lands in that path instead of an error box |
| Permission, validation, model not enabled | **502** | This service is working and its upstream is not. A 500 blames the wrong party and says nothing about retrying |
| Retrieval returned nothing | **200**, "I could not find anything in the indexed papers" | Grounding is this project's whole claim, and handing the model an empty context invites it to invent an answer |

**Retrieval failures are deliberately not caught.** A broken index is a deployment fault; masking
it as a polite answer would hide the one thing worth paging about.

## Trade-off: the budget belongs to the gateway, not to Lambda

The function's timeout is 90s. It is unreachable through the public path, because an HTTP API
integration is cut off at **30s** and that is an AWS maximum rather than a setting (009). A read
timeout longer than the gateway's patience is a timeout nobody waits for.

That is also where this went wrong first. The original setting — connect 3, read 8, three
attempts — shipped with a comment asserting it "must leave room for retrieval and still land
inside 30s". It does not:

    3 + 3x8 + (2 + 4 backoff) = 33s

The backoff **between** attempts was not counted. The failure this change exists to handle would
have been cut off by the gateway and reached the visitor as a bare 504, without the `Retry-After`
or the explanation — correct classification, invisible. Now connect 2, read 6, one retry: 16s,
leaving most of the budget for a cold retrieval on the same clock.

One retry rather than two because the budget buys either a second attempt or a longer read, and
against throttling the second attempt is worth more. Anything beyond that becomes a 503 the caller
retries with a fresh budget of its own.

`retries={"mode": "standard"}` rather than `"adaptive"`: adaptive adds client-side rate limiting
tuned for sustained throughput, which is the wrong shape for a service that is idle and then
answers one question.

## Measured validation

15 fault-injection tests, no AWS and no model: each retryable and non-retryable error code,
connect and read timeouts, a 200 with the wrong body shape, the empty-context path asserting the
model is never called, and three through FastAPI's `TestClient` checking a visitor actually sees
503 with `Retry-After`, 502, or a working answer.

Deployed and verified live: `/ask` answers in **0.57–0.81s** warm, unchanged by the new handling.

Two constraints are now tests rather than comments, because a comment claiming a budget fits is
worth nothing — one did, and it was wrong:

- the retry budget must fit inside the integration timeout, read from `infra/main.tf`
- the model the code calls must be the model the IAM policy grants, read from `infra/variables.tf`

The second came from a related drift found the same day: `rag/generate.py` hardcoded `REGION` and
`BEDROCK_MODEL_ID` while Terraform passed `BEDROCK_MODEL_ID` as an environment variable and scoped
`bedrock:InvokeModel` to the matching ARN. The values were identical, so nothing was broken — but
changing the Terraform variable would have granted one model and called another, and Bedrock would
have answered `AccessDenied` naming neither. Both now live in `rag/config.py`.

## What this does not do

No circuit breaker, no idempotency key, no structured logging. A 503 is observable only as a
status code: nothing records which error code caused it, how long Bedrock took, or how many
retries were spent. That is the next milestone, and it is the reason this one stops here — there
is no `import logging` anywhere in `rag/`, `api/`, `agent/` or `mcp_server/` to build tracing on.
Addressed in 013.

