> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 014 — Tracing that cannot cost an answer

*R3, Step 4 — 2026-10-19*

## Observation

013 made a failure diagnosable: one log line per request names the error, the time spent and the
retries used. It deliberately records chunk **IDs**, not text, so it cannot show what the model was
actually given. For a wrong answer rather than a failed one, that is the question: which passages
did it see, what was the prompt, what did it say back?

## Decision: Langfuse, joined to the log line by key

Each `/ask` becomes one trace with three observations: `request` (question in, answer out),
`retrieve` (the passages, with text and citations), and `generate` (the full prompt, the answer,
tokens and cost).

**The trace ID is derived from the request ID** (`create_trace_id(seed=request_id)`). The
`X-Request-Id` a visitor quotes finds the log line, Lambda's `REPORT` line and the trace, with no
table mapping one to another and nothing extra to store.

**The same call sites feed both.** Tracing is nested beside `telemetry.stage()` in
`rag/generate.py`, not added as a parallel set of hooks, and cost is computed once and handed to
both. A trace and a log line that disagree about what a request cost would make both worthless.

**Off unless configured.** With no keys, `langfuse` is never imported (a test checks
`sys.modules` in a subprocess), and the eval harness, batch scripts and MCP tools are unchanged.

**Only `/ask` is traced.** A page load or health check has nothing to show; tracing them would
spend quota to bury the questions.

## The trade-off that shaped it: the visitor waits for the flush

The SDK exports from a background thread. Lambda freezes the process the moment the handler
returns, so a trace still queued then is late at best and lost at worst — Langfuse's own guidance
is an explicit `flush()` before returning. But the response is not released until the handler
returns, so **the visitor waits through that flush**, on the same 30s gateway clock as everything
else.

The SDK's export timeout defaults to **5s**. Inherited silently, a slow Langfuse would add up to
5s to every answer, and on a throttled request could push a Bedrock call that finished in time past
the gateway — the 011 budget bug again, in a new place.

So the flush runs in a thread joined with a **1s cap**, and returns whether it finished. The cap is
in `rag/config.py` and **counted in the gateway budget test**, which now reads
2 + 2×6 + 2 + 1 = 17s against 24s. A slow Langfuse costs that request its trace, logged as
`flushed: false`, never its answer.

## Trade-offs

- **Data leaves AWS.** The question, passages and answer go to Langfuse Cloud's EU region. The
  passages are from published papers; the client address is not sent, as it is not logged (013).
- **Keys in Parameter Store, created by hand.** Not environment variables (visible in the Lambda
  console), not Terraform variables (written to the state file). The function's role may read
  exactly two parameter names, once per cold start. If that fails, it logs `tracing_disabled`
  with the reason and serves untraced — a test asserts the service survives it.
- **Cold-start cost, moved off the visitor.** Importing `langfuse` takes ~0.8s, plus one Parameter
  Store call. The warm-up ping now builds the client, so on the warmed container neither lands on
  a visitor. Review of this PR found the Parameter Store call had no timeout at all — boto3's
  defaults are 60s connect plus 60s read, with retries — so on an unwarmed container a hanging
  Parameter Store would have taken the answer down with the trace. It is now bounded at
  (2 + 2) × 2 = 8s worst case, with a test, and failure turns tracing off.
- **Cost is an estimate.** Nova Lite's list price in `eu-central-1`, from the AWS Price List API:
  $0.078 / 1M input, $0.312 / 1M output. Three third-party trackers gave three different input
  prices; this is the one AWS bills in Frankfurt. The `eu.` inference profile may serve from
  another EU region, whose rate is not checked, so the number is called an estimate. A model with
  no entry gets no estimate rather than Nova Lite's.

## Measured validation

Fifteen test cases in `tests/test_tracing.py` against a **real** Langfuse client exporting to memory,
so the assertions are on the spans Langfuse would receive. Two deliberate breakages — not marking
a failed root as ERROR, not attaching cost to the generation — each turn exactly one test red.

Three real questions through local Qdrant and Bedrock, traced to Langfuse Cloud, then fetched back
from the API by the ID derived from each request ID. All three were found with the expected
structure, and the trace agreed with the log line written for the same request:

| | trace | log line |
|---|---|---|
| cost | $0.00017511 | $0.00017511 |
| generation latency | 407ms | 407.4ms |
| total | 461ms | 462.5ms |
| tokens | 2,209 in / 9 out | 2,209 / 9 |

Flush, measured from a laptop in Bielefeld: **234ms** for the first (connection setup), then
**74–81ms**. All finished inside the cap.

Per-question cost across the three: **$0.00018–0.00023**, about $1 per 5,000 questions.

### Deployed

Deployed as `e97e623`, the first deploy since `9effa5d`, so it shipped 013's logging as well.
Four questions against the live URL, then checked from both ends:

- **Keys decrypted.** No `tracing_disabled` line. This account had already shown one class of
  resource policy going unenforced (009), so the AWS-managed key's policy was not assumed to work;
  it does.
- **Flush from Lambda: 57–115ms**, every one `flushed: true`. A request with nothing queued
  flushes in 0.2ms.
- **All four traces arrived**, fetched back from Langfuse by the ID derived from each Lambda
  request ID, with all three observations and a cost of $0.00015–0.00023.
- **One ID finds everything.** Filtering CloudWatch on one request ID returns Lambda's `START`,
  the request line, the flush line, `END`, and `REPORT` (955ms billed, 1,001 of 3,008 MB used).

Running those queries found a documentation bug: `rag-deploy` has no default region, so every
command as first written in `infra/DEPLOY.md` failed before reaching AWS. Fixed.

The traces earned their keep on the first four requests. Two answers were *"I don't know."* and
*"You don't know."* — to two of the three example questions on the demo page. The log line could
only have said the call succeeded; the trace shows what the model was given and why it declined.

