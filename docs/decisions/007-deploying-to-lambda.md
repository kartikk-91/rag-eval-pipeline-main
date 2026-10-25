> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 007 — Deploying: Lambda, a public URL, and what bounds the bill

*R1, Step 4 — 2026-10-05*

## Observation

Everything so far is verifiable only by someone who clones the repository and waits nine minutes
for an index build. The project's stated purpose is to be visible — a link someone can open.

## Decision

**One Lambda function behind a Function URL.** No API Gateway, no load balancer, no vector
database. The index is inside the image (decision 006), so between requests there is nothing
running and nothing billed.

**Mangum adapts the same FastAPI app** that `uvicorn` serves locally. One application, one code
path — a separate Lambda variant would drift from the thing the tests exercise, which is the
failure decision 001 was about.

**The URL is public** (`authorization_type = "NONE"`). An IAM-authenticated endpoint is safer and
cheaper, but a link that returns 403 to everyone who clicks it does not do the job the deploy
exists for.

## Trade-off: what actually bounds the cost

A public endpoint that calls a paid model needs a ceiling. Three layers, strongest first, and it is
worth being clear about which is which:

1. **Reserved concurrency = 2** (`infra/main.tf`). A hard AWS-enforced cap on simultaneous
   executions, so the worst case is two Bedrock calls at a time regardless of traffic. This is the
   real defence.
2. **A budget alarm at €5/month.** Not protection — a tripwire that tells you the first two layers
   were not enough.
3. **A per-IP limit of 10/minute, in the application.** Deliberately labelled best-effort: the
   window lives in process memory, so it resets on every cold start and is not shared between
   concurrent containers. It blunts one impatient client and nothing more. Real rate limiting needs
   CloudFront and WAF in front, which costs more than the thing it protects.

Also bounded: `max_tokens` 512 per answer, question length 500 characters, `k` capped at 10, and
an ECR lifecycle policy keeping three images — without which each 2.4 GB deploy would accumulate
at roughly €0.24/month forever.

**IAM is least privilege**: `bedrock:InvokeModel` on one model in one region, not `bedrock:*` on
`*`. Logs expire after 14 days.

## Two environment faults found along the way

**AWS credentials are invalid.** `~/.aws/credentials`, dated June, fails with
`UnrecognizedClientException`. So `/ask` does not currently work anywhere — nor does the LangGraph
agent, which calls Bedrock twice. No test caught this because every test stubs generation, which is
correct for tests but meant the Bedrock path had gone unexercised.

**Console scripts in the venv cannot work from this path.** `.venv/bin/uvicorn` failed with
`exec: /home/zenith/Career Center Slides/.../python3: not found`. Two faults stacked: stale paths
from a directory move, and — after repairing those — the fact that **a shebang cannot express an
interpreter path containing a space**. The kernel splits at the first one and looks for
`/home/zenith/Full`. Since the checkout is under "Full Time Job", every console script in the venv
is unusable and always was; `python -m <tool>` has no shebang and is unaffected. The README and
CLAUDE.md told people to run `uvicorn api.main:app`, which could never have worked.

## A third blocker, from the project's own history

An earlier `CLAUDE.md` records that the IAM user is `bedrock-dev`, carrying exactly
`AmazonBedrockFullAccess` and `AmazonS3ReadOnlyAccess`. Neither covers ECR, Lambda, IAM role
creation, CloudWatch log groups or Budgets — so even with working keys, `terraform apply` would
fail partway and leave a half-created stack.

Deploying needs a second identity. The cleanest split is a dedicated `deploy` user used only by
Terraform, leaving `bedrock-dev` as the minimal runtime identity it already is — which is also the
shape the Lambda execution role takes: `bedrock:InvokeModel` on one model and nothing else.

## Measured validation

```
health                        {"status":"ok"}
GET /                         <title>Ask the papers</title>
question shorter than 3 chars 422
k=99 (capped at 10)           422
rate limit, 8 requests at 5/min   500 500 500 500 500 429 429 429
429 body                      {"detail":"Rate limit is 5 questions per minute."}
```

`terraform validate` passes. 31 tests, ruff clean.

The 500s are the invalid credentials: retrieval works, generation cannot until the keys are
replaced. **Nothing has been applied to AWS.** `terraform plan` needs working credentials, and
`terraform apply` is the repository owner's to run.

## Next

Replace the credentials, `terraform plan`, push the image, apply, and put the URL in the README.

