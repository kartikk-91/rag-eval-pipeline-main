> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 009 — What deploying taught that reading the config did not

*R1, Step 4 (completed late) — 2026-10-13*

## Observation

`infra/` had been written, reviewed and `terraform validate`-d for a week. It had never been
applied, because the AWS account was suspended. When it finally ran, it failed eight times, and
every one of the eight was invisible to inspection:

| # | Failure | Why reading the file could not catch it |
|---|---|---|
| 1 | Container ran `uvicorn`, Lambda timed out | The Dockerfile was correct *for `docker run`* |
| 2 | `CreateFunction`: "Lambda does not have permission to access the ECR image" | A resource policy that was never written cannot be reviewed |
| 3 | Function URL answered 403 to everyone | `authorization_type = "NONE"` reads as "public" |
| 4 | `reserved_concurrent_executions = 2` rejected | Depends on an account quota, not on the file |
| 5 | 60s timeout exceeded | A guess made before anything had run |
| 6 | Bedrock `AccessDenied` from Lambda, fine locally | Depends on which region Bedrock routes to |
| 7 | **Resource policies not enforced at all** | Requires an experiment, not a reading |
| 8 | Cold `/ask` 47s against a 30s gateway ceiling | Requires a deployed cold start to observe |

Numbers 1, 2 and 3 share a shape worth naming: the configuration described a system that could
never have worked, and said so nowhere.

## The one that mattered: an experiment, not a theory

The Function URL returned 403 with `authorization_type = "NONE"` and a correct `Principal: "*"`
grant. The obvious theories — an SCP, an organisation policy, Lambda's block-public-access setting
— were all checked and all wrong: the account is in no organisation, and no such setting existed on
the function.

Two approaches were then built on untested theories. CloudFront with Origin Access Control was
written, applied, and also returned 403. Only afterwards was the actual discriminating experiment
run, which took one call:

| Grant mechanism | Signed GET to `/health` |
|---|---|
| Identity policy (user's own IAM policy) | **200** |
| Resource policy → a named IAM user | **403** |
| Resource policy → `Principal: "*"`, `AuthType: NONE` | **403** |
| Resource policy → `cloudfront.amazonaws.com` | **403** |

The same principal succeeds through an identity policy and fails through `lambda:AddPermission`.
**Lambda resource-based policies are not enforced on this account.** That single fact explains the
anonymous 403, the CloudFront 403, and why neither was ever going to work.

The lesson is about ordering, not about AWS. The experiment cost one call and was available before
CloudFront was built; building first and discriminating afterwards wasted the whole attempt. The
CloudFront work was removed from the branch rather than left in history as a plausible-looking dead
end, because a reader finding it would reasonably assume it worked.

## Decision: authorise through roles, never through resource policies

API Gateway invokes Lambda with an **integration role** (`credentials_arn`), so authorisation runs
through an identity policy. EventBridge does the same. There is no `aws_lambda_permission` anywhere
in `infra/` and no Function URL, and that absence is the design — a future contributor adding one
will find it silently ineffective.

This overturns two of 007's three decisions. "One Lambda function behind a Function URL. No API
Gateway, no load balancer" was a reasonable preference for the smallest thing that works, and "the
URL is public, because a link that returns 403 to everyone does not do the job the deploy exists
for" was exactly right about the goal and wrong about the mechanism — on this account that
configuration returns 403 to everyone. The goal survives; the means do not. 007 is left standing
rather than edited, because a decisions directory that quietly rewrites superseded entries is a
record of nothing.

The underlying defect is an account-level one and worth reporting to AWS support; the reproduction
table above is the case.

## Measured validation

| | |
|---|---|
| `/ask` warm | 0.4–1.1s |
| `/ask` cold | ~47s |
| cold start, 2048 MB | 62.1s |
| cold start, 3008 MB | 18.7s |
| peak memory used | 970 MB |

Cold start is CPU-bound, not memory-bound. Peak usage is well under 2 GB, so raising memory to
3008 MB — this account's ceiling, against a normal Lambda maximum of 10240 — buys cores rather than
RAM, and cuts cold start 3.3×.

That still leaves a cold `/ask` above the 30s HTTP API integration ceiling, which is an AWS maximum
and not a setting. An EventBridge schedule every five minutes keeps one container warm, pinging the
retriever rather than `/health` — `/health` returns in 17.8s cold *because* it never loads the
model, so warming on it would warm the wrong thing. Roughly €0.25/month against ~€30 for provisioned
concurrency.

Verified live: 5 questions sampled from the frozen eval set return the gold chunk in the top 5 for
4 of them, in 0.7–1.0s each, through the public URL.

## Trade-off: the warm schedule is a mitigation, not a fix

It keeps *one* container warm. A second simultaneous visitor gets a cold one and waits, which the
demo page handles by retrying on 503 behind a "waking the model up" message rather than showing a
gateway error.

The real fix is removing torch from the request path — embedding queries through Bedrock instead of
a local `sentence-transformers` model — which would collapse the cold start and shrink the 2.4 GB
image. That is a genuine change to the retrieval path, so it belongs with the reranker work in R2
where it can be measured against the frozen set, not bolted on here.

