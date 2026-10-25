# Deploying

Two identities, on purpose.

| Identity | Holds | Used for |
|---|---|---|
| `bedrock-dev` | `AmazonBedrockFullAccess`, `AmazonS3ReadOnlyAccess` | local development and the Lambda's own runtime calls |
| `rag-deploy` | `infra/deploy-policy.json` | `terraform apply`, and nothing else |

The split is the point. The runtime identity can invoke one model and read one bucket; it cannot
create a role, push an image, or reshape the infrastructure it runs on. Giving the serving identity
deploy rights would be one leaked key away from an attacker who can rewrite the function rather
than merely call it. See `docs/decisions/007`.

`infra/deploy-policy.json` is scoped to the exact resource names in `main.tf` (`locals.name =
"rag-eval-pipeline"`) — rename there and the policy needs the same edit, or `apply` starts failing
with `AccessDenied` on the renamed resource. Account segments are wildcarded (`arn:aws:lambda:eu-central-1:*:function:...`)
because no account ID belongs in this repo; the policy is attached inside a single account, so the
wildcard does not widen it.

Two actions in it are not scoped, because AWS does not let them be. `ecr:GetAuthorizationToken`
authenticates to the registry rather than touching a repository, and `logs:DescribeLogGroups`
enumerates groups rather than reading one — scoping either to a resource ARN grants nothing at all,
silently, and the console reports it only as "actions that do not provide permissions". Both are
read-only and neither exposes log contents: reading actual events stays scoped to this project's
log group under `ReadLogsToDebugDeploys`. When adding an action here, check whether it supports
resource-level permissions before scoping it, or `terraform apply` fails on a grant that looks
correct in the file.

## One-time setup

1. **Create the user.** IAM → Users → Create user → `rag-deploy`. No console access needed.
2. **Attach the policy.** Create inline policy → JSON tab → paste `infra/deploy-policy.json`
   verbatim. IAM accepts only `Version`, `Id` and `Statement` at the top level, so do not add
   comments to it.
3. **Create an access key** for `rag-deploy` and add it as a named profile:

   ```ini
   # ~/.aws/credentials
   [rag-deploy]
   aws_access_key_id = ...
   aws_secret_access_key = ...
   ```

   Prefer a key you can delete afterwards; this identity exists to be used from one machine.
4. **Confirm the split holds:**

   ```bash
   AWS_PROFILE=rag-deploy .venv/bin/python -c \
     "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'].split('/')[-1])"
   # expect: rag-deploy
   ```
5. **Store the Langfuse keys** (optional — without them the function runs untraced). In the
   console, Systems Manager → Parameter Store → Create parameter, twice:

   | Name | Type | Value |
   |---|---|---|
   | `/rag-eval-pipeline/langfuse/public_key` | SecureString, KMS key `alias/aws/ssm` | `pk-lf-…` |
   | `/rag-eval-pipeline/langfuse/secret_key` | SecureString, KMS key `alias/aws/ssm` | `sk-lf-…` |

   The console rather than the CLI, so the secret never enters shell history, and by hand
   rather than in Terraform, so it never enters the state file. `rag-deploy` cannot write
   Parameter Store, and the function's role can read only these two names. If they are missing
   or unreadable, the function logs `tracing_disabled` with the reason and serves untraced.

## Deploying

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars   # set alert_email
export AWS_PROFILE=rag-deploy
TAG=$(git rev-parse --short HEAD)        # a running deployment traceable to a commit

terraform -chdir=infra init
terraform -chdir=infra plan      # ECR repo first: the image must exist before the function does
terraform -chdir=infra apply -target=aws_ecr_repository.app

# build and push (~2.4 GB; a dependency change re-embeds the corpus, ~9 min)
#
# --provenance=false --sbom=false is required, not tidiness: buildx otherwise
# attaches an attestation manifest, making the push an OCI image INDEX rather than
# a single manifest, and Lambda rejects that with an unhelpful "source image does
# not exist" at CreateFunction time.
docker build --provenance=false --sbom=false -t rag-eval-pipeline .
REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
.venv/bin/python -m awscli ecr get-login-password --region eu-central-1 \
  | docker login --username AWS --password-stdin "${REPO%%/*}"
docker tag rag-eval-pipeline "$REPO:$TAG"
docker push "$REPO:$TAG"

# image_tag defaults to "latest"; pass the real tag so the deploy names a commit
terraform -chdir=infra apply -var "image_tag=$TAG"
curl -s "$(terraform -chdir=infra output -raw demo_url)health"
```

`python -m awscli`, not `aws`: the console script carries a shebang pointing at an interpreter path
containing a space, so the kernel splits it and the script cannot execute at all. The `-m` form has
no shebang. Same reason `uvicorn`, `pytest` and `ruff` are all invoked that way in this project.

The two-phase apply is not ceremony: `aws_lambda_function` refers to
`"${aws_ecr_repository.app.repository_url}:${var.image_tag}"`, and Lambda validates that the image
exists at create time. A single `apply` on an empty account therefore fails partway through, having
already created the repository — recoverable, but noisier than doing it in the order the dependency
actually requires.

## Smoke test, after every deploy

Manual, on purpose: automating it would need GitHub-to-AWS credentials, which this project
decided against (`docs/decisions/016`). Five minutes. Every step checks something that has
actually broken, or was only ever verified by hand.

```bash
URL=$(AWS_PROFILE=rag-deploy terraform -chdir=infra output -raw demo_url)
```

1. **The service answers.** The first request after a deploy hits a cold container and may get
   `503` from the gateway; that is the 30s ceiling (009), not a fault. Repeat until `ok`.
   ```bash
   curl -s "${URL}health"                       # {"status":"ok"}
   ```
2. **The page shows answerable examples** — two of the original three were not (014).
   ```bash
   curl -s "$URL" | grep -o 'onclick="fill(this)">[^<]*' | cut -d'>' -f2
   ```
3. **Each example gets a real answer**, not "I don't know", with citations. Keep one request ID.
   ```bash
   curl -s -D - -H 'Content-Type: application/json' \
     -d '{"question":"What is the BenCoref dataset?","k":4}' "${URL}ask" | grep -iE 'x-request-id|answer'
   ```
4. **One request ID finds everything** — expect `START`, the `request` line, `trace_flush` with
   `"flushed": true`, `END` and `REPORT`:
   ```bash
   AWS_PROFILE=rag-deploy .venv/bin/python -m awscli logs filter-log-events --region eu-central-1 \
     --log-group-name /aws/lambda/rag-eval-pipeline --start-time $(( ($(date +%s) - 900) * 1000 )) \
     --filter-pattern '"<request-id>"' --query 'events[].message' --output text
   ```
5. **Tracing came up** — no output means the Langfuse keys loaded:
   ```bash
   AWS_PROFILE=rag-deploy .venv/bin/python -m awscli logs filter-log-events --region eu-central-1 \
     --log-group-name /aws/lambda/rag-eval-pipeline --start-time $(( ($(date +%s) - 900) * 1000 )) \
     --filter-pattern '"tracing_disabled"' --query 'events[].message' --output text
   ```
6. **The trace arrived**, with its three observations (needs the Langfuse keys locally):
   ```bash
   .venv/bin/python -c "from langfuse import Langfuse; from rag import tracing; \
   print(sorted(o.name for o in tracing.client().api.observations.get_many( \
   trace_id=Langfuse.create_trace_id(seed='<request-id>')).data))"   # ['generate', 'request', 'retrieve']
   ```

## Tearing down

```bash
AWS_PROFILE=rag-deploy terraform -chdir=infra destroy
```

ECR holds images Terraform did not create, so `destroy` can fail on a non-empty repository. Delete
the images first (`.venv/bin/python -m awscli ecr batch-delete-image`) or accept the repository
lingering at roughly €0.24/month per 2.4 GB image.

## Investigating a failed request

Every request writes one JSON line: request ID, status, `retrieve_ms` / `generate_ms` /
`total_ms`, whether it paid for a cold start, the chunk IDs retrieved, token counts, Bedrock's own
latency, retries spent, and on failure the Bedrock error code or the traceback. The same ID comes
back to the caller as `X-Request-Id`, and on Lambda it is Lambda's own request ID, so it also
matches the `REPORT` line carrying billed duration and memory.

These run as `rag-deploy`, which already holds `logs:FilterLogEvents`. `--region` is not
optional: that profile has no default region, and without it every command fails before
reaching AWS.

```bash
LOG=/aws/lambda/rag-eval-pipeline

# every server-side failure in the last hour
.venv/bin/python -m awscli logs filter-log-events --region eu-central-1 --log-group-name "$LOG" \
  --start-time $(( ($(date +%s) - 3600) * 1000 )) \
  --filter-pattern '{ $.status >= 500 }' --query 'events[].message' --output text

# one request, from the X-Request-Id a visitor reported
.venv/bin/python -m awscli logs filter-log-events --region eu-central-1 --log-group-name "$LOG" \
  --filter-pattern '"<request-id>"' --query 'events[].message' --output text
```

The second uses a plain-text match rather than `{ $.request_id = … }` on purpose: it also returns
Lambda's `START`/`END`/`REPORT` lines for the same invocation, which are not JSON.

Reading the line:

| Line says | Means |
|---|---|
| `bedrock_error: ThrottlingException`, `retries: 1` | account throttled; both attempts spent; visitor got 503 |
| `bedrock_error: ReadTimeoutError`, `generate_ms` 12000–14000 | Bedrock slow, not refusing: two 6s reads plus the backoff between them |
| `bedrock_error: AccessDeniedException`, `generate_ms` small | permission or model access, not load — retrying will not help |
| `status: 500` with `exception` | retrieval broke; the traceback is in the line |
| `cold_start: true`, large `retrieve_ms` | this request loaded the embedding model; slow but healthy |
| `generate_ms` far above `bedrock_latency_ms` | time in the client and network, not in the model |

For what the model was actually shown — the passages, the prompt, the answer — open the trace in
Langfuse. Its ID is derived from the request ID, so it needs no lookup:

```bash
.venv/bin/python -c "from langfuse import Langfuse; print(Langfuse.create_trace_id(seed='<request-id>'))"
# then https://cloud.langfuse.com/project/<project-id>/traces/<that id>
```

A `trace_flush` line follows each request: `flushed: false` means Langfuse did not accept the
trace within `RAG_TRACE_FLUSH_TIMEOUT` and that request's trace may be missing. The answer was
unaffected.

## Notes

- `aws` as a console script does not work in this checkout — the path contains a space and a
  shebang cannot express it. Use `.venv/bin/python -m awscli` if `aws` is not on PATH system-wide.
- **The public endpoint is API Gateway, not a Lambda Function URL.** Lambda resource-based
  policies are not enforced on this account, so an anonymous Function URL returns 403 whatever the
  policy says, and so does CloudFront with Origin Access Control. API Gateway invokes Lambda
  through an integration role instead, which authorises via an identity policy. There is no
  `aws_lambda_permission` in `infra/`, and adding one would be silently ineffective here. The
  reproduction is in `docs/decisions/009`.
- **`reserved_concurrent_executions` is -1**, because reserving any concurrency requires 10 to stay
  unreserved and this account's total limit *is* 10. The account limit is the ceiling instead. Set
  `max_concurrency` to a real number once that quota is raised.
- The endpoint is public on purpose — a link nobody can open proves nothing — which is why the
  per-IP rate limit and the budget alarm exist. Check the budget alarm is real before leaving the
  URL in a CV.
- **HTTP API integrations time out at 30s**, an AWS maximum. A cold `/ask` takes ~20s (it was
  ~47s before `docs/decisions/017`), close enough to the ceiling that an EventBridge schedule
  keeps one container warm; see `docs/decisions/010` for why the reranker was
  not deployed into that budget.
