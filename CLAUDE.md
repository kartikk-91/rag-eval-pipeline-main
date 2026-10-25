# rag-eval-pipeline

Grounded question answering over the ~48 NLP papers behind a master's thesis on Bengali
coreference resolution. The point of the project is not that RAG works — it is that retrieval
quality is **measured**, with a from-scratch harness (recall@k, hit-rate@k, MRR).

Plan: four releases — R1 correct+ship, R2 measure+improve, R3 operate, R4 harden.
Decisions live in `docs/decisions/`.

## Environment

- **Use `.venv`.** `myenv/` is a stale, half-built environment left over from a directory move;
  it is missing most dependencies.
- **Always invoke tools as `.venv/bin/python -m <tool>`, never `.venv/bin/<tool>`.** This project
  lives under a path containing spaces ("Full Time Job"), and a shebang line cannot express an
  interpreter path with a space in it — the kernel splits at the first one and looks for
  `/home/zenith/Full`. Every console script in the venv (`uvicorn`, `pytest`, `ruff`, `pip`, `aws`)
  is therefore unusable; the `-m` form has no shebang and always works. Moving the project to a
  path without spaces would also fix it.
- Qdrant runs in Docker: `docker compose up -d` (REST on :6333).
- Generation uses Amazon Bedrock (`eu-central-1`) and needs AWS credentials.

## Credentials

**No secrets live in this repo, and none should ever be added.**

- Credentials come from `~/.aws/credentials` (profile `default`); `~/.aws/config` sets
  `region = eu-central-1`. boto3 finds them automatically.
- The IAM user is `bedrock-dev`, carrying `AmazonBedrockFullAccess` and `AmazonS3ReadOnlyAccess`.
  **It cannot deploy** — ECR, Lambda, IAM roles, log groups, API Gateway, EventBridge and Budgets
  are all outside those two policies. `terraform apply` and `scripts/backup_corpus.py` both run as
  the separate `rag-deploy` identity, whose policy is `infra/deploy-policy.json`. See
  `infra/DEPLOY.md` and `docs/decisions/007`.
- Model ID is `eu.amazon.nova-lite-v1:0`. The `eu.` inference-profile prefix is required in this
  region; without it Bedrock reports "on-demand throughput isn't supported".
- `RAG_S3_BUCKET` (e.g. `zenith-rag-docs-2026`) switches ingestion to S3; unset, it reads `data/`.
  The bucket now holds all 48 papers and is versioned — `scripts/backup_corpus.py` keeps it in
  step. It is a backup, not the serving path: the index is baked into the image.
- **Lambda resource-based policies are not enforced on this account.** A grant made with
  `lambda:AddPermission` has no effect, which rules out a public Function URL and CloudFront OAC.
  API Gateway and EventBridge both invoke through integration roles instead. Do not add an
  `aws_lambda_permission` expecting it to work; `docs/decisions/009` has the reproduction.

Rules: never hardcode or print keys, account IDs or `~/.aws/*`. Never change the region or model ID
to work around an error — diagnose it. If credentials are invalid, stop and say so rather than
inventing a workaround.

### Verifying credentials

```bash
# identity — expect an ARN ending in user/bedrock-dev
.venv/bin/python -c "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"
# stale env vars silently override the credentials file — expect no output
env | grep -E '^AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)=' | cut -d= -f1
# a real Bedrock call — expect a grounded answer
.venv/bin/python -m rag.generate
```

Environment variables do not persist between commands here: each runs in its own process. Pass them
inline, or set them in `.claude/settings.local.json` (personal, gitignored).

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `UnrecognizedClientException`, `InvalidClientTokenId`, `SignatureDoesNotMatch` | key deleted, rotated or mistyped | new access key in the IAM console; update `~/.aws/credentials` |
| `ExpiredTokenException` | stale `AWS_SESSION_TOKEN` overriding the file | `unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN` |
| `AccessDeniedException` on Bedrock | model not enabled for the account | enable Nova Lite in the Bedrock console, `eu-central-1` |
| `ValidationException` … "on-demand throughput isn't supported" | model ID missing the `eu.` prefix | use `eu.amazon.nova-lite-v1:0` |
| `NoCredentialsError` | file missing, or running inside a container that cannot see it | run on the host |
| `Connection refused` on 6333 | Qdrant not running | `docker compose up -d` |
| `Collection 'documents' doesn't exist` | fresh Qdrant volume | `.venv/bin/python -m rag.build_index` |

## Invariants — breaking these corrupts data silently

1. **One retrieval path.** Every consumer — eval harness, API, agent, MCP tools, label helper —
   goes through `DenseRetriever.retrieve_chunks()`. `retrieve()` is derived from it so offline
   scores and production results cannot diverge. Never add a second query site.
   Guarded by `tests/test_retrieval_parity.py`. See `docs/decisions/001`.
2. **Never rename or delete files in `data/`.** `make_chunk_id()` hashes `"<filename>:<index>"`,
   so a rename silently invalidates every gold label pointing into that file. *Adding* files is
   safe. Six of the original nine filenames contain literal newlines — leave them alone.
3. **Changing `CHUNK_SIZE` / `CHUNK_OVERLAP` re-labels the eval set.** Chunk IDs survive, but the
   text behind them moves. Treat it as a re-labelling exercise, not a tweak. Any change to chunking,
   the corpus or the embedding model also needs `python -m evaluation.snapshot export` and a new
   baseline in the same PR, or the CI retrieval gate scores a stale index (`docs/decisions/016`).
4. **Gold labels are chunk IDs, not text.** The eval harness never needs document content; keep
   it decoupled from the store.
5. **Embedded Qdrant locks its directory exclusively.** Only one client at a time may hold it, so
   close one before opening another. `rag/config.make_client()` is the only place that decides
   between a server and an embedded index — never construct `QdrantClient` directly.

## Commands

```bash
docker compose up -d                        # Qdrant
.venv/bin/python -m rag.build_index         # (re)build the index — needed after corpus changes
.venv/bin/python -m evaluation.run_eval           # score retrieval
QDRANT_PATH=/tmp/ci-index .venv/bin/python -m evaluation.snapshot restore   # CI's index, locally
QDRANT_PATH=/tmp/ci-index .venv/bin/python -m evaluation.compare_baseline   # ...and CI's gate
.venv/bin/python -m pytest tests/ -q        # full suite (integration tests skip without Qdrant)
.venv/bin/python -m pytest tests/ -q -m "not integration"   # fast subset
.venv/bin/python -m uvicorn api.main:app --reload   # API (see "module form" below)
.venv/bin/python scripts/fetch_corpus.py --dry-run          # corpus, without downloading
git config core.hooksPath .githooks          # enable the shared pre-commit hook (once per clone)
.venv/bin/pip install -e ".[dev]"           # install the package plus pytest + ruff
.venv/bin/python -m ruff check .            # lint (same command CI runs)
docker build -t rag-eval-pipeline .         # image with the index baked in (~9 min: it re-embeds)
docker run -p 8000:8000 rag-eval-pipeline   # serve from the embedded index, no Qdrant needed
```

## Agent tooling

- `/eval` (`.claude/skills/eval/SKILL.md`) — runs the retrieval eval against the committed
  baseline in `evaluation/baseline.json` and reports the delta. Never update the baseline to turn a
  red run green.
- `.claude/hooks/protect-corpus.sh` — a PreToolUse hook that blocks edits to `data/*.pdf`,
  enforcing invariant 2 above. Adding files stays allowed.
- `.githooks/pre-commit` — fast tests plus a duplicate-dict-key check (which caught a real bug
  in `scripts/corpus_sources.py`). Enable with `git config core.hooksPath .githooks`.
- No subagents: nothing here justifies the coordination cost, and inventing a use would be
  decoration rather than engineering.

## Conventions

- One PR per step of the plan; squash merge, keeping the first commit message.
- Configuration lives in `rag/config.py`, never in `build_index.py` — the retriever must not
  import the indexer to learn a hostname.
- Runtime behaviour that must hold for *every* process belongs in Python, not in a shell
  entrypoint: an entrypoint only configures the one process it execs (see `docs/decisions/006`).
- `docker build` needs `data/` populated; run `scripts/fetch_corpus.py` on a fresh clone first.
- Any non-obvious decision gets `docs/decisions/NNN-*.md`:
  observation → hypothesis → decision → trade-off → measured validation.
- Claims about quality come with a measurement. No number in the README that isn't reproducible
  by `evaluation/run_eval.py`.
- Tracing (`rag/tracing.py`) is off unless `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set
  (locally: `.claude/settings.local.json` → `env`) or `LANGFUSE_SSM_PREFIX` points at Parameter
  Store (deployed). Never paste Langfuse keys into chat, code or Terraform.
- Request-path stages report through `rag.telemetry.note()` / `stage()`, which are no-ops outside
  a request. Never `print()` from code the MCP server can reach: its protocol runs over stdout,
  so telemetry logs to stderr.
- Imports that cost seconds belong inside the function that needs them — `langchain_text_splitters`
  alone costs ~11s and was being paid by the API, which never chunks anything.
