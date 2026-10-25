> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 006 — Containerising, and running Qdrant without a server

*R1, Step 4 — 2026-10-05*

## Observation

The deploy target is near-zero cost when idle. The obvious container shape — the app plus a Qdrant
service — fails that immediately: a vector database running 24/7 costs money whether or not anyone
asks a question, and this corpus is static.

## Decision: embed the index, bake it into the image

`qdrant-client` can run **embedded**, reading a directory instead of talking to a server. Measured
on the real corpus before committing to the design:

| | |
|---|---|
| index on disk | **9.9 MB** (1,688 chunks) |
| open the index | 224 ms |
| query | 20 ms |

So the index is small enough to ship inside the image. There is no database to run, nothing to pay
for while idle, and no network hop on the read path.

`rag/config.make_client()` is the single place that chooses between a server and an embedded
directory, which keeps the "one retrieval path" invariant from decision 001 intact: the same
`DenseRetriever` returned identical scores (0.739, 0.727) against both backends.

**Two build stages.** The `index` stage sees only `rag/` and `data/`, so editing `api/` or
`generate.py` leaves it cached; editing `rag/ingest.py` invalidates it, which is correct, because
changing chunking must rebuild the index. The cost of a rebuild is ~9 minutes of CPU embedding.

**CPU-only torch.** The default wheels bundle CUDA and add roughly 2 GB to an image that will never
see a GPU. Final image: **2.4 GB**, comfortably inside Lambda's 10 GB limit.

## Two failures that only running it could find

**The exclusive lock.** `build_index.main()` indexed with one client and then ran a demo query that
built a `DenseRetriever` — a second client on the same directory. Against a Qdrant server that is
fine, which is why 27 passing tests and every local run were silent about it. Embedded, the lock is
exclusive and the Docker build died with *"Storage folder /opt/index is already accessed by another
instance of Qdrant client"*. One `client.close()` fixes it; a regression test now opens, closes and
reopens an embedded index in milliseconds. A server-backed test could not have caught this.

**The entrypoint that only configured one process.** The index was first resolved by a shell
entrypoint exporting `QDRANT_PATH` before `exec`ing the app. `docker exec ... python -c "..."`
printed `QDRANT_PATH = None` and fell back to a Qdrant server that was not there.

An entrypoint sets the environment of the process it execs and nothing else. That would have broken
`docker exec`, a debugging shell, the MCP server — and, most importantly, **AWS Lambda**, which
invokes the handler through its runtime interface client rather than the image's `CMD`. The failure
would have appeared for the first time after deploying, as a connection error from a service whose
health check reported fine.

Resolution moved into `rag/config.index_path()`: Python, so it holds however the process starts.
The copy to a writable location (embedded Qdrant writes a `.lock`, and Lambda's `/var/task` is
read-only) happens there too, once, idempotently. The entrypoint script is deleted.

## Two more things fixed before shipping

**The container ran as root.** It now serves as an unprivileged `app` user. Only the runtime stage
drops privileges: the index stage needs root to write `/opt/index`, and leaving it root keeps that
expensive layer cached.

**A stale staging copy would have failed permanently.** `index_path()` copies to
`/tmp/index.<pid>` before an atomic rename, and `shutil.copytree` raises `FileExistsError` on a
target that already exists. PIDs recycle, and a reused Lambda container keeps the same one, so a
crash mid-copy would have produced the same error on every subsequent invocation rather than
clearing itself. The staging directory is now removed first.

## Trade-off

Baking the index means the image is only as fresh as its last build, and `docker build` needs the
PDFs present — `data/` is gitignored, so a clean checkout must run `scripts/fetch_corpus.py` first.
That is the price of having no database. For a corpus that changes a few times a year it is the
right trade; for one that changes hourly it would not be.

## Known gap

CI does not build this image. Doing so means downloading torch and spending nine minutes embedding
on every run, which is a poor trade for a Dockerfile that changes rarely — but it does mean a broken
Dockerfile reaches `main`. The image is built and exercised by hand before each deploy instead, and
if that stops being enough the honest fix is a scheduled build rather than one on every push.

## Measured validation

31 tests pass. ruff clean. Image 2.4 GB, container healthy 22 s after start, retriever ready in 5 s,
embedded query 20 ms.

Verified in the built image, via `docker exec` so that no entrypoint or `CMD` is involved:

```
health: {"status":"ok"}
user: app
index_path() -> /tmp/index
  0.863  moosavi2016_lea.pdf p.2
  0.848  pradhan2014_scoring.pdf p.1
```

## Next

Terraform: ECR, Lambda from this image, a Function URL, least-privilege Bedrock IAM, `eu-central-1`.

