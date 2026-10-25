---
name: eval
description: Score retrieval against the labelled eval set and compare to the committed baseline. Use when retrieval, chunking, the corpus, or the embedding model changes, or when asked to check/report retrieval quality.
---

# Run and interpret the retrieval evaluation

The numbers in the README are a claim. This re-checks it.

## Steps

1. **Qdrant must be up.** `curl -s localhost:6333/collections` — if it fails, `docker compose up -d`.
2. **The index must match `data/`.** Compare the indexed chunk count with what ingestion
   produces now:
   ```bash
   .venv/bin/python -c "from rag.ingest import load_chunks; print(len(load_chunks()), 'chunks on disk')"
   curl -s localhost:6333/collections/documents | grep -o '"points_count":[0-9]*'
   ```
   If they differ, the index is stale — rebuild with `.venv/bin/python -m rag.build_index`
   **and say so in the report**, because a rebuild changes what every gold label points at.
3. **Compare against the baseline:**
   ```bash
   .venv/bin/python -m evaluation.compare_baseline
   ```
4. **Report** the table, and for anything outside tolerance say *why* it moved. A drop is not
   automatically a bug — expanding the corpus should lower recall@1, and that is a finding, not
   a regression to hide.

## Updating the baseline

Only when the move is understood and intended:

```bash
.venv/bin/python -m evaluation.compare_baseline --update
```

Update `evaluation/baseline.json`'s `corpus` block by hand in the same edit, update the results table in
`README.md`, and record the reason in the commit message. A baseline moved without explanation is
worse than no baseline.

## Rules

- Never update the baseline to make a red run green. That is the one thing this file exists to prevent.
- Numbers quoted anywhere (README, decision records, CV) must be reproducible by this command.
- Report the measured values, not rounded-up impressions.
