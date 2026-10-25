> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 016 — A retrieval gate without the corpus

*R4, Step 1 — 2026-10-19*

## Observation

`evaluation/compare_baseline.py` has exited non-zero on a regression since R2, written so it could
serve as the CI gate. It never did. CI had no index, so the integration tests skipped and retrieval
quality was checked only when someone remembered to run `/eval`. Every number in the README was
protected by habit.

## The options, and what each costs

| | Needs | Per-PR cost |
|---|---|---|
| Build the index in CI | the 48 PDFs, fetched from arXiv and the ACL Anthology | network-dependent, plus ~9 min of embedding |
| Read the S3 backup | GitHub-to-AWS credentials (OIDC role, trust policy) | a new identity to secure, for a demo (007) |
| **Commit a snapshot of the index** | **nothing** | **~40s** |

## Decision: snapshot chunk IDs and vectors — and no text

Gold labels are chunk IDs, not text (CLAUDE.md invariant 4). Scoring retrieval therefore needs
each chunk's ID and its vector, and nothing else. `evaluation/index_snapshot.npz` is exactly that:
1,688 IDs, 1,688 normalised 384-dim vectors, and the name of the model that produced them —
**2.4 MB, with no document text.** Leaving the text out is also a requirement in its own right:
this repository is public, and the papers are not ours to redistribute.

In CI, `python -m evaluation.snapshot restore` rebuilds an embedded Qdrant from it, and
`compare_baseline` scores the frozen eval set against `baseline.json`. The collection's shape —
distance metric, dimension, point order — now has one definition, `write_collection()`, shared by
`build_index` and the restore, so the gate cannot pass on an index production does not use.

## What it catches, and what it cannot

**Catches:** any change to query embedding, the embedding-model setting, the retriever, the Qdrant
client, or the metric code — everything between a question and a ranked list of IDs.

**Cannot catch:** a change to chunking or document embedding. Those move the text under every
chunk ID, which invariant 3 already treats as a relabelling exercise. When it happens, the
snapshot is re-exported (`python -m evaluation.snapshot export`) in the same PR as the new
baseline. A model change is refused with that instruction, not reported as a regression: query
vectors from one model against document vectors from another would score near zero, and a gate
that called that a "retrieval regression" would send someone to debug the wrong thing.

## Measured validation

- **Exact reproduction, locally and on GitHub's runner:** all 13 metrics +0.000 against the
  baseline. Different CPUs did not move a single rank, so the ±3% tolerance is headroom, not
  something already being spent on noise.
- **It fails when it should.** An injected bug that drops each question's last word before
  embedding fails 11 of 13 metrics (recall@1 0.393 → 0.250) with exit code 1.
- Four fast tests keep the snapshot honest: every gold chunk is in it, it matches the configured
  model and the baseline's chunk count, the vectors are normalised, and it contains no fields
  beyond IDs, vectors and the model name.

## What the gate does not mean

It guards *this measurement*. The baseline's intervals are ~0.25 wide and the set flatters every
retriever (015), so passing says "no worse on these 42 questions", never "good". A ±3% tolerance
on a 42-question set is also coarse: one question moving a rank at recall@1 is 0.024, inside it.
That is the trade-off of a gate that cannot cry wolf on noise, and it is why experiments are still
judged with paired intervals rather than by whether CI went green.

