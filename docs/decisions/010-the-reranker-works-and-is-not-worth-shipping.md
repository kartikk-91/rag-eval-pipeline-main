> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 010 — The reranker works, and is not worth shipping

*R2, Steps 4–6 — 2026-10-15*

## Hypothesis, written before measuring

Recorded in 008 and in `CrossEncoderRetriever`'s docstring before the first run: recall@1 is
0.393 while recall@20 is 0.869, so most gold material is already retrieved and merely ordered
badly. A bi-encoder compares two vectors built independently; a cross-encoder reads query and
passage together. Reordering is exactly the lever that gap calls for, so a cross-encoder should
help.

The counterweight was recorded at the same time: it can only reorder what dense retrieval already
returned, so recall@pool is a hard ceiling, and it costs a forward pass per candidate.

## What happened: it works, and more candidates make it worse

`bge-reranker-base`, 42 frozen questions, paired bootstrap on per-query differences.

| pool | recall@10 after rerank | dense ceiling (recall@pool) | ms/query |
|---|---|---|---|
| 10 | 0.774 | 0.774 | 8,309 |
| **20** | **0.869** | **0.869** | 15,781 |
| 50 | 0.857 | 0.917 | 41,052 |
| 100 | 0.821 | 0.940 | 84,693 |

At pool 20 the reranker is **perfect**: 0.869 is exactly dense recall@20, so every gold chunk
reachable in the top 20 was pulled into the top 10. The hypothesis is confirmed as strongly as this
set can confirm anything — ranking *was* the bottleneck, and reranking fixes ranking.

Then it degrades. At pool 50 it reaches 0.857 against a ceiling of 0.917; at pool 100, 0.821
against 0.940. The ceiling rises and the achieved value falls. More candidates means more
plausible-but-wrong passages, and the cross-encoder promotes some of them over gold ones. "Give the
reranker more to work with" is wrong here, and only a sweep shows it.

The single interval clearing zero in the whole sweep:

    pool 20, recall@10:  +0.095   95% CI [+0.024, +0.190]

## Why this is still read as a negative result

**The evidence is thinner than that one line looks.** Twenty intervals were computed (5 metrics ×
4 pools) at 95%, so roughly one false positive is expected by chance, and exactly one appeared.
What argues for a real effect is not that interval but the sign pattern: essentially every delta at
every pool is positive. A consistent small improvement that 42 questions cannot resolve
individually — which is an argument for more questions, not for declaring victory.

**It is not deployable.** 15.8s/query at pool 20 is 132× dense, against the 30s HTTP API
integration ceiling from 009 — an AWS maximum, not a setting. A cold request exceeds it outright,
and 15.8s warm is a worse experience than the 0.5s the service delivers now, to move one metric.

**It does not fix the problem that remains.** Reranking closes the ranking gap almost perfectly at
pool 20 and cannot touch the ~6% of gold chunks dense retrieval never reaches at any depth. That
residue is a coverage problem, and 008 named hybrid retrieval as the lever for it.

## Decision: keep the code, do not deploy it, publish the numbers

`CrossEncoderRetriever` stays in `rag/retriever.py`. It satisfies the same protocol as
`DenseRetriever`, wraps it rather than querying Qdrant itself (invariant 1), and is unit-tested
with a stub model so the ordering logic is verified without a 1.1 GB download. Nothing in the
serving path constructs it.

Shipping it would trade a 0.5s p50 for 15.8s to buy an improvement this eval set cannot firmly
establish. The project's stated purpose is a link someone can open; a demo that takes sixteen
seconds to answer fails that for a gain only a statistician could defend.

## What would change this

- **More questions.** The deltas are consistently positive and individually underpowered. At 100–150
  questions the same effect would either resolve or disappear, and that is the cheapest way to find
  out which.
- **A reranker that fits the latency budget.** A quantised or ONNX cross-encoder, or reranking only
  when the top dense score is low, so most queries never pay for it.
- **Somewhere other than Lambda.** The 30s ceiling is a property of HTTP API integrations, not of
  the model.

Recorded so the next person — including a later me — does not rebuild this and rediscover the same
15.8s.

