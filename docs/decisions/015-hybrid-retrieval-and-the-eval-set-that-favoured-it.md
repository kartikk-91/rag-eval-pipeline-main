> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 015 — Hybrid retrieval, and the eval set that favoured it

*R2, Step 4 — 2026-10-19*

## Observation

Dense retrieval misses where a question names a specific term (a dataset, a metric, a model) and
similarity spreads across papers that discuss the same topic. Lexical matching is the textbook
complement. It is also the best experiment available that does not invalidate the frozen eval
set: chunk size cannot change without relabelling all 42 questions (invariant 3).

## What was built

- **BM25, written out** (`rag/retriever.py`), with `k1 = 1.2` and `b = 0.75`, no stemming and no
  stopword list. Checked against a score computed by hand.
- **Hybrid: Reciprocal Rank Fusion** of the dense and BM25 top 100, `RRF_K = 60` (Cormack et
  al., 2009). It uses ranks, not a weighted sum of scores, because a cosine similarity and a BM25
  score are on unrelated scales and any weight between them would be tuned on the test set.
- **Every parameter was fixed before measuring.** With 42 questions, tuning on them is fitting
  noise and calling it a result.
- **BM25 is built from `DenseRetriever.all_chunks()`**, the same stored chunks dense searches,
  not from re-reading the PDFs. Both rank one corpus, and Qdrant is still touched in one class
  (invariant 1).

## Prediction, written before the numbers

From the plan: lexical matching should help on named-term questions. Of the two gold chunks dense
retrieval cannot reach at k = 100 (008), one shares several query terms and might be recovered;
the other shares one. Hybrid is not a cure for the coverage gap.

## Result on the eval set as written: a clear win

`python -m evaluation.compare_retrievers --system bm25 --system hybrid`, paired 95% intervals:

| | dense | hybrid | delta | 95% CI |
|---|---|---|---|---|
| recall@5 | 0.667 | 0.810 | +0.143 | [+0.048, +0.262] |
| nDCG@10 | 0.598 | 0.671 | +0.074 | [+0.002, +0.151] |
| MRR@10 | 0.578 | 0.630 | +0.052 | [−0.043, +0.146] |
| recall@1 | 0.393 | 0.440 | +0.048 | [−0.071, +0.179] |

At the depths that ship, it gains 4 questions and loses 0 at k = 3 (the API default), and gains
7 and loses 1 at k = 4 (what the demo page sends). Warm latency is p50 65ms → 108ms.

The prediction held exactly: the SBERT chunk was recovered (BM25 rank 8, hybrid rank 29, too deep
to help an answer), and the anaphora chunk stayed unreachable by all three systems.

By that table, this should ship.

## Why it did not: the questions were written beside their answers

Decision 008 writes each question **while reading its gold chunk**. That makes labels independent
of the retriever, which was the point, but a question written next to its answer borrows the
answer's words. Measured: **53%** of a question's informative terms (those in fewer than 5% of
chunks) appear in its gold chunk, against 21% for other chunks of the same paper and 2% for random
ones. BM25 is rewarded for exactly that, and a real visitor has not read the passage.

The paired intervals cannot see this. They account for *which* questions were sampled, not for a
bias in *how* every one of them was written.

**So the questions were reworded and scored again** (`evaluation/paraphrase.py`). Nova Pro
rewrote each as an NLP researcher who had not read the paper would ask it, under rules fixed
beforehand: keep proper names (a real user types "SpanBERT"), change every other content word
where an alternative exists, and keep the meaning exactly. Each rewrite was then reviewed by a
person, **before any was scored**. Six had drifted (for example "accuracy" became "precision",
and the proper name "selffer" was rewritten) and were **excluded, not repaired**, since repairing
would let the reviewer steer the wording toward the retriever they expected to win. That leaves
36, scored against the same gold labels: `python -m evaluation.robustness`.

The rewording did what it was for: informative-term overlap with the gold chunk fell from 53% to
**32%**.

| hybrid − dense, paired, 95% CI | as written | reworded |
|---|---|---|
| recall@3 (API default) | +0.083 [+0.000, +0.194] | −0.014 [−0.139, +0.111] |
| recall@4 (demo page) | **+0.153 [+0.028, +0.292]** | +0.028 [−0.111, +0.167] |
| recall@5 | **+0.167 [+0.056, +0.306]** | +0.028 [−0.111, +0.167] |
| nDCG@10 | **+0.091 [+0.012, +0.178]** | +0.046 [−0.071, +0.158] |
| MRR@10 | +0.074 [−0.028, +0.181] | +0.003 [−0.131, +0.133] |

BM25 alone collapsed under the rewording (MRR@10 0.477 → 0.316) and dense largely held
(0.508 → 0.462), as expected: lexical matching depends on the words, dense matching mostly does
not.

**What this does and does not show.** It does not prove the advantage shrank: the per-question
interval on the shrink itself includes zero at every metric, though only just at recall@5
(+0.139, [−0.014, +0.292]). Thirty-six questions cannot settle that. What it does show is that the
evidence *for* hybrid depends on the wording artifact. On questions phrased the way a visitor
would, its advantage at the depth that ships is indistinguishable from zero.

## Decision: do not ship hybrid; keep it as a measured experiment

The burden of proof sits with the change. Against a gain that is not established on realistic
phrasing, it costs:

- **+43ms p50 per query**, on top of a second ranking system that must be built at cold start
  (0.6s) and kept consistent with the dense index;
- **two of the three demo examples pushed down from rank 1.** SpanBERT and MuRIL are among the 5
  rank-1 losses. They were chosen for the demo page because dense ranks them first, so they could
  only lose — a selection effect, but one a visitor would see.

Like the reranker (010), `BM25Retriever` and `HybridRetriever` stay in `rag/retriever.py`,
tested and runnable, and production stays on dense.

## A finding that outlives this decision

**The eval set flatters every retriever, not just BM25.** On the same 36 questions, dense
recall@5 is 0.625 as written and 0.542 reworded. The README's retrieval numbers are best read as
an optimistic estimate of what a visitor gets, and that belongs in its limitations, not only here.

The eval set stays frozen as the regression baseline: it still detects a change that makes
retrieval worse, which is its job. It is just not an unbiased estimate of absolute quality.

## Multiple comparisons

About thirty intervals were computed across the two scripts. On the written questions, the
hybrid's intervals that strictly clear zero are recall@5 and nDCG@10 in the full set, and
recall@4, recall@5 and nDCG@10 in the 36-question subset. Three more sit with a lower bound of
exactly +0.000. Scattered, a handful clearing zero would be what chance produces. These are not
scattered: one direction, adjacent depths, the column the hypothesis predicts. That argues for a
real effect on the questions as written. It does nothing for the reworded column, where none
clears zero.

## What would change this

- **Real visitor questions.** Langfuse now records every question asked of the live demo (014). A
  few dozen, labelled chunk-first as in 008, would replace an LLM's rewording with the actual
  distribution. If hybrid wins there, ship it: the code is already written.
- **Rewording by a person.** One model's paraphrases are a single sample of "phrased differently";
  32% overlap is lower, not zero.
- **Hybrid plus reranking.** Hybrid improves recall@20, the reranker's ceiling, but the reranker's
  latency (010) rules out the combination on Lambda regardless.

