> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 008 — An eval set the retriever did not help write

*R2, Step 1 — 2026-10-07*

## Observation: the answer key was derived from the thing being graded

`rag/label_helper.py` picked gold labels by running a query through the retriever and choosing from
its top-k. Its docstring, written in PR #1, presented sourcing labels from the production retriever
as a *virtue* — the same path, so no drift.

It is circular. A chunk the retriever ranks 50th can never become a label, because nobody ever sees
it. Every question is therefore guaranteed to have its answer somewhere in the top-k of the system
being measured, and recall is reported against an answer key that system helped write. The number
is not wrong arithmetically; it is measuring the wrong thing.

The ten original questions were built this way. They are kept, because discarding them would
confuse "we changed the labels" with "we changed the retriever", but they are now a minority.

## Decision: draw the chunk first, then write the question

`scripts/sample_chunks.py` samples chunks directly — stratified per paper, references and
acknowledgements skipped, a 600-character floor — and each question is written from the sampled
text without the retriever being consulted at all. The retriever never sees a label before it is
scored on it.

Drafts stay in `evaluation/drafting/` with the quoted line that justifies each gold ID, so a label
can be re-checked rather than taken on trust. That is the difference between a frozen set and a set
nobody can audit.

**42 questions, 50 gold chunks, 37 of the 48 papers.** Frozen before any retrieval comparison, so a
future change is scored against an answer key that predates it.

The first two batches came out heavy on evaluation-metric papers — CEAF, MUC, B³, the mention
identification effect — because that is what the sampler surfaced from a corpus with many such
papers. Batch 3 was deliberately steered toward transformers, data augmentation and cross-lingual
transfer. Stratification makes coverage even across papers, not across *topics*.

## Measured validation, and a conclusion that did not survive it

| k | recall@k | hit-rate@k |
|---|---|---|
| 1 | 0.393 | 0.476 |
| 10 | 0.774 | 0.810 |
| 20 | 0.869 | 0.881 |
| 100 | 0.940 | 0.952 |

Two readings, and they point in different directions:

**Ranking is the main lever.** recall@1 is 0.393 while recall@20 is 0.869. Most relevant material
*is* retrieved and merely ordered badly, which is precisely what a cross-encoder reranker fixes.
A 48-point gap is a large opportunity for a component that only reorders.

**Coverage is not solved.** recall@100 stops at 0.940. Roughly 6% of gold chunks are unreachable at
any practical depth, and no reranker touches them — they never enter the candidate pool.

The plan said, on the 10-question set, that coverage was fine and ranking was the only bottleneck,
and excluded hybrid retrieval on that basis. At 42 questions that no longer holds. Hybrid is back
on the table with a specific target: the ~6% that dense retrieval cannot reach. The order is
unchanged — reranking first, because the 48-point gap dwarfs the 6% — but it is now a ranked list
of two live options rather than one option and one dismissal.

## Trade-off: 42 questions is enough to order two systems, not to trust small gaps

One question moves hit-rate by 2.4 points. The committed baseline's ±3% tolerance is about 1.3
questions, which means the regression gate can fire on noise and a reranker could "win" by moving
two questions that happened to fall its way.

So bootstrap confidence intervals come before any comparison is believed, not after. Quoting a
delta without an interval on a set this size would be the same error as the circular labels: a
number that looks like evidence and is not.

One smaller thing the same work surfaced: MRR is truncated at the deepest k scored, so the identical
run reports 0.578 at k=10 and 0.584 at k=100. It is MRR@k and is always written with its depth.
`evaluation/run_eval.py` prints both so the truncation is visible in output rather than explained in
prose.

