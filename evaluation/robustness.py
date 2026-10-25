"""Does a retriever's advantage survive questions not worded like their answers?

Scores dense, BM25 and hybrid on the same questions twice — as written beside their
gold chunks (evaluation/eval_set.jsonl), and as rewritten by evaluation/paraphrase.py
for a reader who has not seen the passage — against the same gold labels. Only
rewrites a person reviewed as meaning-preserving are used, and the originals are
restricted to those same questions, so the two wordings are paired.

First it checks the rewrite did what it was for: if the gold chunk's vocabulary
overlap did not drop, the second wording tests nothing.

    python -m evaluation.robustness
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from evaluation.eval_harness import (
    EvalExample,
    bootstrap_delta_ci,
    evaluate,
)
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever, tokenize

PARAPHRASED = Path(__file__).resolve().parent / "robustness" / "paraphrased.jsonl"
K_VALUES = (1, 3, 4, 5, 10)  # 3 is what /ask defaults to, 4 what the demo page sends
METRICS = ("recall@1", "recall@3", "recall@4", "recall@5", "ndcg@10", "mrr")


def _overlap(examples, toks, df, cap) -> float:
    """Mean share of a question's informative terms that appear in its gold chunk."""
    shares = []
    for ex in examples:
        terms = {t for t in tokenize(ex.query) if 0 < df[t] < cap}
        if terms:
            shares.append(max(len(terms & toks[g]) for g in ex.relevant_chunk_ids) / len(terms))
    return statistics.mean(shares)


def main() -> int:
    rows = [json.loads(line) for line in PARAPHRASED.open()]
    kept = [r for r in rows if r["review"] == "ok"]
    as_written = [EvalExample(r["original"], frozenset(r["relevant_chunk_ids"])) for r in kept]
    rewritten = [EvalExample(r["query"], frozenset(r["relevant_chunk_ids"])) for r in kept]

    dense = DenseRetriever()
    corpus = dense.all_chunks()
    bm25 = BM25Retriever(corpus)
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    toks = {c.chunk_id: set(tokenize(c.text)) for c in corpus}
    df = Counter(t for s in toks.values() for t in s)
    cap = 0.05 * len(corpus)
    print(f"{len(kept)} of {len(rows)} rewrites reviewed as meaning-preserving\n")
    print("informative query terms (in <5% of chunks) that appear in the gold chunk:")
    print(f"  as written {_overlap(as_written, toks, df, cap):.0%}   "
          f"rewritten {_overlap(rewritten, toks, df, cap):.0%}\n")

    systems = {"dense": dense, "bm25": bm25, "hybrid": hybrid}
    reports = {
        (name, wording): evaluate(r, examples, k_values=K_VALUES)
        for name, r in systems.items()
        for wording, examples in (("written", as_written), ("rewritten", rewritten))
    }

    def mean(name, wording, metric):
        rep = reports[(name, wording)]
        return sum(rep.per_query[metric]) / rep.n_queries

    print(f"{'metric':<10}" + "".join(f"{n + ' ' + w[:5]:>14}" for n in systems
                                       for w in ("written", "rewritten")))
    for metric in METRICS:
        print(f"{metric:<10}" + "".join(f"{mean(n, w, metric):>14.3f}" for n in systems
                                         for w in ("written", "rewritten")))

    print("\nhybrid minus dense, paired, 95% CI — as written vs rewritten:")
    for metric in METRICS:
        cells = []
        for wording in ("written", "rewritten"):
            b = reports[("dense", wording)].per_query[metric]
            c = reports[("hybrid", wording)].per_query[metric]
            lo, hi = bootstrap_delta_ci(b, c)
            cells.append(f"{statistics.mean(c) - statistics.mean(b):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {metric:<10} {cells[0]:>26}   {cells[1]:>26}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
