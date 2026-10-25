"""Compare two retrievers on the frozen eval set, with paired intervals.

The comparison this project is built to make, and the one it is easy to get wrong.
Both systems answer the SAME questions, so the question that matters is not "is
0.52 bigger than 0.39" but "is the per-query difference distinguishable from the
questions happening to suit one of them".

Absolute 95% intervals here are roughly ±0.13 wide, so any two retrievers'
intervals overlap heavily and overlap gets misread as "no difference". The paired
interval cancels the difficulty both systems share; a change that helps 9 questions
and harms 2 shows an interval clear of zero while the absolute intervals sit on top
of each other. So the verdict below is read off the paired interval, never off the
two means.

    python -m evaluation.compare_retrievers                       # dense vs +rerank
    python -m evaluation.compare_retrievers --pool 20 --pool 50 --pool 100
    python -m evaluation.compare_retrievers --system bm25 --system hybrid
"""

import argparse
import time

from evaluation.eval_harness import (
    DEFAULT_EVAL_SET,
    bootstrap_delta_ci,
    evaluate,
    load_eval_set,
)
from rag.config import RERANK_POOL
from rag.retriever import BM25Retriever, CrossEncoderRetriever, DenseRetriever, HybridRetriever

# MRR is truncated at the largest k evaluated, so the headline pass keeps 10 as its
# largest: "mrr" here must mean MRR@10, as it does in the README, the baseline and
# decision 010. Adding 100 to this tuple once silently turned it into MRR@100.
K_VALUES = (1, 3, 5, 10)
HEADLINE = ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr")
# Depth is a separate pass for the same reason: it answers a different question
# (can the system reach the gold chunk at all?) and must not leak into MRR.
DEEP_K = (20, 50, 100)
DEEP = ("recall@20", "recall@100")


def _verdict(lo: float, hi: float) -> str:
    """What the interval licenses saying out loud."""
    if lo > 0:
        return "better"
    if hi < 0:
        return "WORSE"
    return "not distinguishable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        action="append",
        choices=("rerank", "bm25", "hybrid"),
        help="candidate system(s) to compare against dense. Default: rerank.",
    )
    ap.add_argument(
        "--pool",
        type=int,
        action="append",
        help="candidate pool size; repeatable to sweep. Default: the configured one.",
    )
    args = ap.parse_args()
    systems = args.system or ["rerank"]
    pools = args.pool or [RERANK_POOL]

    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    dense = DenseRetriever()

    t0 = time.perf_counter()
    base = evaluate(dense, eval_set, k_values=K_VALUES)
    base_deep = evaluate(dense, eval_set, k_values=DEEP_K)
    dense_s = (time.perf_counter() - t0) / len(eval_set)

    print(f"{len(eval_set)} questions, dense baseline at {dense_s * 1000:.0f} ms/query\n")

    candidates = []
    if "bm25" in systems or "hybrid" in systems:
        # Built once and shared: what a deployed hybrid would pay at cold start.
        t0 = time.perf_counter()
        corpus = dense.all_chunks()
        bm25 = BM25Retriever(corpus)
        print(f"BM25 index: {len(corpus)} chunks, built in {time.perf_counter() - t0:.2f}s\n")
    for system in systems:
        if system == "rerank":
            # Reuse the dense retriever: a second instance would load the embedding
            # model again and change the latency being measured.
            candidates += [(f"+rerank pool {p}", CrossEncoderRetriever(dense=dense, pool=p))
                           for p in pools]
        elif system == "bm25":
            candidates.append(("bm25 alone", bm25))
        else:
            candidates.append(("hybrid (RRF)", HybridRetriever(dense=dense, bm25=bm25)))

    for label, retriever in candidates:
        t0 = time.perf_counter()
        cand = evaluate(retriever, eval_set, k_values=K_VALUES)
        rr_s = (time.perf_counter() - t0) / len(eval_set)

        print(f"=== {label} — {rr_s * 1000:.0f} ms/query ({rr_s / dense_s:.1f}x dense) ===")
        print(f"  {'metric':<11} {'dense':>7} {'cand':>8} {'delta':>7}   95% CI of delta")
        cand_deep = evaluate(retriever, eval_set, k_values=DEEP_K)
        for metric, b_rep, c_rep in [(m, base, cand) for m in HEADLINE] + [
            (m, base_deep, cand_deep) for m in DEEP
        ]:
            b = sum(b_rep.per_query[metric]) / b_rep.n_queries
            c = sum(c_rep.per_query[metric]) / c_rep.n_queries
            lo, hi = bootstrap_delta_ci(b_rep.per_query[metric], c_rep.per_query[metric])
            print(
                f"  {metric:<11} {b:>7.3f} {c:>8.3f} {c - b:>+7.3f}   "
                f"[{lo:>+6.3f}, {hi:>+6.3f}]  {_verdict(lo, hi)}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
