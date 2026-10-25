"""Score the serving hybrid retriever against the active passage-first eval set.

Reports two depths because they answer different questions. The shallow k values
are what a user experiences and what `compare_baseline` tracks. The deep ones ask
whether the gold chunk is reachable at all: a reranker can only reorder what the
retriever already returned, so recall@100 is the ceiling on what reranking could
ever achieve, and the gap to 1.0 is a coverage problem no reranker will fix.

Both blocks are printed so that every retrieval number in the README comes from
this one command, which is the rule the README claims to follow.
"""

from evaluation.eval_harness import DEFAULT_EVAL_SET, evaluate, load_eval_set
from rag.retriever import HybridRetriever

SHALLOW = (1, 3, 5, 10)
DEEP = (20, 50, 100)


def main() -> None:
    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    retriever = HybridRetriever()

    shallow = evaluate(retriever, eval_set, k_values=SHALLOW)
    print(shallow.pretty(intervals=True))
    print(
        f"\n(MRR is truncated at the deepest k scored, so the figure above is "
        f"MRR@{max(SHALLOW)}.)"
    )
    if len(eval_set) < 10:
        print(
            f"(Only {len(eval_set)} questions are in this active set. Treat these numbers "
            "as an indexing/retrieval smoke test, not a reliable corpus-wide estimate.)"
        )
    else:
        print(
            "(Intervals describe sensitivity to which questions were asked. For comparing "
            "retrievers, use bootstrap_delta_ci on the same questions.)"
        )

    deep = evaluate(retriever, eval_set, k_values=DEEP)
    print("\nReachability — the ceiling on what reranking could recover:\n")
    print(deep.pretty())
    print(
        f"\n(MRR@{max(DEEP)} above differs from MRR@{max(SHALLOW)} for the same reason: "
        f"a first hit found deeper than {max(SHALLOW)} is invisible to the shallower run.)"
    )


if __name__ == "__main__":
    main()
