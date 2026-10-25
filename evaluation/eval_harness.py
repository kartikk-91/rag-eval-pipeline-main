"""
eval_harness.py — Retrieval evaluation for a RAG pipeline (recall@k, hit-rate@k, MRR).

Design decisions (these are the things worth being able to defend in an interview):

1. The harness is DECOUPLED from the vector store. Anything implementing the
   `Retriever` protocol — dense, BM25, or hybrid — can be scored on the SAME
   labeled set, so you can report before/after deltas when you change strategy.
2. Metrics are implemented FROM SCRATCH (no eval library). They are pure
   functions, unit-checkable in isolation.
3. Active gold labels are source passages selected before retrieval; a retrieved
   result is relevant only when its text contains the evidence passage. Legacy
   examples may still use chunk IDs.
4. recall@k and hit-rate@k are kept SEPARATE on purpose (most RAG blogs conflate
   them; with multiple gold chunks per query they are not the same metric).

Active eval records in evaluation/eval_set.jsonl carry `gold_evidence` entries
with source URL, location and passage text. Chunk-ID labels remain supported for
legacy sets only.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Anchored to this package, not the working directory. run_eval, compare_baseline
# and the MCP server all need it, and an MCP client may launch the server from
# anywhere; a relative path silently breaks there.
DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalExample:
    """One labeled query.

    relevant_chunk_ids: every chunk a *correct* retriever should surface for this
    query — minimal but complete. Include all chunks that genuinely answer it,
    nothing that doesn't. Quality of these labels caps the quality of your numbers.
    """
    query: str
    relevant_chunk_ids: frozenset[str] = frozenset()
    gold_passages: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict) -> EvalExample:
        return EvalExample(
            query=d["query"],
            relevant_chunk_ids=frozenset(d.get("relevant_chunk_ids", [])),
            gold_passages=tuple(
                item["passage_text"] for item in d.get("gold_evidence", [])
            ),
        )


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """Load a JSONL eval set. Fails loudly on a malformed line (with line number)."""
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(EvalExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Bad eval example on line {line_no}: {e}") from e
    if not examples:
        raise ValueError(f"No eval examples found in {path}")
    return examples


# --------------------------------------------------------------------------- #
# Retriever interface — implement this around Qdrant (or BM25, or hybrid).
# Runtime and active evaluation use rag/retriever.py: HybridRetriever.
# --------------------------------------------------------------------------- #

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return up to k chunk IDs, ranked best-first."""
        ...


def _normalise_passage(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _passage_in_chunk(passage: str, chunk: str) -> bool:
    needle = _normalise_passage(passage)
    haystack = _normalise_passage(chunk)
    # Some PDFs omit spaces between adjacent text spans. Compare the exact
    # normalized word sequence both with and without whitespace so that layout
    # extraction artifacts do not turn a real source passage into a false miss.
    return bool(needle) and (
        needle in haystack or needle.replace(" ", "") in haystack.replace(" ", "")
    )


# --------------------------------------------------------------------------- #
# Metrics (per query). Pure functions.
# --------------------------------------------------------------------------- #

def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """True recall@k:  |top_k ∩ relevant| / |relevant|.

    Measures COVERAGE — of all the chunks that should have been found, how many were.
    Not the same as hit-rate when a query has more than one gold chunk.
    """
    if not relevant:
        return 0.0  # undefined for empty gold; treat as 0 (or filter these out upstream)
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Binary hit-rate@k: 1.0 if ANY relevant chunk is in the top-k, else 0.0.

    This is what many RAG tutorials loosely call 'recall@k'. Reported separately
    so the distinction is explicit.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    """Reciprocal of the 1-indexed rank of the FIRST relevant chunk; 0.0 if none.

    Measures RANKING QUALITY — how near the top the first good chunk lands.
    Averaged across queries this is MRR.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Normalised discounted cumulative gain at k, with binary relevance.

    Fills a gap the other three leave. MRR sees only the FIRST hit, so moving a
    second gold chunk from rank 9 to rank 2 does not change it. recall@k sees only
    membership, so that same move does not change it either. nDCG is the one that
    rewards it, which is exactly the improvement a reranker is supposed to produce
    — without it a reranker could do real work and score flat on every metric here.

    Binary gains, so DCG = sum over hits of 1/log2(rank+1), and the ideal ordering
    puts min(len(relevant), k) hits in the top positions. Graded relevance would
    need a judgement of *how* relevant each chunk is; these labels do not carry one,
    and inventing grades to feed a formula would be worse than binary.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 20261007,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-query scores.

    Resamples QUERIES with replacement, which is the thing that varies: the eval
    set is one sample of the questions that could have been asked, and a different
    42 would give different numbers. The interval says how much of a reported
    difference is the retriever and how much is which questions got picked.

    Resampling scores rather than re-running retrieval is deliberate and not merely
    faster — retrieval is deterministic here, so re-running it 10,000 times would
    produce the identical ranked list every time and measure nothing.

    Seeded by default: a confidence interval that moves when you re-run it is not
    reproducible, and this project's README may only quote numbers its own commands
    reproduce.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(values[0]), float(values[0]))

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    lo = means[int(tail * iterations)]
    hi = means[min(int((1.0 - tail) * iterations), iterations - 1)]
    return (lo, hi)


def bootstrap_delta_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 20261007,
) -> tuple[float, float]:
    """Interval for the mean per-query DIFFERENCE, candidate minus baseline.

    This, not two separate intervals, is how two retrievers get compared here.
    On this eval set recall@1 carries a 95% interval roughly ±0.13 wide, so any
    two systems' intervals will overlap heavily and overlap is routinely misread
    as "no difference". It is not: the intervals are wide because the QUESTIONS
    vary in difficulty, and both systems answer the same questions.

    Resampling the per-query differences cancels that shared difficulty. A
    reranker that improves 9 questions and harms 2 shows a positive interval well
    away from zero even while both absolute intervals straddle each other.

    The interval excluding zero is the claim worth making: "better on this set"
    rather than "a bigger number on this set".
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs the same queries in the same order: "
            f"{len(baseline)} vs {len(candidate)}"
        )
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_mean_ci(deltas, confidence=confidence, iterations=iterations, seed=seed)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    n_queries: int
    k_values: tuple[int, ...]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float] = field(default_factory=dict)

    # Per-query scores, keyed by metric name ("recall@5", "mrr", ...). Kept so an
    # interval can be computed without re-running retrieval, and so a future
    # comparison can pair queries between two systems rather than only compare
    # their means. Not printed; this is the raw material under the aggregates.
    per_query: dict[str, list[float]] = field(default_factory=dict)

    def ci(self, metric: str, confidence: float = 0.95) -> tuple[float, float]:
        """Bootstrap interval for one metric, e.g. ci("recall@5") or ci("mrr")."""
        return bootstrap_mean_ci(self.per_query[metric], confidence=confidence)

    def pretty(self, intervals: bool = False) -> str:
        """Render the report. With intervals=True, append 95% bootstrap CIs.

        Off by default because computing them costs ~10,000 resamples per metric,
        and because an interval printed beside every number invites reading the
        width as precision rather than as what it is: how much the figure depends
        on which active eval questions were asked.
        """
        lines = [
            f"Queries evaluated : {self.n_queries}",
            f"MRR               : {self.mrr:.3f}",
        ]
        if intervals and "mrr" in self.per_query:
            lo, hi = self.ci("mrr")
            lines[-1] += f"   95% CI [{lo:.3f}, {hi:.3f}]"

        header = f"{'k':>4} | {'recall@k':>9} | {'hit@k':>6} | {'nDCG@k':>7}"
        if intervals:
            header += f" | {'recall@k 95% CI':>18}"
        lines += ["", header, "-" * len(header)]

        for k in self.k_values:
            row = (
                f"{k:>4} | {self.recall_at_k[k]:>9.3f} | {self.hit_rate_at_k[k]:>6.3f} "
                f"| {self.ndcg_at_k.get(k, float('nan')):>7.3f}"
            )
            if intervals and f"recall@{k}" in self.per_query:
                lo, hi = self.ci(f"recall@{k}")
                row += f" | [{lo:>6.3f}, {hi:>6.3f}]"
            lines.append(row)
        return "\n".join(lines)


def evaluate(
    retriever: Retriever,
    eval_set: Sequence[EvalExample],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    """Run the retriever over the labeled set and aggregate.

    Retrieves once at max(k_values) per query, then slices — so MRR can find the
    first relevant chunk anywhere up to the deepest k, and every k shares one call.
    Using multiple k values lets you separate a RETRIEVAL problem (low recall@10)
    from a RANKING problem (good recall@10 but low recall@1 / low MRR).
    """
    max_k = max(k_values)
    per_query: dict[str, list[float]] = {"mrr": []}
    for k in k_values:
        per_query[f"recall@{k}"] = []
        per_query[f"hit@{k}"] = []
        per_query[f"ndcg@{k}"] = []

    for ex in eval_set:
        if ex.gold_passages and hasattr(retriever, "retrieve_chunks"):
            chunks = retriever.retrieve_chunks(ex.query, max_k)
            # A gold passage is relevant only when its text is present in the
            # retrieved chunk. Returning the same source/paper is not evidence.
            retrieved = []
            for rank, chunk in enumerate(chunks, start=1):
                hits = [
                    f"passage-{i}"
                    for i, passage in enumerate(ex.gold_passages)
                    if _passage_in_chunk(passage, chunk.text)
                ]
                retrieved.extend(hits or [f"chunk-{rank}"])
            relevant = frozenset(f"passage-{i}" for i in range(len(ex.gold_passages)))
        else:
            retrieved = retriever.retrieve(ex.query, max_k)
            relevant = ex.relevant_chunk_ids
        for k in k_values:
            per_query[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            per_query[f"hit@{k}"].append(hit_rate_at_k(retrieved, relevant, k))
            per_query[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
        per_query["mrr"].append(reciprocal_rank(retrieved, relevant))

    n = len(eval_set)
    mean = lambda vs: sum(vs) / n  # noqa: E731 — n, not len(vs); they are equal here

    return EvalReport(
        n_queries=n,
        k_values=k_values,
        recall_at_k={k: mean(per_query[f"recall@{k}"]) for k in k_values},
        hit_rate_at_k={k: mean(per_query[f"hit@{k}"]) for k in k_values},
        ndcg_at_k={k: mean(per_query[f"ndcg@{k}"]) for k in k_values},
        mrr=mean(per_query["mrr"]),
        per_query=per_query,
    )


# --------------------------------------------------------------------------- #
# There is deliberately NO retriever implementation in this module. The harness
# is decoupled from the store: anything with a .retrieve(query, k) -> list[str]
# satisfies the protocol above. The project's implementation lives in
# rag/retriever.py, and is the single retrieval path the API and agent also use
# (see docs/decisions/001-unified-retrieval-path.md).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Self-check: a stub retriever with KNOWN output so you can verify the harness
# before wiring in Qdrant.  Run:  python eval_harness.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    class _StubRetriever:
        """Returns a fixed ranked list regardless of query — for testing metrics."""
        def __init__(self, ranked: list[str]) -> None:
            self._ranked = ranked

        def retrieve(self, query: str, k: int) -> list[str]:
            return self._ranked[:k]

    # gold = {c2, c5};  ranked = [c9, c2, c7, c5, c1]
    #   first relevant (c2) at rank 2          -> RR = 1/2 = 0.5
    #   recall@1 = 0/2=0.0  recall@3 = 1/2=0.5  recall@5 = 2/2=1.0
    #   hit@1   = 0.0       hit@3    = 1.0       hit@5    = 1.0
    stub = _StubRetriever(["c9", "c2", "c7", "c5", "c1"])
    example = EvalExample(query="(stub)", relevant_chunk_ids=frozenset({"c2", "c5"}))
    report = evaluate(stub, [example], k_values=(1, 3, 5))

    assert abs(report.mrr - 0.5) < 1e-9, report.mrr
    assert report.recall_at_k[1] == 0.0
    assert report.recall_at_k[3] == 0.5
    assert report.recall_at_k[5] == 1.0
    assert report.hit_rate_at_k[1] == 0.0
    assert report.hit_rate_at_k[3] == 1.0
    print("Self-check passed.\n")
    print(report.pretty())
