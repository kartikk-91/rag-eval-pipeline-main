"""BM25 and Reciprocal Rank Fusion, checked against arithmetic done by hand.

No Qdrant and no model: BM25 is built from in-memory chunks, and the hybrid is given
stub rankings. What matters here is that the scoring is the formula it claims to be;
whether it helps is the eval set's question, not a unit test's.
"""

import math

import pytest

from rag.retriever import BM25Retriever, HybridRetriever, tokenize
from rag.types import Chunk


def _c(cid: str, text: str = "", score: float = 0.0) -> Chunk:
    return Chunk(chunk_id=cid, document_id="d", source="s.pdf", text=text, score=score)


CORPUS = [
    _c("a", "coreference resolution with spanbert"),
    _c("b", "coreference resolution for bengali"),
    _c("c", "the muril model for indian languages and bengali"),
]


def test_tokenize_lowercases_and_splits_on_anything_not_alphanumeric():
    assert tokenize("SpanBERT's span-boundary (SBO) objective") == [
        "spanbert", "s", "span", "boundary", "sbo", "objective"
    ]


def test_bm25_score_matches_the_formula_computed_by_hand():
    bm25 = BM25Retriever(CORPUS)
    (top,) = bm25.retrieve_chunks("spanbert", 1)

    # "spanbert" occurs once, in doc a (4 tokens); avgdl = (4 + 4 + 8) / 3
    n, df, tf, dl, avgdl, k1, b = 3, 1, 1, 4, 16 / 3, 1.2, 0.75
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    assert top.chunk_id == "a"
    assert top.score == pytest.approx(expected)


def test_a_rare_term_outweighs_a_common_one():
    """'muril' is in one chunk, 'coreference' in two: the rare term should decide."""
    bm25 = BM25Retriever(CORPUS)
    assert bm25.retrieve("muril coreference", 1) == ["c"]


def test_a_term_in_every_chunk_still_scores_positive():
    """The +1 inside the idf log: without it, very common terms score negatively."""
    bm25 = BM25Retriever([_c("x", "model a"), _c("y", "model b")])
    assert all(c.score > 0 for c in bm25.retrieve_chunks("model", 2))


def test_a_query_with_no_known_terms_returns_nothing_rather_than_noise():
    assert BM25Retriever(CORPUS).retrieve_chunks("zzz qqq", 5) == []


def test_ties_are_broken_by_corpus_order_so_rankings_are_reproducible():
    bm25 = BM25Retriever([_c("x", "same words"), _c("y", "same words")])
    assert bm25.retrieve("same", 2) == ["x", "y"]


class _Ranked:
    """A retriever that returns a fixed ranking, for checking fusion arithmetic."""

    def __init__(self, ids):
        self._ids = ids

    def retrieve_chunks(self, query, k):
        return [_c(cid) for cid in self._ids[:k]]


def test_rrf_rewards_agreement_over_a_single_strong_vote():
    """Second in both rankings beats first in only one: 2/62 > 1/61."""
    hybrid = HybridRetriever(dense=_Ranked(["x", "both"]), bm25=_Ranked(["y", "both"]), pool=10)
    assert hybrid.retrieve("q", 1) == ["both"]


def test_rrf_score_is_the_sum_of_reciprocal_ranks():
    hybrid = HybridRetriever(dense=_Ranked(["a", "b"]), bm25=_Ranked(["b"]), pool=10)
    scores = {c.chunk_id: c.score for c in hybrid.retrieve_chunks("q", 2)}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)


def test_equal_fused_scores_keep_dense_order():
    """Symmetric rankings tie exactly; dense breaks it, being the measured baseline."""
    hybrid = HybridRetriever(dense=_Ranked(["d1"]), bm25=_Ranked(["l1"]), pool=10)
    assert hybrid.retrieve("q", 2) == ["d1", "l1"]


def test_the_hybrid_never_queries_the_store_itself():
    """Invariant 1: given its two rankers, it needs nothing else — no client, no model."""
    hybrid = HybridRetriever(dense=_Ranked(["a"]), bm25=_Ranked(["b"]), pool=10)
    assert not any(hasattr(hybrid, attr) for attr in ("_client", "_model"))
    assert set(hybrid.retrieve("q", 2)) == {"a", "b"}


def test_the_pool_is_never_smaller_than_k():
    seen = {}

    class _Spy(_Ranked):
        def retrieve_chunks(self, query, k):
            seen["k"] = k
            return super().retrieve_chunks(query, k)

    HybridRetriever(dense=_Spy(["a"]), bm25=_Ranked([]), pool=5).retrieve("q", 20)
    assert seen["k"] == 20
