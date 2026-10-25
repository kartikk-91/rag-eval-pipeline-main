"""Ordering behaviour of CrossEncoderRetriever, with no model downloaded.

The cross-encoder itself is a stub returning scores chosen by the test. What is
under test is everything around it: that the pool is honoured, that results are
reordered by the reranker's scores rather than the retriever's, that ties keep
the dense order, and that the score a caller sees is the reranker's.
"""

from rag.retriever import CrossEncoderRetriever
from rag.types import Chunk


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        source="paper.pdf",
        text=text,
        score=0.5,
        page=1,
        section=None,
    )


class _FakeDense:
    """Returns a fixed ranked list and records the k it was asked for."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.asked_for = None

    def retrieve_chunks(self, query, k):
        self.asked_for = k
        return self._chunks[:k]


class _FakeModel:
    """Scores pairs from a lookup keyed on passage text."""

    def __init__(self, scores_by_text):
        self._scores = scores_by_text

    def predict(self, pairs):
        return [self._scores[text] for _query, text in pairs]


def test_reranker_reorders_by_its_own_scores():
    chunks = [_chunk("a", "low"), _chunk("b", "high"), _chunk("c", "mid")]
    rr = CrossEncoderRetriever(
        dense=_FakeDense(chunks),
        model=_FakeModel({"low": 0.1, "high": 0.9, "mid": 0.5}),
        pool=3,
    )
    assert rr.retrieve("q", 3) == ["b", "c", "a"]


def test_reranker_asks_for_the_pool_not_k():
    # The whole point: fetch many candidates, return few. Asking dense for k
    # would leave the reranker nothing to reorder.
    chunks = [_chunk(str(i), f"t{i}") for i in range(50)]
    dense = _FakeDense(chunks)
    rr = CrossEncoderRetriever(
        dense=dense, model=_FakeModel({f"t{i}": float(i) for i in range(50)}), pool=20
    )
    rr.retrieve("q", 3)
    assert dense.asked_for == 20


def test_pool_never_shrinks_below_k():
    # A pool smaller than k would silently return fewer results than asked for.
    chunks = [_chunk(str(i), f"t{i}") for i in range(10)]
    dense = _FakeDense(chunks)
    rr = CrossEncoderRetriever(
        dense=dense, model=_FakeModel({f"t{i}": float(i) for i in range(10)}), pool=2
    )
    assert len(rr.retrieve("q", 5)) == 5
    assert dense.asked_for == 5


def test_ties_keep_the_dense_order():
    # sorted() is stable, so equally scored candidates must not be reshuffled by
    # an implementation detail — the upstream ranking is the sensible tiebreak.
    chunks = [_chunk("first", "x"), _chunk("second", "y"), _chunk("third", "z")]
    rr = CrossEncoderRetriever(
        dense=_FakeDense(chunks), model=_FakeModel({"x": 0.4, "y": 0.4, "z": 0.4}), pool=3
    )
    assert rr.retrieve("q", 3) == ["first", "second", "third"]


def test_returned_score_is_the_rerankers_not_qdrants():
    # The two are different quantities on different scales; a consumer comparing
    # one to the other would be comparing nothing.
    chunks = [_chunk("a", "only")]
    rr = CrossEncoderRetriever(
        dense=_FakeDense(chunks), model=_FakeModel({"only": 0.77}), pool=1
    )
    (got,) = rr.retrieve_chunks("q", 1)
    assert got.score == 0.77
    assert got.chunk_id == "a" and got.page == 1  # everything else is preserved


def test_no_candidates_returns_empty_without_calling_the_model():
    class _Exploding:
        def predict(self, pairs):
            raise AssertionError("must not score an empty candidate list")

    rr = CrossEncoderRetriever(dense=_FakeDense([]), model=_Exploding(), pool=10)
    assert rr.retrieve("q", 5) == []
