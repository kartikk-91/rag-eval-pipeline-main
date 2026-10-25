"""The guard on R1's central fix.

The project previously had five places issuing or unpacking a Qdrant query, so
the offline evaluation scored a different code path than the API served. These
tests fail if a second retrieval path is ever reintroduced.
"""

import pytest

import rag.build_index as build_index

# rag.generate pulls in sentence_transformers (~13s). It is imported inside the
# integration tests so the fast subset — and the pre-commit hook — stay quick.

QUERY = "What is the BenCoref dataset?"
K = 5


def test_build_index_exposes_no_search_helper():
    """Structural guard: querying belongs to the retriever, indexing to build_index."""
    assert not hasattr(build_index, "search"), (
        "build_index.search() is back — that was the second retrieval path. "
        "Query through rag.retriever.DenseRetriever instead."
    )


@pytest.mark.integration
def test_eval_and_generation_paths_return_identical_chunks(retriever):
    """The IDs the eval harness scores are the IDs generation actually uses."""
    from rag.generate import retrieve_chunks

    eval_ids = retriever.retrieve(QUERY, K)
    generation_ids = [c.chunk_id for c in retrieve_chunks(QUERY, K)]
    assert eval_ids == generation_ids
    assert len(eval_ids) == K


@pytest.mark.integration
def test_answer_sources_match_the_retriever(retriever, monkeypatch):
    """End-to-end: the sources /ask reports are the retriever's top-k, in order.

    Generation is stubbed — this asserts the retrieval half without calling Bedrock.
    """
    from rag.generate import answer_question

    monkeypatch.setattr("rag.generate.generate_from_chunks", lambda q, chunks: "stub answer")
    _answer, chunks = answer_question(QUERY, K)
    assert [c.chunk_id for c in chunks] == retriever.retrieve(QUERY, K)
    # The chunks carry what a citation needs, so the API never has to go back to
    # the store for provenance retrieval already had.
    assert all(c.cite() for c in chunks)


@pytest.mark.integration
def test_retrieve_is_derived_from_retrieve_chunks(retriever):
    """The two views of one query cannot drift apart."""
    chunks = retriever.retrieve_chunks(QUERY, K)
    assert retriever.retrieve(QUERY, K) == [c.chunk_id for c in chunks]
    assert all(c.text and c.source and isinstance(c.score, float) for c in chunks)
