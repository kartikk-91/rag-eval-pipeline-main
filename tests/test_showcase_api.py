"""The browser showcase must work without invoking an answer model."""

from rag.types import Chunk


def test_showcase_serves_browser_and_corpus_overview():
    from api.main import _PAGE, overview

    summary = overview()

    assert "Indic Research Desk" in _PAGE
    assert "data-query=" in _PAGE
    assert summary["catalog_papers"] >= 50


def test_search_returns_exact_evidence_and_catalog_citation(monkeypatch):
    import api.main as api

    chunk = Chunk(
        chunk_id="paper-02-chunk-0",
        document_id="paper-02",
        source="02-deephcoref_a_deep_neural_coreference_resolution_.pdf",
        text="The model combines a Bi-GRU-CNN with a biaffine classifier.",
        score=0.82,
        page=1,
        section="Abstract",
    )

    class FakeRetriever:
        def retrieve_chunks(self, query, k):
            assert query == "How does DeepHCoref work?"
            assert k == 5
            return [chunk]

    monkeypatch.setattr(api, "_bm25_searcher", lambda: FakeRetriever())
    response = api.search(
        api.SearchRequest(question="How does DeepHCoref work?", mode="bm25", k=5)
    )

    result = response.results[0].model_dump()
    assert result["title"].startswith("DeepHCoref")
    assert result["evidence"] == chunk.text
    assert result["citation"]


def test_local_showcase_defaults_are_used_only_without_explicit_qdrant_config(monkeypatch, tmp_path):
    import os

    import api.main as api

    index = tmp_path / api._SHOWCASE_INDEX_NAME
    index.mkdir()
    papers = tmp_path / "data" / "hindi_indic"
    papers.mkdir(parents=True)
    for name in ("QDRANT_PATH", "QDRANT_HOST", "QDRANT_BAKED_INDEX", "RAG_COLLECTION", "RAG_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)

    api._configure_local_showcase_defaults(tmp_path)

    assert os.environ["QDRANT_PATH"] == str(index)
    assert os.environ["RAG_COLLECTION"] == api._SHOWCASE_COLLECTION
    assert os.environ["RAG_DATA_DIR"] == str(papers)


def test_local_showcase_defaults_respect_explicit_qdrant_target(monkeypatch, tmp_path):
    import os

    import api.main as api

    (tmp_path / api._SHOWCASE_INDEX_NAME).mkdir()
    monkeypatch.setenv("QDRANT_HOST", "qdrant.example.invalid")
    monkeypatch.delenv("QDRANT_PATH", raising=False)
    monkeypatch.delenv("QDRANT_BAKED_INDEX", raising=False)

    api._configure_local_showcase_defaults(tmp_path)

    assert "QDRANT_PATH" not in os.environ
