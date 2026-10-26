"""The browser showcase must work without invoking an answer model."""

from rag.types import Chunk


def test_showcase_serves_browser_and_corpus_overview():
    from api.main import _PAGE, index, overview

    summary = overview()

    assert "Indic Paper Search" in _PAGE
    assert "Search papers" in _PAGE
    assert "Ask the paper collection" in _PAGE
    assert 'id="statHybridRecall"' in _PAGE
    assert 'value="How does DeepHCoref build its Hindi coreference system?"' in _PAGE
    assert "Search settings" in _PAGE
    assert "Cosine similarity" in _PAGE
    assert "Generate cited answer" in _PAGE
    assert "local demo ready" not in _PAGE.lower()
    assert "may incur" not in _PAGE.lower()
    assert "Bedrock" not in _PAGE
    assert "AWS" not in _PAGE
    assert "data-query=" in _PAGE
    assert index() == _PAGE
    assert summary["catalog_papers"] >= 50
    assert summary["metrics"]["metrics_are_current"] is True
    assert summary["metrics"]["indexed_chunks"] == 2608
    assert summary["metrics"]["systems"]["hybrid"]["recall_at_10"] == 0.956


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
    assert result["chunk_id"] == chunk.chunk_id
    assert result["title"].startswith("DeepHCoref")
    assert result["evidence"] == chunk.text
    assert result["citation"]


def test_benchmark_runs_all_systems_on_the_same_query(monkeypatch):
    import api.main as api

    chunk = Chunk(
        chunk_id="paper-49-chunk-0",
        document_id="paper-49",
        source="49-benchmarking_and_building_zero_shot_hindi_retrie.pdf",
        text="Hindi retrieval systems are evaluated on benchmark datasets.",
        score=0.42,
        page=2,
        section="Evaluation",
    )

    class FakeRetriever:
        def retrieve_chunks(self, query, k):
            assert query == "How are Hindi retrieval systems evaluated?"
            assert k == 3
            return [chunk]

    retriever = FakeRetriever()
    monkeypatch.setattr(api, "_bm25_searcher", lambda: retriever)
    monkeypatch.setattr(api, "_dense_searcher", lambda: retriever)
    monkeypatch.setattr(api, "_hybrid_searcher", lambda: retriever)

    response = api.benchmark(
        api.AskRequest(question="How are Hindi retrieval systems evaluated?", k=3)
    )

    assert [system.name for system in response.systems] == [
        "BM25",
        "Dense BGE-small encoder",
        "Hybrid reciprocal-rank fusion",
    ]
    assert all(system.results[0].evidence == chunk.text for system in response.systems)
    assert all(system.results[0].chunk_id == chunk.chunk_id for system in response.systems)


def test_cited_answer_uses_selected_searcher_and_returns_its_exact_evidence(monkeypatch):
    import api.main as api

    chunk = Chunk(
        chunk_id="paper-02-chunk-0",
        document_id="paper-02",
        source="02-deephcoref_a_deep_neural_coreference_resolution_.pdf",
        text="The Hindi system combines a Bi-GRU-CNN with a biaffine classifier.",
        score=0.82,
        page=1,
        section="Abstract",
    )

    class FakeRetriever:
        def retrieve_chunks(self, query, k):
            assert query == "How does DeepHCoref work?"
            assert k == 3
            return [chunk]

    monkeypatch.setattr(api, "_searcher_for_mode", lambda mode: FakeRetriever())
    monkeypatch.setattr(
        api,
        "generate_from_chunks",
        lambda question, chunks: "It combines a Bi-GRU-CNN and biaffine classifier [S1].",
    )
    response = api.cited_answer(
        api.SearchRequest(question="How does DeepHCoref work?", mode="bm25", k=3)
    )

    assert response.answer.endswith("[S1].")
    assert response.sources[0].evidence == chunk.text
    assert response.sources[0].chunk_id == chunk.chunk_id
    assert response.sources[0].title.startswith("DeepHCoref")


def test_local_showcase_defaults_are_used_only_without_explicit_qdrant_config(monkeypatch, tmp_path):
    import os

    import api.main as api

    index = tmp_path / api._SHOWCASE_INDEX_NAME
    index.mkdir()
    (index / "meta.json").write_text(
        '{"collections":{"hindi_indic_benchmark_20261025_v5":{}}}',
        encoding="utf-8",
    )
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
