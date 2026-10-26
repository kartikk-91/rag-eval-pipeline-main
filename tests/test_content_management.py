"""Local uploads and source-first question drafts must preserve the active corpus."""

import csv
import json
from io import BytesIO

from starlette.datastructures import UploadFile
from starlette.requests import Request

CATALOG_FIELDS = [
    "id", "title", "authors", "year", "topic", "why_in_scope", "source_url",
    "pdf_url", "license_terms", "status",
]


def _local_request():
    return Request({
        "type": "http", "method": "POST", "path": "/",
        "headers": [(b"host", b"testclient")], "query_string": b"",
        "server": ("testclient", 80), "scheme": "http", "client": ("testclient", 50000),
    })


def _configure_local_files(api, tmp_path, monkeypatch):
    data_dir = tmp_path / "papers"
    catalog_path = tmp_path / "papers.csv"
    drafts_path = tmp_path / "question_drafts.jsonl"
    pending_path = data_dir / ".index-refresh-needed.json"
    data_dir.mkdir()
    with catalog_path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=CATALOG_FIELDS).writeheader()
    monkeypatch.setattr(api, "DATA_DIR", data_dir)
    monkeypatch.setattr(api, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(api, "QUESTION_DRAFT_PATH", drafts_path)
    monkeypatch.setattr(api, "INDEX_PENDING_PATH", pending_path)
    monkeypatch.setattr(api, "S3_BUCKET", None)
    return data_dir, catalog_path, drafts_path, pending_path


def test_upload_adds_pdf_and_catalog_record_without_overwriting(monkeypatch, tmp_path):
    import api.main as api

    data_dir, catalog_path, _, pending_path = _configure_local_files(api, tmp_path, monkeypatch)
    monkeypatch.setattr(api, "_validate_pdf", lambda content: (3, 600))
    content = b"%PDF-1.7 local test PDF bytes"
    fields = {
        "title": "New Hindi Coreference System", "authors": "A. Author; B. Researcher",
        "year": 2025, "topic": "Hindi coreference",
        "why_in_scope": "A system paper for Hindi coreference research.",
        "license_notes": "Author permission for local research use.",
        "source_url": "https://example.org/paper",
    }

    response = api.upload_paper(
        _local_request(), UploadFile(filename="paper.pdf", file=BytesIO(content)), **fields
    )
    assert response["bm25_ready"] is True
    assert response["vector_index_pending"] is True
    assert (data_dir / response["filename"]).read_bytes() == content
    assert pending_path.exists()
    with catalog_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["title"] == fields["title"]
    assert "not independently verified" in rows[0]["license_terms"].casefold()
    monkeypatch.setattr(api, "load_chunks", lambda: [{
        "chunk_id": "uploaded-chunk", "document_id": "uploaded-paper", "source": response["filename"],
        "page": 1, "section": "Abstract", "text": "A Hindi coreference system with strong results.",
    }])
    result = api._bm25_searcher().retrieve_chunks("Hindi coreference results", 1)
    assert result[0].source == response["filename"]
    api._bm25_searcher.cache_clear()

    try:
        api.upload_paper(
            _local_request(), UploadFile(filename="paper.pdf", file=BytesIO(content)), **fields
        )
    except api.HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("Identical PDFs must be rejected")
    assert len(list(data_dir.glob("*.pdf"))) == 1


def test_question_draft_requires_a_passage_from_selected_page(monkeypatch, tmp_path):
    import api.main as api

    data_dir, _, drafts_path, _ = _configure_local_files(api, tmp_path, monkeypatch)
    title = "Hindi Coreference Findings"
    filename = api._filename_for_paper("7", title)
    (data_dir / filename).write_bytes(b"%PDF-1.7 test placeholder")
    api._atomic_write_catalog([{
        "id": "7", "title": title, "authors": "A. Author", "year": "2023",
        "topic": "Hindi coreference", "why_in_scope": "Relevant Hindi coreference evidence.",
        "source_url": "https://example.org/source", "pdf_url": "", "license_terms": "Local test",
        "status": "user upload",
    }])

    exact_text = (
        "The Hindi model resolves coreference links across the document using contextual representations."
    )

    class FakePage:
        def extract_text(self):
            return f"Abstract\n{exact_text}\nAdditional text."

    class FakeReader:
        is_encrypted = False
        pages = [FakePage()]

        def __init__(self, source, strict=False):
            pass

    monkeypatch.setattr(api, "PdfReader", FakeReader)
    payload = {
        "question": "How does the model resolve Hindi coreference?",
        "expected_claim": "It resolves document-level links using contextual representations.",
        "source_pdf": filename,
        "page": 1,
        "section": "Abstract",
        "passage_text": exact_text,
    }
    response = api.add_question_draft(_local_request(), api.QuestionDraftRequest(**payload))
    assert response["active_in_benchmark"] is False
    assert "manual_question_review_pending" in response["question_review_status"]
    assert exact_text in drafts_path.read_text(encoding="utf-8")

    payload["passage_text"] = "A passage that is not present in the selected PDF page at all."
    try:
        api.add_question_draft(_local_request(), api.QuestionDraftRequest(**payload))
    except api.HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("Evidence absent from the selected PDF page must be rejected")
    assert len(drafts_path.read_text(encoding="utf-8").splitlines()) == 1


def test_content_management_rejects_non_loopback_requests():
    import api.main as api

    scope = {
        "type": "http", "method": "POST", "path": "/api/questions/drafts",
        "headers": [(b"host", b"papers.example.com"), (b"origin", b"https://attacker.example")],
        "query_string": b"", "server": ("papers.example.com", 443), "scheme": "https",
        "client": ("203.0.113.10", 50000),
    }
    request = Request(scope)
    try:
        api._local_admin_only(request)
    except api.HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("Remote content-management requests must be rejected")


def test_content_management_rejects_cross_origin_loopback_requests():
    import api.main as api

    request = Request({
        "type": "http", "method": "POST", "path": "/api/papers/upload",
        "headers": [(b"host", b"127.0.0.1:8000"), (b"origin", b"https://attacker.example")],
        "query_string": b"", "server": ("127.0.0.1", 8000), "scheme": "http",
        "client": ("127.0.0.1", 50000),
    })
    try:
        api._local_admin_only(request)
    except api.HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("Cross-origin content-management requests must be rejected")


def test_upload_route_accepts_multipart_form_schema():
    import api.main as api

    schema = api.app.openapi()["paths"]["/api/papers/upload"]["post"]["requestBody"]
    assert "multipart/form-data" in schema["content"]


def test_overview_marks_the_old_benchmark_stale_after_an_upload(monkeypatch, tmp_path):
    import api.main as api

    data_dir, _, drafts_path, pending_path = _configure_local_files(api, tmp_path, monkeypatch)
    (data_dir / "new-paper.pdf").write_bytes(b"local PDF placeholder")
    pending_path.write_text(json.dumps({"files": ["new-paper.pdf"]}), encoding="utf-8")
    eval_path = tmp_path / "eval_set.jsonl"
    eval_path.write_text('{"id":"one"}\n', encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics_are_current": True, "corpus_papers": 45, "indexed_chunks": 2608}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "EVAL_PATH", eval_path)
    monkeypatch.setattr(api, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(api, "QUESTION_DRAFT_PATH", drafts_path)

    result = api.overview()
    assert result["local_pdfs"] == 1
    assert result["index_refresh_pending"] is True
    assert result["metrics"]["metrics_are_current"] is False


def test_local_startup_selects_verified_versioned_index_pointer(monkeypatch, tmp_path):
    import api.main as api

    data_dir = tmp_path / "data" / "hindi_indic"
    data_dir.mkdir(parents=True)
    pdf_path = data_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    index_path = tmp_path / "qdrant_hindi_indic_added_test"
    index_path.mkdir()
    (index_path / "meta.json").write_text(
        '{"collections":{"hindi_indic_added_test":{}}}', encoding="utf-8"
    )
    pdf_state = {
        pdf_path.name: {"size": pdf_path.stat().st_size, "mtime_ns": pdf_path.stat().st_mtime_ns}
    }
    (data_dir / ".local_index.json").write_text(json.dumps({
        "index_path": index_path.name,
        "collection": "hindi_indic_added_test",
        "pdf_state": pdf_state,
    }), encoding="utf-8")
    pending = data_dir / ".index-refresh-needed.json"
    pending.write_text('{"files":["paper.pdf"]}', encoding="utf-8")
    for name in ("QDRANT_PATH", "QDRANT_HOST", "QDRANT_BAKED_INDEX", "RAG_COLLECTION", "RAG_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)

    api._configure_local_showcase_defaults(tmp_path)

    assert api.os.environ["QDRANT_PATH"] == str(index_path)
    assert api.os.environ["RAG_COLLECTION"] == "hindi_indic_added_test"
    assert not pending.exists()
