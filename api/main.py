"""FastAPI service exposing the RAG pipeline."""

import csv
import hashlib
import ipaddress
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE_INDEX_NAME = "qdrant_hindi_indic_benchmark_20261025_v5"
_SHOWCASE_COLLECTION = "hindi_indic_benchmark_20261025_v5"
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 1000


def _configure_local_showcase_defaults(project_root: Path = ROOT) -> None:
    """Use the built local presentation index when no Qdrant target was configured.

    Explicit Qdrant settings and deployment's baked-index configuration always win.
    This keeps the browser demo usable when launched from a fresh PowerShell window,
    where environment variables set in another terminal are not inherited.
    """
    if any(name in os.environ for name in ("QDRANT_PATH", "QDRANT_HOST", "QDRANT_BAKED_INDEX")):
        return

    data_dir = Path(os.environ.get("RAG_DATA_DIR", project_root / "data" / "hindi_indic"))
    local_index = project_root / _SHOWCASE_INDEX_NAME
    collection = _SHOWCASE_COLLECTION
    pointer_path = data_dir / ".local_index.json"
    pointer_is_current = False
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        candidate = (project_root / pointer["index_path"]).resolve()
        if candidate.is_relative_to(project_root.resolve()) and candidate.is_dir():
            local_index = candidate
            collection = pointer["collection"]
            expected = pointer.get("pdf_state", {})
            current = {
                pdf.name: {"size": pdf.stat().st_size, "mtime_ns": pdf.stat().st_mtime_ns}
                for pdf in data_dir.glob("*.pdf")
            }
            pointer_is_current = expected == current
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not local_index.is_dir():
        return
    metadata_path = local_index / "meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if collection not in metadata.get("collections", {}):
        return
    if pointer_is_current:
        (data_dir / ".index-refresh-needed.json").unlink(missing_ok=True)

    os.environ.setdefault("QDRANT_PATH", str(local_index))
    os.environ.setdefault("RAG_COLLECTION", collection)
    if data_dir.is_dir():
        os.environ.setdefault("RAG_DATA_DIR", str(data_dir))


_configure_local_showcase_defaults()

from rag import telemetry, tracing  # noqa: E402
from rag.errors import PermanentGenerationError, TransientGenerationError  # noqa: E402
from rag.generate import answer_question, generate_from_chunks  # noqa: E402
from rag.ingest import DATA_DIR, S3_BUCKET, load_chunks  # noqa: E402
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from rag.types import Chunk  # noqa: E402

telemetry.configure_logging()
app = FastAPI(title="RAG Evaluation Pipeline")

# Per-IP limit. Best-effort and deliberately so: the window lives in process
# memory, so it resets on every cold start and is not shared between concurrent
# Lambda containers. The real ceiling on cost is the function's reserved
# concurrency (infra/main.tf); this only blunts a single impatient client.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def limit_pdf_upload_size(request: Request, call_next):
    if request.url.path == "/api/papers/upload":
        raw_length = request.headers.get("content-length")
        try:
            content_length = int(raw_length) if raw_length is not None else 0
        except ValueError:
            content_length = 0
        if content_length > MAX_PDF_BYTES + 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "PDF uploads must be 25 MB or smaller."})
    return await call_next(request)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path not in {"/ask", "/api/answer"}:
        return await call_next(request)

    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit is {RATE_LIMIT_PER_MIN} questions per minute."},
        )
    window.append(now)
    return await call_next(request)


# Registered after rate_limit, so it wraps it: a 429 is logged like anything else.
@app.middleware("http")
async def observe(request: Request, call_next):
    # Lambda's own request ID when there is one, so this line joins the REPORT line
    # Lambda writes for the same invocation — billed duration, memory, init time.
    # Locally there is no Lambda context, and a random ID serves the same purpose.
    context = request.scope.get("aws.context")
    request_id = getattr(context, "aws_request_id", None) or uuid.uuid4().hex
    record = telemetry.start(request_id=request_id, method=request.method, path=request.url.path)
    began = time.perf_counter()

    def total_ms() -> float:
        return round((time.perf_counter() - began) * 1000, 1)

    try:
        # The trace's root opens here, so retrieval and generation nest under it;
        # its ID is derived from request_id, which is what joins the two records.
        # Only /ask: a page load or a health check has nothing to show, and tracing
        # one would spend quota to bury the questions under empty traces.
        traced = (
            tracing.request(request_id)
            if request.url.path in {"/ask", "/api/answer"}
            else nullcontext(tracing.OFF)
        )
        with traced as root:
            response = await call_next(request)
            root.update(
                metadata={"status": response.status_code},
                # The same severity the log line gets, so filtering traces by level
                # and filtering logs by level find the same requests.
                level={"ERROR": "ERROR", "WARNING": "WARNING"}.get(
                    logging.getLevelName(_level(response.status_code)), "DEFAULT"
                ),
            )
    except Exception:
        # A retrieval failure is deliberately not caught by ask() — it is a
        # deployment fault. It is still a 500, logged at ERROR with the traceback;
        # what changes is that the visitor gets the ID. Re-raising here handed it to
        # Starlette, whose bare "Internal Server Error" dropped the header — so the
        # one failure that most needs reporting was the one that could not be.
        record.update(status=500, total_ms=total_ms())
        telemetry.logger.error("request", extra={"fields": record}, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error. Quote request {request_id} when reporting it."},
            headers={"X-Request-Id": request_id},
        )

    record.update(status=response.status_code, total_ms=total_ms())
    telemetry.logger.log(_level(response.status_code), "request", extra={"fields": record})
    # Returned to the caller, so "it failed" can arrive with the one string that
    # finds the line.
    response.headers["X-Request-Id"] = request_id
    return response


def _level(status: int) -> int:
    # 503 is the service working as designed under throttling: worth seeing, not
    # worth paging about. 502 and 500 mean something is broken.
    if status >= 500 and status != 503:
        return logging.ERROR
    if status in (429, 503):
        return logging.WARNING
    return logging.INFO


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    citation: str
    title: str = ""
    authors: str = ""
    year: int = 0
    section: str | None = None
    score: float
    evidence: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field(default="bm25", pattern="^(bm25|dense|hybrid)$")
    k: int = Field(default=5, ge=1, le=10)


class SearchSource(BaseModel):
    chunk_id: str
    citation: str
    title: str
    authors: str
    year: int
    section: str | None = None
    score: float
    evidence: str


class SearchResponse(BaseModel):
    question: str
    mode: str
    results: list[SearchSource]


class BenchmarkSystem(BaseModel):
    name: str
    latency_ms: float
    results: list[SearchSource]


class BenchmarkResponse(BaseModel):
    question: str
    initialization_ms: float
    systems: list[BenchmarkSystem]


class BenchmarkRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


class QuestionDraftRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    expected_claim: str = Field(min_length=8, max_length=700)
    source_pdf: str = Field(min_length=1, max_length=200)
    page: int = Field(ge=1, le=MAX_PDF_PAGES)
    section: str = Field(default="", max_length=120)
    passage_text: str = Field(min_length=40, max_length=2000)


CATALOG_PATH = ROOT / "scripts" / "hindi_indic_papers.csv"
EVAL_PATH = ROOT / "evaluation" / "eval_set.jsonl"
QUESTION_DRAFT_PATH = ROOT / "evaluation" / "question_drafts.jsonl"
METRICS_PATH = ROOT / "evaluation" / "hindi_indic_baseline.json"
INDEX_PENDING_PATH = DATA_DIR / ".index-refresh-needed.json"
_CONTENT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _bm25_searcher() -> BM25Retriever:
    # An uploaded PDF is immediately available to local BM25 without waiting for
    # vector encoding. Dense/hybrid use the last built Qdrant snapshot until the
    # user creates a fresh versioned index with scripts/rebuild_local_index.py.
    if _pending_index_files():
        return BM25Retriever(
            [
                Chunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    source=row["source"],
                    text=row["text"],
                    score=0.0,
                    page=row["page"],
                    section=row["section"],
                )
                for row in load_chunks()
            ]
        )
    # Otherwise reuse the exact passages stored in Qdrant for parity with dense.
    store = DenseRetriever(load_model=False)
    try:
        return BM25Retriever(store.all_chunks())
    finally:
        store._client.close()


@lru_cache(maxsize=1)
def _dense_searcher() -> DenseRetriever:
    # Optional neural baseline; model and Qdrant client are cached after selection.
    return DenseRetriever()


@lru_cache(maxsize=1)
def _hybrid_searcher() -> HybridRetriever:
    # Share exactly the same dense and BM25 instances used by the other baselines.
    # Build BM25 first because embedded Qdrant permits one open client per path.
    bm25 = _bm25_searcher()
    dense = _dense_searcher()
    return HybridRetriever(dense=dense, bm25=bm25)


def _catalog_rows() -> list[dict[str, str]]:
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _pending_index_files() -> set[str]:
    try:
        data = json.loads(INDEX_PENDING_PATH.read_text(encoding="utf-8"))
        return {Path(name).name.casefold() for name in data.get("files", [])}
    except (OSError, ValueError, TypeError):
        return set()


def _local_admin_only(request: Request) -> None:
    """Keep filesystem-writing endpoints unavailable to remote visitors/deployments."""
    host = request.client.host if request.client else ""
    if host == "testclient":
        return
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    origin = request.headers.get("origin")
    same_origin = not origin or urlparse(origin).netloc.casefold() == request.url.netloc.casefold()
    if not local or not same_origin or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        raise HTTPException(status_code=403, detail="Content management is available in local development only.")


def _filename_for_paper(paper_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48] or "paper"
    return f"{paper_id.zfill(2)}-{slug}.pdf"


def _atomic_write_catalog(rows: list[dict[str, str]]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CATALOG_PATH.with_name(f".{CATALOG_PATH.name}.{uuid.uuid4().hex}.tmp")
    fields = ["id", "title", "authors", "year", "topic", "why_in_scope", "source_url", "pdf_url", "license_terms", "status"]
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, CATALOG_PATH)
    finally:
        temp_path.unlink(missing_ok=True)


def _add_pending_index_file(filename: str) -> None:
    INDEX_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    pending = _pending_index_files()
    pending.add(filename.casefold())
    temp_path = INDEX_PENDING_PATH.with_name(f".{INDEX_PENDING_PATH.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps({"files": sorted(pending), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, INDEX_PENDING_PATH)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_pdf(content: bytes) -> tuple[int, int]:
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The selected file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported.")
        page_count = len(reader.pages)
        if page_count < 1 or page_count > MAX_PDF_PAGES:
            raise HTTPException(status_code=422, detail=f"PDF must contain 1–{MAX_PDF_PAGES} pages.")
        extracted_chars = sum(len(page.extract_text() or "") for page in reader.pages)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The PDF could not be read. Check that it is not damaged.") from exc
    if extracted_chars < 100:
        raise HTTPException(status_code=422, detail="This PDF has too little extractable text. Scanned PDFs need OCR before upload.")
    return page_count, extracted_chars


def _next_paper_id(rows: list[dict[str, str]]) -> str:
    numeric_ids = [int(row["id"]) for row in rows if row.get("id", "").isdigit()]
    return str(max(numeric_ids, default=0) + 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalised_passage(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _paper_for_source(source: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    name = Path(source).name.casefold()
    for row in rows:
        slug = "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in row["title"]).split())
        prefix = f"{row['id'].zfill(2)}-"
        if name.startswith(prefix) and slug[:20].replace("-", "_") in name:
            return row
    # Filenames are stable but are generated from titles with a different slug
    # normalization; ID prefixes are the unambiguous key for this corpus.
    for row in rows:
        if name.startswith(f"{row['id'].zfill(2)}-"):
            return row
    return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def overview() -> dict:
    rows = _catalog_rows()
    pdfs = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    try:
        questions = sum(1 for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        questions = 0
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metrics = None
    pending = _pending_index_files()
    metrics = dict(metrics) if isinstance(metrics, dict) else None
    if metrics is not None:
        metrics["metrics_are_current"] = bool(
            metrics.get("metrics_are_current")
            and metrics.get("corpus_papers") == len(pdfs)
            and not pending
        )
    try:
        question_drafts = sum(
            1 for line in QUESTION_DRAFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    except OSError:
        question_drafts = 0
    return {
        "catalog_papers": len(rows),
        "local_pdfs": len(pdfs),
        "evaluation_questions": questions,
        "question_drafts": question_drafts,
        "index_refresh_pending": bool(pending),
        "pending_papers": sorted(pending),
        "metrics": metrics,
        "generation": "optional-bedrock",
    }


@app.get("/api/papers")
def papers() -> list[dict]:
    rows = _catalog_rows()
    pdfs = {p.name.casefold() for p in DATA_DIR.glob("*.pdf")} if DATA_DIR.exists() else set()
    pending = _pending_index_files()
    output = []
    for row in rows:
        filename = _filename_for_paper(row["id"], row["title"]).casefold()
        downloaded = filename in pdfs
        output.append(
            {
                "id": row["id"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "topic": row["topic"],
                "status": row["status"],
                "source_url": row["source_url"],
                "downloaded": downloaded,
                "indexed": downloaded and filename not in pending,
                "index_refresh_pending": downloaded and filename in pending,
                "local_pdf": filename if downloaded else None,
            }
        )
    return output


@app.post("/api/papers/upload", status_code=201)
def upload_paper(
    request: Request,
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(min_length=3, max_length=240)],
    authors: Annotated[str, Form(min_length=2, max_length=500)],
    year: Annotated[int, Form(ge=1900, le=2100)],
    topic: Annotated[str, Form(min_length=2, max_length=240)],
    why_in_scope: Annotated[str, Form(min_length=5, max_length=600)],
    license_notes: Annotated[str, Form(min_length=5, max_length=600)],
    source_url: Annotated[str, Form(max_length=500)] = "",
) -> dict:
    """Add one validated local PDF and its provenance to the paper catalogue."""
    _local_admin_only(request)
    if S3_BUCKET:
        raise HTTPException(status_code=409, detail="Local uploads are disabled while S3 ingestion is configured.")
    if not file.filename or Path(file.filename).suffix.casefold() != ".pdf":
        raise HTTPException(status_code=415, detail="Choose a file with a .pdf extension.")
    if source_url:
        parsed = urlparse(source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise HTTPException(status_code=422, detail="Source links must use HTTPS.")

    content = file.file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDFs must be 25 MB or smaller.")
    page_count, extracted_chars = _validate_pdf(content)
    digest = hashlib.sha256(content).hexdigest()

    with _CONTENT_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for existing in DATA_DIR.glob("*.pdf"):
            existing_digest = _file_sha256(existing)
            if existing_digest == digest:
                raise HTTPException(status_code=409, detail=f"This PDF is already in the collection as {existing.name}.")

        rows = _catalog_rows()
        paper_id = _next_paper_id(rows)
        filename = _filename_for_paper(paper_id, title)
        destination = DATA_DIR / filename
        while destination.exists():
            paper_id = str(int(paper_id) + 1)
            filename = _filename_for_paper(paper_id, title)
            destination = DATA_DIR / filename
        new_row = {
            "id": paper_id,
            "title": title.strip(),
            "authors": authors.strip(),
            "year": str(year),
            "topic": topic.strip(),
            "why_in_scope": why_in_scope.strip(),
            "source_url": source_url.strip(),
            "pdf_url": "",
            "license_terms": (
                f"User-supplied access/rights note: {license_notes.strip().rstrip(' .;')}. "
                "Not independently verified; local upload only."
            ),
            "status": "user upload; rights not independently verified",
        }
        catalog_written = False
        try:
            with destination.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_write_catalog([*rows, new_row])
            catalog_written = True
            _add_pending_index_file(filename)
        except Exception as exc:
            if catalog_written:
                try:
                    _atomic_write_catalog(rows)
                except OSError:
                    pass
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="The PDF could not be added to the local catalogue.") from exc

    _bm25_searcher.cache_clear()
    _hybrid_searcher.cache_clear()
    return {
        "id": paper_id,
        "title": title.strip(),
        "filename": filename,
        "pages": page_count,
        "extracted_characters": extracted_chars,
        "bm25_ready": True,
        "vector_index_pending": True,
        "message": "Paper added. BM25 will include it on the next search; rebuild the vector index to update dense and hybrid retrieval.",
    }


@app.get("/api/questions/drafts")
def get_question_drafts(request: Request) -> list[dict]:
    _local_admin_only(request)
    if not QUESTION_DRAFT_PATH.exists():
        return []
    drafts = []
    for line_number, line in enumerate(QUESTION_DRAFT_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            drafts.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Question draft file is invalid near line {line_number}.") from exc
    return drafts


@app.post("/api/questions/drafts", status_code=201)
def add_question_draft(request: Request, draft: QuestionDraftRequest) -> dict:
    """Store a source-first evaluation question without changing the active benchmark."""
    _local_admin_only(request)
    filename = Path(draft.source_pdf).name
    if filename != draft.source_pdf or Path(filename).suffix.casefold() != ".pdf":
        raise HTTPException(status_code=422, detail="Choose a PDF from the local collection.")
    pdf_path = DATA_DIR / filename
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="That PDF is not available in the local collection.")
    paper = next(
        (row for row in _catalog_rows() if _filename_for_paper(row["id"], row["title"]).casefold() == filename.casefold()),
        None,
    )
    if paper is None:
        raise HTTPException(status_code=422, detail="The PDF needs a catalogue record before it can support a question.")
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted or draft.page > len(reader.pages):
            raise HTTPException(status_code=422, detail="Select a page that exists in the chosen PDF.")
        page_text = reader.pages[draft.page - 1].extract_text() or ""
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not read the selected PDF page.") from exc
    if _normalised_passage(draft.passage_text) not in _normalised_passage(page_text):
        raise HTTPException(
            status_code=422,
            detail="The evidence text was not found on that PDF page. Copy a passage from the source PDF and try again.",
        )

    record = {
        "id": f"draft-{uuid.uuid4().hex[:12]}",
        "query": draft.question.strip(),
        "gold_evidence": [
            {
                "source_id": paper["id"],
                "source_url": paper["source_url"],
                "local_pdf": filename,
                "location": f"page {draft.page}" + (f", {draft.section.strip()}" if draft.section.strip() else ""),
                "page": draft.page,
                "passage_text": draft.passage_text.strip(),
            }
        ],
        "expected_claim": draft.expected_claim.strip(),
        "question_review_status": "source_passage_verified_on_page; manual_question_review_pending",
        "active_in_benchmark": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _CONTENT_LOCK:
        QUESTION_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = QUESTION_DRAFT_PATH.read_text(encoding="utf-8").splitlines() if QUESTION_DRAFT_PATH.exists() else []
        temp_path = QUESTION_DRAFT_PATH.with_name(f".{QUESTION_DRAFT_PATH.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
                for line in existing:
                    if line.strip():
                        stream.write(line + "\n")
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, QUESTION_DRAFT_PATH)
        finally:
            temp_path.unlink(missing_ok=True)
    return record


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    rows = _catalog_rows()
    retriever = _searcher_for_mode(request.mode)
    chunks = retriever.retrieve_chunks(request.question, request.k)
    results = _format_search_sources(chunks, rows)
    return SearchResponse(question=request.question, mode=request.mode, results=results)


def _searcher_for_mode(mode: str):
    return {"bm25": _bm25_searcher, "dense": _dense_searcher, "hybrid": _hybrid_searcher}[mode]()


def _format_search_sources(chunks, rows: list[dict[str, str]]) -> list[SearchSource]:
    results = []
    for chunk in chunks:
        paper = _paper_for_source(chunk.source, rows) or {}
        results.append(
            SearchSource(
                chunk_id=chunk.chunk_id,
                citation=chunk.cite(),
                title=paper.get("title", Path(chunk.source).stem),
                authors=paper.get("authors", ""),
                year=int(paper.get("year", 0) or 0),
                section=chunk.section,
                score=round(chunk.score, 4),
                evidence=chunk.text,
            )
        )
    return results


def _format_answer_sources(chunks) -> list[Source]:
    rows = _catalog_rows()
    sources = []
    for chunk in chunks:
        paper = _paper_for_source(chunk.source, rows) or {}
        sources.append(
            Source(
                chunk_id=chunk.chunk_id,
                citation=chunk.cite(),
                title=paper.get("title", Path(chunk.source).stem),
                authors=paper.get("authors", ""),
                year=int(paper.get("year", 0) or 0),
                section=chunk.section,
                score=round(chunk.score, 3),
                evidence=chunk.text,
            )
        )
    return sources


def _generation_error_detail(exc: PermanentGenerationError) -> str:
    """Translate known Bedrock failures into safe, actionable UI guidance.

    Never forward AWS exception messages: they can contain account or resource
    identifiers. The code is from a closed set of Bedrock/API error categories.
    """
    if exc.code == "NO_AWS_IDENTITY":
        return (
            "No AWS identity is available to this app. Sign in with your approved AWS "
            "SSO profile (for example, `aws sso login --profile <name>`) or run the app "
            "with an authorized AWS role; then restart it. Do not paste access keys here."
        )
    if exc.code == "AccessDeniedException":
        return (
            "AWS denied answer generation. Ask the AWS administrator to allow "
            "`bedrock:InvokeModel` for the configured Nova Lite inference profile in "
            "the selected region, and confirm model access is enabled."
        )
    if exc.code in {"ResourceNotFoundException", "ValidationException"}:
        return (
            "Bedrock could not use the configured model. Check `BEDROCK_MODEL_ID` and "
            "`AWS_REGION`, and confirm the inference profile is available to the account."
        )
    if exc.code == "INVALID_RESPONSE":
        return "Bedrock returned an unusable answer. Try again; if it repeats, check the configured model."
    if exc.code == "CITATION_MISSING_CITATION":
        return (
            "Bedrock returned an answer without recognizable [S#] citations, so the draft "
            "was withheld. Try a more specific question or another ranking method."
        )
    if exc.code == "CITATION_INVALID_SOURCE_LABEL":
        return (
            "Bedrock cited a passage label that was not in the retrieved evidence, so the "
            "draft was withheld. Try another ranking method or fewer passages."
        )
    if exc.code == "CITATION_UNCITED_SENTENCE":
        return (
            "At least one sentence in Bedrock's answer had no passage citation, so the draft "
            "was withheld. Try a more specific question or another ranking method."
        )
    return (
        "Bedrock rejected answer generation. Check the configured model, region, and "
        "runtime identity permissions. The detailed AWS error is kept out of the browser response."
    )


@app.post("/api/answer", response_model=AskResponse)
def cited_answer(request: SearchRequest) -> AskResponse:
    """Generate a cited answer using the chosen retriever and its exact passages."""
    chunks = _searcher_for_mode(request.mode).retrieve_chunks(request.question, request.k)
    if not chunks:
        answer = generate_from_chunks(request.question, chunks)
    else:
        try:
            answer = generate_from_chunks(request.question, chunks)
        except TransientGenerationError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Bedrock is temporarily unavailable ({exc.code}). Please try again shortly."
                    if exc.code
                    else "Bedrock is temporarily unavailable. Please try again shortly."
                ),
                headers={"Retry-After": "5"},
            ) from exc
        except PermanentGenerationError as exc:
            raise HTTPException(
                status_code=502,
                detail=_generation_error_detail(exc),
            ) from exc
    return AskResponse(answer=answer, sources=_format_answer_sources(chunks))


@app.post("/api/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run the identical query through all three retrieval baselines."""
    rows = _catalog_rows()
    setup_began = time.perf_counter()
    bm25 = _bm25_searcher()
    dense = _dense_searcher()
    hybrid = _hybrid_searcher()
    initialization_ms = round((time.perf_counter() - setup_began) * 1000, 1)
    systems = [
        ("BM25", bm25),
        ("Dense BGE-small encoder", dense),
        ("Hybrid reciprocal-rank fusion", hybrid),
    ]
    output = []
    for name, retriever in systems:
        began = time.perf_counter()
        chunks = retriever.retrieve_chunks(request.question, request.k)
        elapsed = round((time.perf_counter() - began) * 1000, 1)
        output.append(
            BenchmarkSystem(
                name=name,
                latency_ms=elapsed,
                results=_format_search_sources(chunks, rows),
            )
        )
    return BenchmarkResponse(
        question=request.question,
        initialization_ms=initialization_ms,
        systems=output,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # Generation failures become a status that says whether retrying helps, rather
    # than an unhandled exception and a bare 500. The demo page already retries on
    # 503, so a throttled request reaches the visitor as "waking the model up"
    # instead of an error box. Retrieval failures are deliberately NOT caught: a
    # broken index is a deployment fault, and masking it would hide the one thing
    # worth paging about.
    # The question is logged; the client address is not. The question is what makes
    # a bad answer diagnosable, and this is a public demo over published papers.
    # The address would identify a person and diagnoses nothing the rate limiter
    # does not already handle.
    telemetry.note(question=request.question)
    tracing.annotate(input=request.question)
    try:
        answer, chunks = answer_question(request.question, request.k)
    except TransientGenerationError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Bedrock is temporarily unavailable ({exc.code}). Please try again in a moment."
                if exc.code
                else "Bedrock is temporarily unavailable. Please try again in a moment."
            ),
            headers={"Retry-After": "5"},
        ) from exc
    except PermanentGenerationError as exc:
        # 502, not 500: this service is working and its upstream is not.
        detail = _generation_error_detail(exc)
        if not (exc.code or "").startswith("CITATION_"):
            detail += " Retrying will not help until the configuration is fixed."
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    tracing.annotate(output=answer)
    return AskResponse(
        answer=answer,
        sources=_format_answer_sources(chunks),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = (ROOT / "api" / "templates" / "index.html").read_text(encoding="utf-8")
