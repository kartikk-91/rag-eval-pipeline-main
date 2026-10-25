"""FastAPI service exposing the RAG pipeline."""

import csv
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE_INDEX_NAME = "qdrant_hindi_indic_showcase_20261025_v2"
_SHOWCASE_COLLECTION = "hindi_indic_showcase_20261025_v2"


def _configure_local_showcase_defaults(project_root: Path = ROOT) -> None:
    """Use the built local presentation index when no Qdrant target was configured.

    Explicit Qdrant settings and deployment's baked-index configuration always win.
    This keeps the browser demo usable when launched from a fresh PowerShell window,
    where environment variables set in another terminal are not inherited.
    """
    if any(name in os.environ for name in ("QDRANT_PATH", "QDRANT_HOST", "QDRANT_BAKED_INDEX")):
        return

    local_index = project_root / _SHOWCASE_INDEX_NAME
    if not local_index.is_dir():
        return

    os.environ.setdefault("QDRANT_PATH", str(local_index))
    os.environ.setdefault("RAG_COLLECTION", _SHOWCASE_COLLECTION)
    local_papers = project_root / "data" / "hindi_indic"
    if local_papers.is_dir():
        os.environ.setdefault("RAG_DATA_DIR", str(local_papers))


_configure_local_showcase_defaults()

from rag import telemetry, tracing  # noqa: E402
from rag.errors import PermanentGenerationError, TransientGenerationError  # noqa: E402
from rag.generate import answer_question  # noqa: E402
from rag.ingest import DATA_DIR  # noqa: E402
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

telemetry.configure_logging()
app = FastAPI(title="RAG Evaluation Pipeline")

# Per-IP limit. Best-effort and deliberately so: the window lives in process
# memory, so it resets on every cold start and is not shared between concurrent
# Lambda containers. The real ceiling on cost is the function's reserved
# concurrency (infra/main.tf); this only blunts a single impatient client.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path != "/ask":
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
        traced = tracing.request(request_id) if request.url.path == "/ask" else nullcontext(tracing.OFF)
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
    section: str | None = None
    score: float
    evidence: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field(default="bm25", pattern="^(bm25|hybrid)$")
    k: int = Field(default=5, ge=1, le=10)


class SearchSource(BaseModel):
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


CATALOG_PATH = ROOT / "scripts" / "hindi_indic_papers.csv"
EVAL_PATH = ROOT / "evaluation" / "eval_set.jsonl"
METRICS_PATH = ROOT / "evaluation" / "hindi_indic_baseline.json"


@lru_cache(maxsize=1)
def _bm25_searcher() -> BM25Retriever:
    # The presentation demo's default is entirely local and does not load an
    # embedding model or contact AWS. It reads the already built Qdrant payloads.
    store = DenseRetriever(load_model=False)
    try:
        return BM25Retriever(store.all_chunks())
    finally:
        store._client.close()


@lru_cache(maxsize=1)
def _hybrid_searcher() -> HybridRetriever:
    # Optional comparison mode; dense weights are loaded only when selected.
    return HybridRetriever()


def _catalog_rows() -> list[dict[str, str]]:
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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
    return {
        "catalog_papers": len(rows),
        "local_pdfs": len(pdfs),
        "evaluation_questions": questions,
        "metrics": metrics,
        "generation": "optional-bedrock",
    }


@app.get("/api/papers")
def papers() -> list[dict]:
    rows = _catalog_rows()
    pdfs = {p.name.casefold() for p in DATA_DIR.glob("*.pdf")} if DATA_DIR.exists() else set()
    output = []
    for row in rows:
        slug = "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in row["title"]).split())[:48]
        filename = f"{row['id'].zfill(2)}-{slug}.pdf".casefold()
        output.append(
            {
                "id": row["id"],
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "topic": row["topic"],
                "status": row["status"],
                "source_url": row["source_url"],
                "downloaded": filename in pdfs,
            }
        )
    return output


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    rows = _catalog_rows()
    retriever = _bm25_searcher() if request.mode == "bm25" else _hybrid_searcher()
    chunks = retriever.retrieve_chunks(request.question, request.k)
    results = []
    for chunk in chunks:
        paper = _paper_for_source(chunk.source, rows) or {}
        results.append(
            SearchSource(
                citation=chunk.cite(),
                title=paper.get("title", Path(chunk.source).stem),
                authors=paper.get("authors", ""),
                year=int(paper.get("year", 0) or 0),
                section=chunk.section,
                score=round(chunk.score, 4),
                evidence=chunk.text,
            )
        )
    return SearchResponse(question=request.question, mode=request.mode, results=results)


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
            detail=f"{exc} Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from exc
    except PermanentGenerationError as exc:
        # 502, not 500: this service is working and its upstream is not.
        raise HTTPException(
            status_code=502,
            detail=f"{exc} Retrying will not help; this needs fixing on the server.",
        ) from exc

    tracing.annotate(output=answer)
    return AskResponse(
        answer=answer,
        sources=[
            Source(
                chunk_id=c.chunk_id,
                citation=c.cite(),
                section=c.section,
                score=round(c.score, 3),
                evidence=c.text,
            )
            for c in chunks
        ],
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#10151d"><title>Indic Research Desk</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#17212c;background:#f4f6f7;font-synthesis:none}
*{box-sizing:border-box}body{margin:0}.shell{min-height:100vh;display:grid;grid-template-columns:244px 1fr}
aside{background:#101820;color:#dce4e7;padding:25px 16px;display:flex;flex-direction:column;gap:34px}
.brand{display:flex;align-items:center;gap:12px;padding:0 8px}.mark{width:38px;height:38px;border-radius:12px;background:#b9ec7b;color:#142018;display:grid;place-items:center;font-weight:900}
.brand strong{display:block;font-size:14px;letter-spacing:.02em}.brand small{color:#8f9ca4}.nav{display:grid;gap:7px}.nav button{background:transparent;border:0;color:#a9b5bb;padding:11px 12px;text-align:left;border-radius:9px;font:inherit;cursor:pointer}
.nav button.active,.nav button:hover{background:#202c34;color:#fff}.side-note{margin-top:auto;border:1px solid #2b3942;border-radius:12px;padding:14px;font-size:12px;color:#9caab1;line-height:1.55}
main{min-width:0;padding:30px clamp(18px,4vw,58px) 64px}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:33px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#61716f;font-weight:800}.top h1{font-size:22px;margin:4px 0 0;letter-spacing:-.03em}.pill{display:inline-flex;align-items:center;gap:7px;background:#e7f4dc;color:#326021;border-radius:999px;padding:8px 11px;font-size:12px;font-weight:700}.dot{width:7px;height:7px;border-radius:50%;background:#5caa3a}
.view{display:none}.view.active{display:block}.hero{background:#17271f;color:#f3f5ea;border-radius:20px;padding:clamp(24px,4vw,42px);position:relative;overflow:hidden}.hero:after{content:"अ";position:absolute;font-size:230px;right:4%;top:-90px;color:#ffffff08;font-weight:800;pointer-events:none}.hero h2{max-width:620px;font-size:clamp(29px,4vw,43px);line-height:1.08;letter-spacing:-.045em;margin:10px 0 13px}.hero p{color:#c1ccc0;max-width:590px;margin:0;line-height:1.6}.hero .eyebrow{color:#b9ec7b}
.searchbox{background:#fff;border:1px solid #dce3e3;border-radius:15px;padding:10px;margin-top:21px;position:relative;z-index:1;box-shadow:0 12px 30px #07150f12}.searchrow{display:flex;gap:9px}.searchrow input{flex:1;min-width:80px;border:0;outline:0;padding:13px 12px;font:inherit;color:#17212c;background:transparent}.searchrow button,.primary{background:#b9ec7b;border:0;border-radius:10px;padding:12px 18px;color:#182419;font:inherit;font-weight:700;font-size:14px;cursor:pointer}.searchrow button:hover,.primary:hover{background:#a9de68}.answer-button{background:#fff!important;border:1px solid #dbe4dc!important;color:#355640!important}.answer-button:hover{background:#f1f7ee!important}.controls{display:flex;align-items:center;gap:10px;padding:4px 12px 8px;color:#677572;font-size:12px}.controls select{border:0;background:#f1f4f1;padding:7px 9px;border-radius:7px;color:#33403a;font:inherit}.hint{margin-left:auto}.examples{display:flex;flex-wrap:wrap;gap:8px;margin:13px 0 0}.examples button{border:1px solid #d7dfda;background:#fff;border-radius:999px;padding:8px 11px;color:#42514a;font:inherit;font-size:12px;cursor:pointer}.examples button:hover{border-color:#6b9870;color:#1e5d36}
.statgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0 28px}.stat{background:#fff;border:1px solid #e0e6e5;border-radius:13px;padding:15px 17px}.stat span{display:block;color:#72807c;font-size:11px}.stat strong{display:block;font-size:25px;margin-top:4px;letter-spacing:-.04em}.stat em{font-style:normal;color:#799b67;font-size:11px}
.sectionhead{display:flex;align-items:end;justify-content:space-between;gap:14px;margin:27px 0 14px}.sectionhead h2{font-size:18px;margin:0;letter-spacing:-.025em}.sectionhead p{font-size:12px;color:#77827e;margin:4px 0 0}.textlink{color:#4a7251;border:0;background:none;font:inherit;font-weight:600;font-size:12px;cursor:pointer}
.results{display:grid;gap:12px}.answer-panel{margin:18px 0}.answer-text{white-space:pre-wrap;line-height:1.65;color:#35423d}.answer-source{margin-top:10px}.answer-source summary{font:inherit;font-size:12px;color:#53635b;cursor:pointer}.result{background:#fff;border:1px solid #e0e6e5;border-radius:14px;padding:17px 18px}.resulttop{display:flex;align-items:start;justify-content:space-between;gap:12px}.rank{width:25px;height:25px;border-radius:8px;display:grid;place-items:center;background:#edf4e7;color:#4b7543;font-size:12px;font-weight:800;flex:none}.papername{display:flex;gap:10px;align-items:start;min-width:0}.papername h3{margin:0;font-size:15px;line-height:1.35}.meta{font-size:11px;color:#75817e;margin-top:4px}.score{font-size:11px;color:#71807b;white-space:nowrap}.citation{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#47704d;margin:12px 0 0}.result details{margin-top:9px}.result summary{font-size:12px;color:#54645e;cursor:pointer}.result blockquote{margin:8px 0 0;padding:12px 14px;border-left:3px solid #b9ec7b;background:#f7f9f5;color:#3f4946;font-size:13px;line-height:1.65}.empty,.loading{border:1px dashed #cbd6d1;background:#fff;border-radius:13px;padding:20px;color:#687670;font-size:13px}
.toolbar{display:flex;gap:10px;margin:14px 0}.toolbar input,.toolbar select{border:1px solid #dce3e3;background:#fff;border-radius:9px;padding:10px 12px;font:inherit;font-size:13px;color:#26332e}.toolbar input{flex:1}.papers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px}.paper{background:#fff;border:1px solid #e0e6e5;border-radius:13px;padding:15px}.paper h3{font-size:14px;line-height:1.4;margin:0 0 7px}.paper p{font-size:11px;line-height:1.5;color:#78827f;margin:0}.tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:11px}.tag{background:#eef3ee;border-radius:999px;color:#536d56;padding:4px 8px;font-size:10px}.tag.pending{background:#fff3dd;color:#936c2b}.paper a{color:#3e7149;font-size:11px;display:inline-block;margin-top:11px;text-decoration:none}
.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.metriccard{background:#fff;border:1px solid #e0e6e5;border-radius:14px;padding:18px}.metriccard h3{margin:0 0 5px;font-size:15px}.metriccard p{margin:0;color:#76817e;font-size:12px}.bigmetric{font-size:32px;font-weight:800;letter-spacing:-.05em;margin:16px 0 8px}.bar{height:7px;border-radius:99px;background:#edf1ee;overflow:hidden}.bar i{display:block;height:100%;background:#6da66b;border-radius:99px}.metricrow{display:flex;justify-content:space-between;font-size:12px;color:#52615a;margin-top:11px}.notice{background:#fff8e9;border:1px solid #f0dfb7;border-radius:11px;padding:12px 14px;margin:15px 0;color:#7e6736;font-size:12px;line-height:1.55}.footer{border-top:1px solid #e0e6e5;margin-top:34px;padding-top:14px;color:#8a9691;font-size:11px;display:flex;justify-content:space-between;gap:14px}
@media(max-width:850px){.shell{grid-template-columns:1fr}aside{padding:12px 15px;flex-direction:row;align-items:center;justify-content:space-between;gap:10px}.brand{padding:0}.nav{display:flex}.nav button{padding:8px}.side-note{display:none}main{padding-top:20px}.statgrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:580px){.top{align-items:start;flex-direction:column}.searchrow{flex-wrap:wrap}.searchrow input{flex-basis:100%}.searchrow button{width:100%}.hint{display:none}.papers,.metrics{grid-template-columns:1fr}.footer{flex-direction:column}.hero h2{font-size:31px}}
</style>
</head>
<body><div class="shell">
<aside><div class="brand"><div class="mark">IR</div><div><strong>Indic Research Desk</strong><small>Hindi NLP paper explorer</small></div></div>
<nav class="nav"><button class="active" data-view="explore">⌕ &nbsp; Explore papers</button><button data-view="library">▤ &nbsp; Paper library</button><button data-view="evaluation">◷ &nbsp; Evaluation</button></nav>
<div class="side-note"><strong>Evidence-first research</strong><br>Search results show the exact indexed passage, page and paper. The local demo does not call AWS or a paid model.</div></aside>
<main><header class="top"><div><div class="eyebrow">Research project · Hindi & Indic NLP</div><h1 id="pageTitle">Evidence search</h1></div><span class="pill"><i class="dot"></i> Local demo ready</span></header>
<section class="view active" id="explore"><div class="hero"><div class="eyebrow">A curated research collection</div><h2>Ask a question.<br>Follow the evidence.</h2><p>Explore Hindi coreference, Indic language resources, benchmarks and methods. Every result opens the exact passage retrieved from a paper.</p>
<div class="searchbox"><form id="searchForm"><div class="searchrow"><input id="query" placeholder="e.g. What data is used to evaluate Hindi coreference?" autocomplete="off" required><button id="searchButton">Search evidence&nbsp; →</button><button id="answerButton" type="button" class="answer-button" title="Calls Amazon Bedrock and may incur charges">Generate cited answer · AWS may charge</button></div><div class="controls"><label for="mode">Retrieval</label><select id="mode"><option value="bm25">BM25 · fast lexical</option><option value="hybrid">Hybrid · dense + lexical</option></select><label for="k">Passages</label><select id="k"><option>5</option><option>3</option><option>8</option></select><span class="hint">Evidence search is free and local</span></div></form></div>
<div class="examples"><button data-query="How does DeepHCoref build its Hindi coreference system?">Hindi coreference systems</button><button data-query="Which languages are included in the MCECR event-coreference dataset?">Multilingual coreference</button><button data-query="Which five Hindi datasets are introduced for LLM evaluation?">Hindi benchmarks</button></div></div>
<div class="statgrid"><div class="stat"><span>Curated records</span><strong id="catalogCount">—</strong><em>source-linked papers</em></div><div class="stat"><span>PDFs available locally</span><strong id="pdfCount">—</strong><em>rights-aware downloads</em></div><div class="stat"><span>Evaluation questions</span><strong id="evalCount">—</strong><em>source-first evidence</em></div><div class="stat"><span>Coreference focus</span><strong>Hindi</strong><em>plus Indic NLP</em></div></div>
<div id="answerPanel" class="answer-panel" hidden></div><div class="sectionhead"><div><h2 id="resultHeading">Evidence passages</h2><p id="resultSub">Search the local index to inspect relevant paper passages.</p></div><button class="textlink" data-goto="library">Browse collection →</button></div><div id="results" class="results"><div class="empty">Enter a research question to retrieve passages from the indexed papers.</div></div></section>
<section class="view" id="library"><div class="sectionhead"><div><h2>Paper library</h2><p>Bibliographic records, source pages and local PDF availability.</p></div></div><div class="toolbar"><input id="paperFilter" placeholder="Filter by title, author, or topic"><select id="statusFilter"><option value="all">All records</option><option value="downloaded">PDF available</option><option value="pending">Metadata / access pending</option></select></div><div id="paperGrid" class="papers"><div class="loading">Loading the catalog…</div></div></section>
<section class="view" id="evaluation"><div class="sectionhead"><div><h2>Retrieval evaluation</h2><p>Last saved paired comparison on the active source-first evaluation set.</p></div></div><div id="metricsNotice" class="notice">Metrics are descriptive, not a claim of broad Hindi QA quality. The current set is small and needs more human-reviewed natural questions.</div><div id="metricGrid" class="metrics"><div class="loading">Loading evaluation results…</div></div></section>
<footer class="footer"><span>Local evidence search · no AWS calls for this demo</span><span>Generated answers are a separate, optional Bedrock feature and may incur charges.</span></footer>
</main></div>
<script>
const byId=id=>document.getElementById(id);let catalog=[];
function textNode(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;n.textContent=text;return n}
function showView(name){document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id===name));document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===name));byId('pageTitle').textContent={explore:'Evidence search',library:'Paper library',evaluation:'Retrieval evaluation'}[name]||'Evidence search'}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>showView(b.dataset.view));document.querySelectorAll('[data-goto]').forEach(b=>b.onclick=()=>showView(b.dataset.goto));
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw Error('Request failed ('+r.status+')');return r.json()}
async function loadOverview(){try{const d=await getJSON('/api/overview');byId('catalogCount').textContent=d.catalog_papers;byId('pdfCount').textContent=d.local_pdfs;byId('evalCount').textContent=d.evaluation_questions;renderMetrics(d.metrics)}catch(e){byId('catalogCount').textContent='—';byId('pdfCount').textContent='—'}}
function renderMetrics(m){const root=byId('metricGrid');root.replaceChildren();if(!m||!m.metrics_are_current){root.append(textNode('div','empty','The saved retrieval snapshot belongs to an earlier 29-paper index and is marked historical. Rebuild the 45-paper showcase index and rerun evaluation before presenting current scores.'));return}
for(const [key,label] of [['bm25','BM25 · lexical retrieval'],['hybrid','Hybrid · dense + lexical']]){const s=m.systems?.[key];if(!s)continue;const card=textNode('article','metriccard','');card.append(textNode('h3','',label),textNode('p','',m.questions+' source-first questions · '+m.corpus_papers+' local PDFs'));const big=textNode('div','bigmetric',Number(s.recall_at_10).toFixed(3));card.append(big,textNode('div','','Recall@10'));const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=Math.max(0,Math.min(100,100*s.recall_at_10))+'%';bar.append(fill);card.append(bar);for(const [k,v] of [['MRR@10',s.mrr_at_10],['nDCG@10',s.ndcg_at_10],['Median query latency',s.latency_ms+' ms']]){const row=document.createElement('div');row.className='metricrow';row.append(textNode('span','',k),textNode('strong','',typeof v==='number'?v.toFixed(3):String(v)));card.append(row)}root.append(card)}
}
async function runSearch(e){e.preventDefault();const button=byId('searchButton'),root=byId('results');button.disabled=true;root.replaceChildren(textNode('div','loading','Searching the local index…'));try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:byId('query').value,mode:byId('mode').value,k:Number(byId('k').value)})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Search failed');byId('resultSub').textContent=d.results.length+' passages · '+d.mode.toUpperCase()+' retrieval · evidence copied from indexed chunks';root.replaceChildren();if(!d.results.length){root.append(textNode('div','empty','No matching passages found. Try a paper title, model name, or dataset.'));return}d.results.forEach((s,i)=>{const card=document.createElement('article');card.className='result';const top=document.createElement('div');top.className='resulttop';const name=document.createElement('div');name.className='papername';name.append(textNode('span','rank',String(i+1)));const titlebox=document.createElement('div');titlebox.append(textNode('h3','',s.title));titlebox.append(textNode('div','meta',(s.authors||'Research paper')+' · '+(s.year||'year unavailable')+(s.section?' · '+s.section:'')));name.append(titlebox);top.append(name,textNode('span','score','score '+s.score.toFixed(3)));card.append(top,textNode('div','citation',s.citation));const details=document.createElement('details');const summary=textNode('summary','', 'Inspect retrieved evidence passage');const quote=textNode('blockquote','',s.evidence);details.append(summary,quote);card.append(details);root.append(card)})}catch(err){root.replaceChildren(textNode('div','empty','Search error: '+err.message))}finally{button.disabled=false}}
async function generateAnswer(){const button=byId('answerButton'),root=byId('answerPanel');button.disabled=true;root.hidden=false;root.replaceChildren(textNode('div','loading','Generating a cited answer with Amazon Bedrock… this may incur AWS charges.'));try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:byId('query').value,k:Number(byId('k').value)})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Answer generation failed');root.replaceChildren();const card=document.createElement('article');card.className='result';card.append(textNode('h3','','Grounded answer'),textNode('p','answer-text',d.answer));for(const source of d.sources){const item=document.createElement('details');item.className='answer-source';item.append(textNode('summary','',source.citation+(source.section?' · '+source.section:'')),textNode('blockquote','',source.evidence));card.append(item)}root.append(card)}catch(err){root.replaceChildren(textNode('div','empty','Answer error: '+err.message))}finally{button.disabled=false}}
byId('searchForm').onsubmit=runSearch;byId('answerButton').onclick=generateAnswer;document.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>{byId('query').value=b.dataset.query;byId('searchForm').requestSubmit()});
function renderPapers(){const q=byId('paperFilter').value.trim().toLowerCase(),f=byId('statusFilter').value;const root=byId('paperGrid');root.replaceChildren();const visible=catalog.filter(p=>{const match=(p.title+' '+p.authors+' '+p.topic).toLowerCase().includes(q);return match&&(f==='all'||(f==='downloaded'?p.downloaded:!p.downloaded))});if(!visible.length){root.append(textNode('div','empty','No papers match this filter.'));return}visible.forEach(p=>{const card=document.createElement('article');card.className='paper';card.append(textNode('h3','',p.title),textNode('p','',p.authors+' · '+p.year),textNode('p','',p.topic));const tags=document.createElement('div');tags.className='tags';tags.append(textNode('span',p.downloaded?'tag':'tag pending',p.downloaded?'PDF available':'PDF not local'));tags.append(textNode('span',p.status==='verified downloadable'?'tag':'tag pending',p.status));card.append(tags);const a=document.createElement('a');a.href=p.source_url;a.target='_blank';a.rel='noreferrer';a.textContent='Open bibliographic source ↗';card.append(a);root.append(card)})}
byId('paperFilter').oninput=renderPapers;byId('statusFilter').onchange=renderPapers;loadOverview();getJSON('/api/papers').then(d=>{catalog=d;renderPapers()}).catch(e=>{byId('paperGrid').replaceChildren(textNode('div','empty',e.message))});
</script></body></html>"""
