"""FastAPI service exposing the RAG pipeline."""

import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from rag import telemetry, tracing
from rag.errors import PermanentGenerationError, TransientGenerationError
from rag.generate import answer_question

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask the papers</title>
<style>
  :root { color-scheme: light dark; --fg:#111; --muted:#666; --line:#ddd; --bg:#fff; --accent:#0b5; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --muted:#999; --line:#333; --bg:#111; --accent:#3d8; }
  }
  body { font: 16px/1.6 ui-sans-serif, system-ui, sans-serif; color: var(--fg); background: var(--bg);
         max-width: 46rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  p.sub { color: var(--muted); margin: 0 0 1.5rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .7rem .8rem; font: inherit; border: 1px solid var(--line);
          border-radius: 6px; background: transparent; color: inherit; }
  button { padding: .7rem 1.1rem; font: inherit; border: 0; border-radius: 6px;
           background: var(--accent); color: #fff; cursor: pointer; }
  button[disabled] { opacity: .5; cursor: progress; }
  .examples { margin: .75rem 0 0; font-size: .9rem; color: var(--muted); }
  .examples button { background: none; color: var(--accent); border: 0; padding: 0 .4rem 0 0;
                     font-size: inherit; text-decoration: underline; cursor: pointer; }
  #answer { margin-top: 2rem; white-space: pre-wrap; }
  ol { padding-left: 1.2rem; }
  li { margin: .35rem 0; font-size: .92rem; }
  .cite { font-variant-numeric: tabular-nums; }
  .sec { color: var(--muted); }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
           font-size: .85rem; color: var(--muted); }
  a { color: var(--accent); }
</style>
</head>
<body>
<h1>Ask the papers</h1>
<p class="sub">Grounded question answering over a curated Hindi coreference and Indic NLP paper collection.
Each answer citation links to the retrieved passage used as evidence.</p>

<form id="f">
  <input id="q" name="q" placeholder="How is coreference resolution evaluated?" autocomplete="off" required>
  <button id="go" type="submit">Ask</button>
</form>
<p class="examples">
  Try:
  <button type="button" onclick="fill(this)">How does DeepHCoref build its Hindi coreference system?</button>
  <button type="button" onclick="fill(this)">Which languages does MCECR cover?</button>
  <button type="button" onclick="fill(this)">Which datasets are used to evaluate Hindi LLMs?</button>
</p>

<div id="answer"></div>

<footer>
  Retrieval quality is measured, not assumed &mdash; recall@k, hit-rate@k and MRR on a hand-labelled set.
  <a href="https://github.com/zenith19/rag-eval-pipeline">Source and results on GitHub</a>.
</footer>

<script>
const f = document.getElementById('f'), q = document.getElementById('q'),
      go = document.getElementById('go'), out = document.getElementById('answer');

function fill(b) { q.value = b.textContent; q.focus(); }

f.onsubmit = async (e) => {
  e.preventDefault();
  go.disabled = true;
  out.textContent = 'Thinking\\u2026 (a cold start takes a few seconds)';
  try {
    // A schedule keeps one container warm, but a second simultaneous visitor gets
    // a cold one, and a cold /ask loads the embedding model for ~47s against the
    // gateway's 30s ceiling. That surfaces as 503/504 before the app can answer,
    // so retry rather than show a raw gateway error: by the second attempt the
    // container that timed out is warm and answers in well under a second.
    const ask = () => fetch('ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q.value, k: 4 })
    });
    let r = await ask();
    for (let attempt = 1; (r.status === 503 || r.status === 504) && attempt <= 2; attempt++) {
      out.textContent = 'Waking the model up\\u2026 (first question after a quiet spell)';
      await new Promise(res => setTimeout(res, 3000));
      r = await ask();
    }
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      out.textContent = e.detail || ('Request failed: ' + r.status);
      return;
    }
    const d = await r.json();
    out.innerHTML = '';
    const p = document.createElement('p');
    p.textContent = d.answer;
    out.appendChild(p);
    const h = document.createElement('p');
    h.innerHTML = '<strong>Sources</strong>';
    out.appendChild(h);
    const ol = document.createElement('ol');
    for (const s of d.sources) {
      const li = document.createElement('li');
      const c = document.createElement('span');
      c.className = 'cite';
      c.textContent = s.citation;
      li.appendChild(c);
      if (s.section) {
        const sec = document.createElement('span');
        sec.className = 'sec';
        sec.textContent = ' \\u2014 ' + s.section;
        li.appendChild(sec);
      }
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Evidence passage';
      const evidence = document.createElement('blockquote');
      evidence.textContent = s.evidence;
      details.append(summary, evidence);
      li.appendChild(details);
      ol.appendChild(li);
    }
    out.appendChild(ol);
  } catch (err) {
    out.textContent = 'Could not reach the service: ' + err.message;
  } finally {
    go.disabled = false;
  }
};
</script>
</body>
</html>
"""
