"""One structured log line per request, built up by the stages that served it.

Before this, a failed production request was observable only as a status code: a
503 did not say which Bedrock error caused it, how long the call took, or whether a
retry had been spent. Diagnosing one meant reproducing it locally, which for a
throttle or a timeout is usually impossible.

The shape: the API opens a record when a request arrives, retrieval and generation
add what they know to it with note(), and the API writes it as one JSON line when
the response leaves. One line per request rather than one per stage, so a single
CloudWatch filter answers "what happened to request X" without joining.

The stages never receive the record as an argument. It travels in a ContextVar, so
generate_from_chunks() keeps its return type and every caller outside a request —
the eval harness, the batch scripts, the MCP tools — gets a no-op. Starlette runs
sync endpoints in a worker thread with a copy of the context; the copy points at the
same dict, so what the thread notes is what the middleware writes.

Logs go to stderr, never stdout. The MCP server speaks its protocol over stdout, and
a log line there is a corrupted message. Lambda forwards both to CloudWatch.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar

_record: ContextVar[dict | None] = ContextVar("rag_request_record", default=None)

logger = logging.getLogger("rag.request")


class JsonFormatter(logging.Formatter):
    """A log record as one JSON object, with the request fields at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        # default=str: a stray non-JSON value should cost a little fidelity in the
        # log, not raise inside the logging call and lose the whole line.
        return json.dumps(line, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Attach the JSON handler once. Safe to call from every entry point.

    Own handler, propagate=False: the Lambda runtime installs a handler on the root
    logger whose format is not ours, and inheriting it would print every line twice
    in two shapes. No timestamp field either — CloudWatch stamps each line on arrival.
    """
    if any(getattr(h, "_rag_json", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._rag_json = True
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("RAG_LOG_LEVEL", "INFO"))
    logger.propagate = False


def start(**fields) -> dict:
    """Open the record for the current request and return it."""
    record = dict(fields)
    _record.set(record)
    return record


def note(**fields) -> None:
    """Add fields to the current request's record. A no-op outside a request."""
    record = _record.get()
    if record is not None:
        record.update(fields)


@contextmanager
def stage(name: str):
    """Time a stage into `<name>_ms`, including when it raises.

    A stage that fails is exactly the one whose duration matters: the error code
    says what went wrong, and the time says how much of the gateway's 30s budget
    the failure consumed before anyone was told.
    """
    began = time.perf_counter()
    try:
        yield
    finally:
        note(**{f"{name}_ms": round((time.perf_counter() - began) * 1000, 1)})
