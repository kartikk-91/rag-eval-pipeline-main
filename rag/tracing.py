"""Langfuse traces for the request path: the log line, made inspectable.

The log line (rag/telemetry.py, docs/decisions/013) says what happened to a request.
A trace shows what the model was given: the question, the passages retrieval
returned, the prompt, the answer, tokens and cost, on one timeline per request and
aggregated across requests. The log line deliberately carries chunk IDs, not text;
the trace is where the text goes.

The two are joined by key, not by lookup table. The trace ID is derived from the
request ID, so the X-Request-Id a visitor quotes finds the log line, Lambda's REPORT
line and the trace.

Off unless configured. With no keys, langfuse is never imported, every call here is
a no-op, and the eval harness, batch scripts and tests behave exactly as before.
Tracing must never be able to fail a request: a client that cannot be built is
reported once and then left off.
"""

import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from rag.config import AWS_REGION, LANGFUSE_SSM_PREFIX, TRACE_FLUSH_TIMEOUT
from rag.telemetry import logger


class _Off:
    """Stands in for an observation when tracing is off. Accepts and drops updates."""

    def update(self, **_fields) -> None:
        pass


OFF = _Off()
_root: ContextVar = ContextVar("rag_trace_root", default=OFF)


def _keys_from_ssm(prefix: str) -> tuple[str, str]:
    """Fetch the keys from Parameter Store, bounded.

    The first /ask on a container that was not warmed pays for this, on the
    gateway's 30s clock. boto3's defaults are 60s connect plus 60s read, with
    retries — a hanging Parameter Store would take the answer down with the trace.
    Bounded here, it can cost at most a few seconds and then turns tracing off.
    """
    import boto3
    from botocore.config import Config

    bounded = Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 2})
    names = [f"{prefix}/public_key", f"{prefix}/secret_key"]
    found = boto3.client("ssm", region_name=AWS_REGION, config=bounded).get_parameters(
        Names=names, WithDecryption=True
    )
    values = {p["Name"]: p["Value"] for p in found["Parameters"]}
    missing = [n for n in names if n not in values]
    if missing:
        raise LookupError(f"SSM parameters not found: {missing}")
    return values[names[0]], values[names[1]]


@lru_cache(maxsize=1)
def client():
    """The Langfuse client, or None when tracing is off. Built once per process."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    try:
        if not (public and secret) and LANGFUSE_SSM_PREFIX:
            public, secret = _keys_from_ssm(LANGFUSE_SSM_PREFIX)
        if not (public and secret):
            return None
        from langfuse import Langfuse

        return Langfuse(public_key=public, secret_key=secret)
    except Exception as exc:  # noqa: BLE001 — any failure here means "no tracing", never "no answer"
        logger.warning(
            "tracing_disabled", extra={"fields": {"reason": f"{type(exc).__name__}: {exc}"}}
        )
        return None


@contextmanager
def request(request_id: str, **metadata):
    """Open the root observation for one request, with a trace ID derived from its ID."""
    lf = client()
    if lf is None:
        yield OFF
        return
    trace_id = lf.create_trace_id(seed=request_id)
    with lf.start_as_current_observation(
        name="request",
        trace_context={"trace_id": trace_id},
        metadata={"request_id": request_id, **metadata},
    ) as root:
        token = _root.set(root)
        try:
            yield root
        except Exception as exc:
            root.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _root.reset(token)


def annotate(**fields) -> None:
    """Update the request's root observation, from anywhere inside the request."""
    _root.get().update(**fields)


@contextmanager
def observation(name: str, as_type: str = "span", **fields):
    """A child observation. Marked ERROR, with the reason, when its body raises.

    A failed generation is the trace most worth opening, so it must not look like a
    successful one that happens to have no output.
    """
    lf = client()
    if lf is None:
        yield OFF
        return
    with lf.start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        try:
            yield obs
        except Exception as exc:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise


def flush() -> dict | None:
    """Send what is queued, waiting at most TRACE_FLUSH_TIMEOUT.

    Returns how long it waited and whether the flush finished inside that, because a
    trace that was given up on and a trace that was sent look identical otherwise.

    Lambda freezes the process the moment the handler returns, and the SDK exports
    from a background thread, so a trace not sent by then is late at best and lost at
    worst. But a flush waits on the network, and the visitor is still waiting on the
    flush. Bounded in a thread rather than trusting the SDK's own timeout: if
    Langfuse is slow, this request loses its trace, not its answer.
    """
    lf = client()
    if lf is None:
        return None
    began = time.perf_counter()
    worker = threading.Thread(target=lf.flush, daemon=True)
    worker.start()
    worker.join(TRACE_FLUSH_TIMEOUT)
    return {
        "flush_ms": round((time.perf_counter() - began) * 1000, 1),
        "flushed": not worker.is_alive(),
    }
