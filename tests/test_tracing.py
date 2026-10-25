"""Langfuse traces, checked against the spans the real SDK produces.

A real Langfuse client with an in-memory exporter: no network, no account, but the
attributes asserted on are the ones Langfuse would receive. Stubbing the client
instead would test that this code calls methods, not that a trace comes out right.

No AWS, no Qdrant, no model.
"""

import json
import logging
import subprocess
import sys
import time
from functools import lru_cache

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag import telemetry, tracing
from rag.types import Chunk

_OK = {
    "output": {"message": {"content": [{"text": "Kuhn-Munkres."}]}},
    "usage": {"inputTokens": 2500, "outputTokens": 20},
    "metrics": {"latencyMs": 340},
    "ResponseMetadata": {"RetryAttempts": 0},
}
_PASSAGE = "Kuhn-Munkres finds the optimal alignment."


class _Bedrock:
    def __init__(self, raises=None):
        self._raises = raises

    def converse(self, **_kwargs):
        if self._raises is not None:
            raise self._raises
        return _OK


class _Retriever:
    def __init__(self, raises=None):
        self._raises = raises

    def retrieve_chunks(self, query, k):
        if self._raises is not None:
            raise self._raises
        return [Chunk("c1", "doc", "paper.pdf", _PASSAGE, 0.91, page=2)]


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(telemetry.JsonFormatter())
        self.lines: list[dict] = []

    def emit(self, record):
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def traced(monkeypatch):
    """Tracing on, exporting to memory; returns (install, spans-by-name, log lines)."""
    from langfuse import Langfuse

    import api.main
    import rag.generate as gen

    api.main._hits.clear()
    exporter = InMemorySpanExporter()
    # A key per test: the SDK keeps one client per public key, and a client shared
    # across tests would export into whichever exporter it was built with first.
    lf = Langfuse(
        public_key=f"pk-lf-{time.perf_counter_ns()}",
        secret_key="sk-lf-test",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    monkeypatch.setattr(tracing, "client", lambda: lf)
    log = _Lines()
    telemetry.logger.addHandler(log)

    def install(bedrock=None, retriever=None):
        stub = retriever or _Retriever()
        monkeypatch.setattr(gen, "_retriever", lru_cache(maxsize=1)(lambda: stub))
        monkeypatch.setattr(gen, "_bedrock", lambda: bedrock or _Bedrock())

    def spans() -> dict:
        lf.flush()
        return {s.name: s for s in exporter.get_finished_spans()}

    yield install, spans, log
    telemetry.logger.removeHandler(log)


def _ask():
    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    return client.post("/ask", json={"question": "How are mentions aligned?"})


def _attr(span, name):
    value = span.attributes.get(f"langfuse.observation.{name}")
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def test_the_trace_id_is_derived_from_the_request_id(traced):
    """The X-Request-Id a visitor quotes must find the trace, not just the log line."""
    from langfuse import Langfuse

    install, spans, _ = traced
    install()
    r = _ask()
    expected = Langfuse.create_trace_id(seed=r.headers["X-Request-Id"])

    found = spans()
    assert set(found) == {"request", "retrieve", "generate"}
    assert {format(s.context.trace_id, "032x") for s in found.values()} == {expected}


def test_the_trace_shows_what_the_model_was_given_and_what_it_said(traced):
    install, spans, _ = traced
    install()
    _ask()
    found = spans()

    root, retrieve, generate = found["request"], found["retrieve"], found["generate"]
    assert _attr(root, "input") == "How are mentions aligned?"
    assert _attr(root, "output") == "Kuhn-Munkres."
    assert _attr(retrieve, "type") == "retriever"
    # The text the log line deliberately omits is exactly what the trace is for.
    assert _attr(retrieve, "output")[0]["text"] == _PASSAGE
    assert _attr(generate, "type") == "generation"
    assert _PASSAGE in json.dumps(_attr(generate, "input"))
    for child in (retrieve, generate):
        assert child.parent.span_id == root.context.span_id


def test_the_trace_and_the_log_line_agree_on_cost(traced):
    """Cost is computed once and handed to both, so they cannot disagree."""
    install, spans, log = traced
    install()
    _ask()
    generate = spans()["generate"]
    (line,) = [entry for entry in log.lines if entry["msg"] == "request"]

    assert _attr(generate, "usage_details") == {"input": 2500, "output": 20}
    assert _attr(generate, "cost_details")["total"] == line["estimated_cost_usd"]
    # 2500 x 0.078/1M + 20 x 0.312/1M, at the eu-central-1 list price
    assert line["estimated_cost_usd"] == pytest.approx(0.00020124)


def test_a_throttled_generation_is_an_error_span_under_a_warning_trace(traced):
    install, spans, _ = traced
    throttle = ClientError({"Error": {"Code": "ThrottlingException", "Message": "x"}}, "Converse")
    install(bedrock=_Bedrock(raises=throttle))
    assert _ask().status_code == 503

    found = spans()
    assert _attr(found["generate"], "level") == "ERROR"
    assert "ThrottlingException" in _attr(found["generate"], "status_message")
    # Matching the log line: a 503 is the service working as designed.
    assert _attr(found["request"], "level") == "WARNING"


def test_an_uncaught_retrieval_failure_marks_the_trace_as_an_error(traced):
    install, spans, _ = traced
    install(retriever=_Retriever(raises=RuntimeError("index missing")))
    assert _ask().status_code == 500

    found = spans()
    assert _attr(found["request"], "level") == "ERROR"
    assert "index missing" in _attr(found["retrieve"], "status_message")


def test_lambda_flushes_before_freezing_and_logs_whether_it_finished(traced):
    from api.lambda_handler import handler

    install, _, log = traced
    install()

    class _Context:
        aws_request_id = "lambda-5678"

    event = {
        "version": "2.0", "routeKey": "$default", "rawPath": "/health", "rawQueryString": "",
        "headers": {"host": "example.com"},
        "requestContext": {"http": {"method": "GET", "path": "/health", "protocol": "HTTP/1.1",
                                    "sourceIp": "203.0.113.7", "userAgent": "t"},
                           "requestId": "gw", "stage": "$default"},
        "isBase64Encoded": False,
    }
    handler(event, _Context())
    (flush,) = [entry for entry in log.lines if entry["msg"] == "trace_flush"]
    assert flush["request_id"] == "lambda-5678" and flush["flushed"] is True


def test_a_slow_langfuse_costs_the_trace_not_the_answer(monkeypatch):
    """The flush gives up at its cap and says so, instead of holding the visitor."""

    class _Stuck:
        def flush(self):
            time.sleep(5)

    monkeypatch.setattr(tracing, "client", lambda: _Stuck())
    monkeypatch.setattr(tracing, "TRACE_FLUSH_TIMEOUT", 0.2)
    sent = tracing.flush()
    assert sent["flushed"] is False
    assert sent["flush_ms"] < 1000


def test_keys_that_cannot_be_fetched_turn_tracing_off_not_the_service(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tracing, "LANGFUSE_SSM_PREFIX", "/rag-eval-pipeline/langfuse")

    def _denied(_prefix):
        raise PermissionError("AccessDeniedException")

    monkeypatch.setattr(tracing, "_keys_from_ssm", _denied)
    log = _Lines()
    telemetry.logger.addHandler(log)
    tracing.client.cache_clear()
    try:
        assert tracing.client() is None
    finally:
        tracing.client.cache_clear()
        telemetry.logger.removeHandler(log)
    assert log.lines[0]["msg"] == "tracing_disabled"
    assert "AccessDeniedException" in log.lines[0]["reason"]


def test_without_keys_langfuse_is_never_imported():
    """Off means off: no import cost, no background thread, for every non-API caller."""
    code = (
        "import sys\n"
        "from rag import tracing\n"
        "assert tracing.client() is None\n"
        "with tracing.request('r') as root, tracing.observation('x') as obs:\n"
        "    root.update(output=1); obs.update(output=2)\n"
        "assert tracing.flush() is None\n"
        "print('langfuse' in sys.modules)\n"
    )
    env = {"PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("model", "tokens", "expected"),
    [
        ("eu.amazon.nova-lite-v1:0", (1000, 1000), 0.00039),
        ("eu.amazon.nova-lite-v1:0", (None, 10), None),
        ("some.unpriced-model", (1000, 1000), None),
    ],
)
def test_an_unknown_price_is_none_not_zero(monkeypatch, model, tokens, expected):
    import rag.generate as gen

    monkeypatch.setattr(gen, "BEDROCK_MODEL_ID", model)
    assert gen.estimated_cost_usd(*tokens) == expected


def test_only_questions_are_traced_not_page_loads(traced):
    """The demo page and health checks would otherwise bury every question."""
    from api.main import app

    install, spans, _ = traced
    install()
    client = TestClient(app)
    client.get("/")
    client.get("/health")
    assert spans() == {}


def test_the_warm_ping_builds_the_tracing_client_off_the_visitors_clock(monkeypatch):
    import api.lambda_handler as lh
    import rag.generate as gen

    built = []
    monkeypatch.setattr(gen, "retrieve_chunks", lambda q, k: [])
    monkeypatch.setattr(tracing, "client", lambda: built.append(1))
    assert lh.handler({"warm": True}, None) == {"warmed": True}
    assert built == [1]


def test_the_parameter_store_call_is_bounded(monkeypatch):
    """Unbounded, boto3 would wait 60s + 60s with retries inside a 30s gateway."""
    import boto3

    seen = {}

    class _SSM:
        def get_parameters(self, Names, WithDecryption):
            return {"Parameters": [{"Name": n, "Value": "v"} for n in Names]}

    def _client(service, region_name, config):
        seen["config"] = config
        return _SSM()

    monkeypatch.setattr(boto3, "client", _client)
    assert tracing._keys_from_ssm("/p") == ("v", "v")
    cfg = seen["config"]
    worst = (cfg.connect_timeout + cfg.read_timeout) * cfg.retries["max_attempts"]
    assert worst <= 10
