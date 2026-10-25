"""The request log line: what a failed production request leaves behind.

The acceptance test for this milestone is that a failure can be diagnosed from what
was recorded, without reproducing it. So each failure mode from
test_failure_modes.py is replayed here and the assertion is about the log line, not
the status code: does it name the cause, the time spent, and the retries used?

No AWS, no Qdrant, no model.
"""

import json
import logging
import subprocess
import sys
from functools import lru_cache

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi.testclient import TestClient

from rag import telemetry
from rag.types import Chunk

_OK = {
    "output": {"message": {"content": [{"text": "Kuhn-Munkres."}]}},
    "usage": {"inputTokens": 812, "outputTokens": 23},
    "metrics": {"latencyMs": 431},
    "ResponseMetadata": {"RetryAttempts": 0},
}


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        document_id="doc",
        source="paper.pdf",
        text="Kuhn-Munkres finds the optimal alignment.",
        score=0.91234,
        page=2,
    )


class _Bedrock:
    def __init__(self, raises=None, response=None):
        self._raises, self._response = raises, response

    def converse(self, **_kwargs):
        if self._raises is not None:
            raise self._raises
        return self._response


class _Retriever:
    def __init__(self, raises=None):
        self._raises = raises

    def retrieve_chunks(self, query, k):
        if self._raises is not None:
            raise self._raises
        return [_chunk()]


class _Lines(logging.Handler):
    """Collects what the real formatter produces, parsed back."""

    def __init__(self):
        super().__init__()
        self.setFormatter(telemetry.JsonFormatter())
        self.raw: list[str] = []

    def emit(self, record):
        self.raw.append(self.format(record))

    @property
    def lines(self) -> list[dict]:
        return [json.loads(r) for r in self.raw]


@pytest.fixture
def pipeline(monkeypatch):
    """Install stubs for retrieval and Bedrock; return the captured log lines."""
    import api.main
    import rag.generate as gen

    # The rate limiter's window is process memory shared by every test in the run,
    # and the suite sends more than RATE_LIMIT_PER_MIN questions from one client.
    # Without this, the next test added anywhere turns a 200 here into a 429.
    api.main._hits.clear()
    captured = _Lines()
    telemetry.logger.addHandler(captured)

    def _install(bedrock=None, retriever=None):
        # Wrapped in lru_cache so retrieve_chunks() can still ask whether the
        # retriever was already loaded, exactly as it does in production.
        stub = retriever or _Retriever()
        monkeypatch.setattr(gen, "_retriever", lru_cache(maxsize=1)(lambda: stub))
        monkeypatch.setattr(gen, "_bedrock", lambda: bedrock or _Bedrock(response=_OK))
        return captured

    yield _install
    telemetry.logger.removeHandler(captured)


def _ask(client, **headers):
    return client.post("/ask", json={"question": "How are mentions aligned?"}, headers=headers)


def _client():
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_a_successful_request_leaves_one_line_with_every_stage(pipeline):
    log = pipeline()
    r = _ask(_client())
    assert r.status_code == 200

    (line,) = log.lines
    assert line["request_id"] == r.headers["X-Request-Id"]
    assert line["status"] == 200 and line["level"] == "INFO"
    assert line["question"] == "How are mentions aligned?"
    assert line["chunk_ids"] == ["c1"] and line["top_score"] == 0.912
    assert line["input_tokens"] == 812 and line["output_tokens"] == 23
    assert line["bedrock_latency_ms"] == 431 and line["retries"] == 0
    assert line["model_id"]
    for timing in ("retrieve_ms", "generate_ms", "total_ms"):
        assert line[timing] >= 0


def test_cold_start_is_true_only_for_the_request_that_loaded_the_model(pipeline):
    log = pipeline()
    client = _client()
    _ask(client)
    _ask(client)
    assert [line["cold_start"] for line in log.lines] == [True, False]


def test_the_client_address_is_not_logged(pipeline):
    log = pipeline()
    _ask(_client(), **{"x-forwarded-for": "203.0.113.7"})
    assert "203.0.113.7" not in log.raw[0]


def test_a_throttled_request_names_the_error_and_the_retries_spent(pipeline):
    throttle = ClientError(
        {
            "Error": {"Code": "ThrottlingException", "Message": "slow down"},
            "ResponseMetadata": {"RetryAttempts": 1},
        },
        "Converse",
    )
    log = pipeline(bedrock=_Bedrock(raises=throttle))
    assert _ask(_client()).status_code == 503

    (line,) = log.lines
    assert line["bedrock_error"] == "ThrottlingException"
    assert line["retries"] == 1
    assert line["level"] == "WARNING"  # working as designed: seen, not paged
    assert "generate_ms" in line  # timed even though it raised


def test_a_timeout_is_told_apart_from_a_throttle(pipeline):
    log = pipeline(bedrock=_Bedrock(raises=ReadTimeoutError(endpoint_url="https://bedrock")))
    assert _ask(_client()).status_code == 503
    assert log.lines[0]["bedrock_error"] == "ReadTimeoutError"


def test_a_permanent_failure_is_logged_as_an_error(pipeline):
    denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse")
    log = pipeline(bedrock=_Bedrock(raises=denied))
    assert _ask(_client()).status_code == 502
    line = log.lines[0]
    assert line["level"] == "ERROR" and line["bedrock_error"] == "AccessDeniedException"


def test_an_uncaught_retrieval_failure_still_gets_its_line_and_traceback(pipeline):
    log = pipeline(retriever=_Retriever(raises=RuntimeError("index missing")))
    r = _ask(_client())
    assert r.status_code == 500

    (line,) = log.lines
    # The failure that most needs reporting must carry the ID that finds it.
    assert r.headers["X-Request-Id"] == line["request_id"]
    assert line["request_id"] in r.json()["detail"]
    assert line["status"] == 500 and line["level"] == "ERROR"
    assert "index missing" in line["exception"]
    assert "retrieve_ms" in line


def test_the_lambda_request_id_is_used_so_the_line_joins_lambdas_report(pipeline):
    """Through Mangum with an HTTP API v2 event: the path production actually takes."""
    from api.lambda_handler import handler

    log = pipeline()
    event = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/health",
        "rawQueryString": "",
        "headers": {"host": "example.execute-api.eu-central-1.amazonaws.com"},
        "requestContext": {
            "http": {"method": "GET", "path": "/health", "protocol": "HTTP/1.1",
                     "sourceIp": "203.0.113.7", "userAgent": "test"},
            "requestId": "gateway-id",
            "stage": "$default",
        },
        "isBase64Encoded": False,
    }

    class _Context:
        aws_request_id = "lambda-1234"

    response = handler(event, _Context())
    assert response["statusCode"] == 200
    assert response["headers"]["x-request-id"] == "lambda-1234"
    assert log.lines[0]["request_id"] == "lambda-1234"


def test_note_outside_a_request_is_a_no_op():
    """The eval harness and batch scripts call the same code with no request open."""
    from contextvars import copy_context

    copy_context().run(telemetry.note, anything=1)  # must not raise


def test_log_lines_go_to_stderr_never_stdout():
    """The MCP server speaks its protocol over stdout; a log line there corrupts it."""
    code = (
        "from rag import telemetry\n"
        "telemetry.configure_logging()\n"
        "telemetry.logger.info('request', extra={'fields': {'status': 200}})\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout == ""
    assert json.loads(out.stderr)["status"] == 200
