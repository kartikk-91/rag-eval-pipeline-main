"""Shared fixtures.

Unit tests run anywhere. Tests that need the Qdrant container are marked
`integration` and skip cleanly when it isn't reachable, so CI stays green
without infrastructure.
"""

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def _qdrant_up() -> bool:
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=2):
            return True
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs the Qdrant container running")


def pytest_collection_modifyitems(config, items):
    if _qdrant_up():
        return
    skip = pytest.mark.skip(reason="Qdrant not reachable on localhost:6333 (docker compose up -d)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def retriever():
    from rag.retriever import HybridRetriever

    return HybridRetriever()
