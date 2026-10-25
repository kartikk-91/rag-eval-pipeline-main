"""Provenance: page is exact, section is a heuristic, neither may move the split.

The expensive property — that annotating provenance left all 249 original chunk
texts byte-identical — was verified against a fingerprint taken before the change
and is recorded in docs/decisions/004. These tests cover the parts that can be
checked quickly, plus a real single-PDF read.
"""

from pathlib import Path

import pytest

from rag.ingest import (
    _clean_heading,
    _page_for_offset,
    _plausible_section_number,
    _read_pdf,
    make_chunk_id,
    make_document_id,
)

CORPUS = Path(__file__).resolve().parent.parent / "data"
SAMPLE = CORPUS / "luo2005_ceaf.pdf"


def test_document_id_is_stable_and_distinct():
    assert make_document_id("a.pdf") == make_document_id("a.pdf")
    assert make_document_id("a.pdf") != make_document_id("b.pdf")


def test_document_id_is_independent_of_chunk_id():
    """Two different handles for two different jobs: one names the file, one names a piece of it."""
    assert make_document_id("a.pdf") != make_chunk_id("a.pdf", 0)


@pytest.mark.parametrize(
    "number,ok",
    [
        ("1", True), ("2.6.2", True), ("12", True),
        ("0.63532", False),   # a results-table row seen in the real corpus
        ("08", False),        # a reference line
        ("31", False),        # beyond any plausible section count here
        ("1.2.3.4", False),   # too deeply nested to be a heading
    ],
)
def test_section_numbers_reject_table_rows_and_references(number, ok):
    assert _plausible_section_number(number) is ok


def test_small_caps_extraction_artifacts_are_repaired():
    # PDFs render small-capped headings as a large initial plus a capital run,
    # so extraction yields "I NTRODUCTION".
    assert _clean_heading("I NTRODUCTION") == "Introduction"
    assert _clean_heading("L OW-R ANK ADAPTATION") == "Low-Rank Adaptation"
    assert _clean_heading("Related Work") == "Related Work"


def test_page_lookup_maps_offsets_to_pages():
    spans = [(1, 0, 100), (2, 101, 250), (3, 251, 400)]
    assert _page_for_offset(spans, 0) == 1
    assert _page_for_offset(spans, 150) == 2
    assert _page_for_offset(spans, 399) == 3
    assert _page_for_offset(spans, 10_000) == 3  # past the end: last page, not a crash


@pytest.mark.skipif(not SAMPLE.exists(), reason="corpus not fetched (scripts/fetch_corpus.py)")
def test_real_pdf_text_is_unchanged_by_page_tracking():
    """The spans are metadata: the text must still be exactly the joined pages."""
    from pypdf import PdfReader

    text, spans = _read_pdf(str(SAMPLE))
    expected = "\n".join(page.extract_text() or "" for page in PdfReader(str(SAMPLE)).pages)
    assert text == expected
    assert [p for p, _, _ in spans] == list(range(1, len(spans) + 1))
    for _page, start, end in spans:
        assert 0 <= start <= end <= len(text)


def test_citation_never_contains_a_newline():
    """Six corpus filenames contain literal newlines; a citation must stay on one line."""
    from rag.types import Chunk

    messy = "BenCoref: A Multi-Domain Dataset of Nominal Phrases and Pronominal\nReference Annotations.pdf"
    cite = Chunk(chunk_id="a", document_id="d", source=messy, text="t", score=0.7, page=3).cite()
    assert "\n" not in cite
    assert cite.endswith(" p.3")


def test_numbered_bibliography_entries_are_not_sections():
    """"30 Rhea Sukthanker et al" is a reference line, not a heading — seen in the real corpus."""
    from rag.ingest import _CITATION_ISH, _plausible_section_number

    assert _CITATION_ISH.search("30 Rhea Sukthanker et al")
    assert not _plausible_section_number("30")


def test_embedded_index_allows_a_second_client_after_close(tmp_path):
    """Embedded Qdrant locks its directory exclusively.

    build_index indexes with one client and then demos with another, which fails
    in embedded mode unless the first is closed. That broke the Docker build, and
    a server-backed test could never have caught it.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    path = str(tmp_path / "idx")
    first = QdrantClient(path=path)
    first.create_collection("t", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    first.close()

    second = QdrantClient(path=path)  # would raise if the lock were still held
    assert second.collection_exists("t")
    second.close()


def test_baked_index_is_copied_somewhere_writable(tmp_path, monkeypatch):
    """A read-only baked index must be copied before embedded Qdrant can lock it.

    Resolved in Python, not by a shell entrypoint: an entrypoint sets the
    environment of one process, so `docker exec`, the MCP server and a Lambda
    handler would all miss it and fall back to a server that is not there.
    """
    import rag.config as config

    baked = tmp_path / "baked"
    baked.mkdir()
    (baked / "meta.json").write_text("{}")
    writable = tmp_path / "writable"

    monkeypatch.setattr(config, "QDRANT_PATH", None)
    monkeypatch.setattr(config, "QDRANT_BAKED_INDEX", str(baked))
    monkeypatch.setattr(config, "QDRANT_WRITABLE_INDEX", str(writable))

    assert config.index_path() == str(writable)
    assert (writable / "meta.json").exists()
    # Second call must not re-copy or fail.
    assert config.index_path() == str(writable)


def test_no_baked_index_means_server_mode(monkeypatch):
    import rag.config as config

    monkeypatch.setattr(config, "QDRANT_PATH", None)
    monkeypatch.setattr(config, "QDRANT_BAKED_INDEX", None)
    assert config.index_path() is None


def test_stale_staging_directory_does_not_break_the_copy(tmp_path, monkeypatch):
    """A crash mid-copy leaves a staging directory behind, and PIDs recycle.

    A reused container keeps the same PID, so without cleanup the FileExistsError
    would repeat on every invocation rather than clearing itself.
    """
    import os

    import rag.config as config

    baked = tmp_path / "baked"
    baked.mkdir()
    (baked / "meta.json").write_text("{}")
    writable = tmp_path / "writable"

    stale = tmp_path / f"writable.{os.getpid()}"
    stale.mkdir()
    (stale / "half-written.json").write_text("truncated")

    monkeypatch.setattr(config, "QDRANT_PATH", None)
    monkeypatch.setattr(config, "QDRANT_BAKED_INDEX", str(baked))
    monkeypatch.setattr(config, "QDRANT_WRITABLE_INDEX", str(writable))

    assert config.index_path() == str(writable)
    assert (writable / "meta.json").exists()
    assert not (writable / "half-written.json").exists()
