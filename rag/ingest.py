"""PDF ingestion: load documents, split into overlapping chunks, attach provenance.

Chunk text and boundaries are deliberately unchanged by the provenance work.
Page and section are *annotated onto* the existing split by locating each chunk's
offset in the document, never derived from a new one — because chunk IDs hash
"<filename>:<index>", so moving a boundary silently repoints every gold label at
different text (CLAUDE.md invariant 3).
"""

import hashlib
import os
import re
from collections import Counter
from io import BytesIO
from pathlib import Path

import boto3
from pypdf import PdfReader

from rag.config import AWS_REGION

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data/hindi_indic"))
S3_BUCKET = os.environ.get("RAG_S3_BUCKET")  # if set, read PDFs from S3; otherwise from data/

CHUNK_SIZE = 2000      # characters, ~500 tokens
CHUNK_OVERLAP = 200    # ~10% overlap so context isn't lost at chunk boundaries


def _sanitize_extracted_text(text: str) -> str:
    """Replace malformed surrogate code points emitted by some PDF font maps."""
    return text.encode("utf-8", errors="replace").decode("utf-8")

# A section heading in a paper is either numbered ("3.2 Datasets") or one of the
# conventional names. Deliberately conservative: a chunk with no heading above it
# gets None rather than a guess, since a wrong section label is worse than none.
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,60})\s*$")
# Reference lines masquerade as numbered headings ("6659 IEEE, 2013." was picked up
# 15 times). Two cheap filters: real section numbers are small, and headings don't
# carry years, DOIs or page ranges.
_MAX_SECTION_NUMBER = 20
# "et al" catches numbered bibliography entries such as "30 Rhea Sukthanker et al",
# which otherwise look exactly like a numbered heading.
_CITATION_ISH = re.compile(
    r"(?:19|20)\d{2}|doi|arxiv|\bpp\.|\bvol\.|\bet al\b", re.IGNORECASE
)
_NAMED_HEADING = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experimental setup|experiments?|results?|analysis|discussion|"
    r"conclusions?|limitations|references|appendix|acknowledgments?)\s*$",
    re.IGNORECASE,
)


def _read_pdf(source) -> tuple[str, list[tuple[int, int, int]]]:
    """Return the document text plus (page_number, start, end) offsets into it.

    The text is byte-identical to "\\n".join(page texts) — the spans simply record
    where each page landed, so a chunk can be traced back to a page.
    """
    reader = PdfReader(source)
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Some PDF font maps return unpaired UTF-16 surrogate code points for
        # diagram/math glyphs. They are valid Python strings but rejected by the
        # fast tokenizer used for embeddings. Replace only malformed Unicode;
        # ordinary Devanagari and supplementary-plane characters remain intact.
        text = _sanitize_extracted_text(text)
        if parts:
            cursor += 1  # the separator "\n" that join() will insert
        spans.append((page_number, cursor, cursor + len(text)))
        cursor += len(text)
        parts.append(text)
    return "\n".join(parts), spans


def _load_documents() -> list[tuple[str, str, list[tuple[int, int, int]]]]:
    """Return (filename, text, page_spans) per PDF, from S3 if configured else data/."""
    if S3_BUCKET:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        documents = []
        for obj in s3.list_objects_v2(Bucket=S3_BUCKET).get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                text, spans = _read_pdf(BytesIO(body))
                documents.append((Path(key).name, text, spans))
        return documents
    return [(p.name, *_read_pdf(str(p))) for p in sorted(DATA_DIR.glob("*.pdf"))]


def make_chunk_id(source: str, index: int) -> str:
    # Deterministic so chunk IDs stay stable across runs and eval labels remain valid.
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:12]


def make_document_id(source: str) -> str:
    """Stable per-document handle. Filenames are human-readable but messy — six of
    the original nine contain literal newlines — so grouping and filtering use this."""
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _page_for_offset(spans: list[tuple[int, int, int]], offset: int) -> int | None:
    for page_number, start, end in spans:
        if start <= offset <= end:
            return page_number
    return spans[-1][0] if spans else None


def _plausible_section_number(number: str) -> bool:
    """Reject the table rows and reference fragments that look like headings.

    Caught in practice: "0.63532 The highest result..." (a results-table row),
    "6 Others 26" (a table row), "08 Workshop on Named Entity Recognition..."
    (a reference). Section numbers have no leading zeros, start at 1, stay small,
    and rarely nest more than three deep.
    """
    parts = number.split(".")
    if len(parts) > 3:
        return False
    if any(len(p) > 1 and p.startswith("0") for p in parts):
        return False
    first = int(parts[0])
    return 1 <= first <= _MAX_SECTION_NUMBER


def _clean_heading(text: str) -> str:
    """Repair small-caps extraction artifacts: PDFs give "I NTRODUCTION" for
    small-capped "Introduction", because the large initial is a separate glyph run."""
    text = re.sub(r"\b([A-Z]) ([A-Z]{2,})", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text.title() if text.isupper() else text


def _section_for_offset(text: str, offset: int) -> str | None:
    """Nearest heading at or above `offset`. A heuristic, and labelled as one."""
    head = text[:offset]
    for line in reversed(head.splitlines()):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        numbered = _NUMBERED_HEADING.match(stripped)
        if numbered and not _CITATION_ISH.search(stripped):
            number, title = numbered.group(1), numbered.group(2)
            if _plausible_section_number(number) and not re.search(r"\d", title):
                return f"{number} {_clean_heading(title)}"
        if _NAMED_HEADING.match(stripped):
            return _clean_heading(stripped).title()
    return None


def load_chunks() -> list[dict]:
    # Imported here, not at module scope: langchain_text_splitters costs ~11s to
    # import (it pulls in LangChain core) and is only needed when building the
    # index. Importing it at module level made every consumer pay for it —
    # including the API, which never chunks anything at query time.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[dict] = []
    for source, text, spans in _load_documents():
        document_id = make_document_id(source)
        cursor = 0
        for i, piece in enumerate(splitter.split_text(text)):
            # Locate the piece to recover its page. Chunks overlap, so the search
            # advances by one character rather than by the chunk length.
            offset = text.find(piece, cursor)
            if offset == -1:
                offset = text.find(piece)
            if offset != -1:
                cursor = offset + 1
            chunks.append(
                {
                    "chunk_id": make_chunk_id(source, i),
                    "document_id": document_id,
                    "source": source,
                    "page": _page_for_offset(spans, offset) if offset != -1 else None,
                    "section": _section_for_offset(text, offset) if offset != -1 else None,
                    "text": piece,
                }
            )
    return chunks


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        print(
            f"No PDFs found in '{DATA_DIR}'. Add licensed PDFs to RAG_DATA_DIR "
            "or configure RAG_S3_BUCKET for deployed ingestion."
        )
        return

    for source, count in sorted(Counter(c["source"] for c in chunks).items()):
        print(f"  {source!r}: {count} chunks")
    with_page = sum(1 for c in chunks if c["page"] is not None)
    with_section = sum(1 for c in chunks if c["section"] is not None)
    origin = f"S3 bucket '{S3_BUCKET}'" if S3_BUCKET else f"local folder '{DATA_DIR}/'"
    print(f"\n{len(chunks)} chunks from {len({c['source'] for c in chunks})} document(s) [{origin}].")
    print(f"page known: {with_page}/{len(chunks)}   section known: {with_section}/{len(chunks)}")


if __name__ == "__main__":
    main()
