"""Shared retrieval types.

`Chunk` is the single currency between retrieval and everything downstream
(generation, the agent, the MCP tools). Keeping it here rather than in the
retriever avoids import cycles and stops the Qdrant payload shape leaking into
callers — before this, four modules unpacked `hit.payload[...]` themselves.

`page` is exact — derived from character offsets into the per-page text, so it
is as reliable as the PDF extraction itself. `section` is a documented heuristic
over headings in extracted text and is advisory only: present it as context,
never as an authoritative citation. See docs/decisions/004.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrieved chunk, ranked."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None

    def cite(self) -> str:
        """Human-readable provenance, e.g. 'hu2022_lora.pdf p.3'.

        Whitespace is collapsed because six of the original corpus filenames
        contain literal newlines, which would otherwise split a citation across
        lines in an API response or on the page.
        """
        source = re.sub(r"\s+", " ", self.source).strip()
        return f"{source} p.{self.page}" if self.page else source
