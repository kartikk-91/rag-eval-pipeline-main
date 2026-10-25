"""Chunk IDs are the contract between the corpus and the hand-labelled eval set.

If these properties break, every gold label silently points at the wrong text.
"""

from rag.ingest import make_chunk_id


def test_chunk_id_is_deterministic():
    assert make_chunk_id("paper.pdf", 3) == make_chunk_id("paper.pdf", 3)


def test_chunk_id_depends_on_both_source_and_index():
    assert make_chunk_id("a.pdf", 0) != make_chunk_id("b.pdf", 0)
    assert make_chunk_id("a.pdf", 0) != make_chunk_id("a.pdf", 1)


def test_chunk_ids_are_scoped_per_file():
    """Adding a document must not disturb another document's IDs.

    This is what makes corpus expansion safe: R1 Step 3 adds ~42 papers without
    invalidating the existing labels. It holds because the ID hashes only
    "<filename>:<index>" — nothing global, no corpus-wide ordering.
    """
    before = [make_chunk_id("existing.pdf", i) for i in range(5)]
    _new_doc = [make_chunk_id("newly_added.pdf", i) for i in range(5)]
    after = [make_chunk_id("existing.pdf", i) for i in range(5)]
    assert before == after
