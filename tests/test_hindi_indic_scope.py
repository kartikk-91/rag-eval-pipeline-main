import csv
import json
from pathlib import Path

from evaluation.eval_harness import EvalExample, _passage_in_chunk, evaluate, load_eval_set
from rag.generate import citation_validation_error, citations_are_valid, normalize_citation_markers
from rag.ingest import _sanitize_extracted_text
from rag.retriever import tokenize
from rag.types import Chunk
from scripts.fetch_hindi_indic import MANIFEST, fetchable


def test_bm25_tokenizer_keeps_devanagari_words():
    assert tokenize("हिंदी coreference डेटा") == ["हिंदी", "coreference", "डेटा"]


def test_passage_matching_tolerates_pdf_text_spans_without_spaces():
    assert _passage_in_chunk("22 of these languages", "22oftheselanguages")


def test_pdf_extraction_replaces_unpaired_surrogates_but_keeps_hindi():
    cleaned = _sanitize_extracted_text("Hindi हिंदी \ud835 text")
    assert cleaned == "Hindi हिंदी ? text"
    assert cleaned.encode("utf-8")


def test_claim_sentences_need_in_range_citations():
    assert citations_are_valid("Hindi data is used [S1]. It has labels [S1][S2].", 2)
    assert not citations_are_valid("Hindi data is used.", 2)
    assert not citations_are_valid("Hindi data is used [S3].", 2)
    assert citations_are_valid("I don't know.", 2)
    assert not citations_are_valid("I don't know. This unrelated claim is true [S1].", 2)


def test_citation_validator_handles_numbered_answers_and_common_abbreviations():
    answer = "1. Dr. Rao's method uses Hindi mentions [S1].\n2. It reports an F1 score [S2]."
    assert citations_are_valid(answer, 2)


def test_grouped_citations_are_normalized_for_clickable_source_links():
    assert normalize_citation_markers("The results are comparable [S1, S2].") == (
        "The results are comparable [S1][S2]."
    )
    assert citations_are_valid("The results are comparable (S1, S2).", 2)


def test_trailing_citation_after_sentence_punctuation_moves_inline():
    answer = "The method uses mentions. [S1] It reports an F1 score. [S2]"
    assert normalize_citation_markers(answer) == (
        "The method uses mentions [S1]. It reports an F1 score [S2]."
    )
    assert citations_are_valid(answer, 2)


def test_validator_reports_why_a_model_draft_was_withheld():
    assert citation_validation_error("This is a factual claim.", 2) == "missing_citation"
    assert citation_validation_error("A claim cites an unavailable source [S3].", 2) == (
        "invalid_source_label"
    )
    assert citation_validation_error("First claim [S1]. Second claim is uncited.", 2) == (
        "uncited_sentence"
    )


def test_active_questions_are_source_first_and_labeled():
    examples = load_eval_set(Path(__file__).parents[1] / "evaluation" / "eval_set.jsonl")
    assert len(examples) == 45
    eval_path = Path(__file__).parents[1] / "evaluation" / "eval_set.jsonl"
    raw = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    assert all(e.gold_passages for e in examples)
    assert all("source_passage_checked_in_ingested_pdf" in r["question_review_status"] for r in raw)
    assert all("evidence_selected_before_retrieval" in r["question_review_status"] for r in raw)
    assert not any("relevant_chunk_ids" in r for r in raw)
    assert len({r["gold_evidence"][0]["local_pdf"] for r in raw}) == 45


def test_hindi_indic_catalog_has_verified_scope_and_conservative_downloads():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 59
    assert {row["status"] for row in rows} <= {
        "verified downloadable",
        "metadata only",
        "access/license unclear",
    }
    eligible = [row for row in rows if row["status"] == "verified downloadable"]
    assert len(eligible) == 45
    assert all(fetchable(row) for row in eligible)
    assert not any(
        row["status"] != "verified downloadable" and fetchable(row) for row in rows
    )
    titles = {row["title"] for row in rows}
    assert any(title.startswith("DeepHCoref") for title in titles)
    assert any(title.startswith("MCECR") for title in titles)
    assert any(title.startswith("Bridge the GAP") for title in titles)
    assert not any("African Languages" in title for title in titles)


class _PassageRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, k):
        return [c.chunk_id for c in self.retrieve_chunks(query, k)]

    def retrieve_chunks(self, query, k):
        return self.chunks[:k]


def test_passage_eval_checks_retrieved_text_not_source_identity():
    chunks = [
        Chunk("wrong", "d", "same-paper.pdf", "A neighboring passage on Hindi NER.", 0.9),
        Chunk("gold", "d", "same-paper.pdf", "The scheme labels coreference relations by type.", 0.8),
    ]
    example = EvalExample("How are relations typed?", gold_passages=("coreference relations by type",))
    report = evaluate(_PassageRetriever(chunks), [example], k_values=(1, 2))
    assert report.hit_rate_at_k[1] == 0.0
    assert report.hit_rate_at_k[2] == 1.0
    assert report.mrr == 0.5
