import pytest

from evaluation.citation_support import score_records, validate_record


def _record():
    return {
        "id": "answer-1",
        "answer": "Hindi data uses a biaffine classifier [S1].",
        "retrieved_passages": [
            {
                "label": "S1",
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": "paper-0",
                "text": "The Hindi system uses a biaffine classifier.",
            }
        ],
        "claims": [
            {
                "text": "Hindi data uses a biaffine classifier.",
                "citations": ["S1"],
                "review": "supports",
            }
        ],
    }


def test_citation_review_requires_actual_retrieved_passage_text():
    row = _record()
    row["retrieved_passages"][0].pop("text")

    assert any("exact text" in error for error in validate_record(row, 1))


def test_citation_review_rejects_source_labels_that_were_not_retrieved():
    row = _record()
    row["claims"][0]["citations"] = ["S2"]

    assert any("not a retrieved passage" in error for error in validate_record(row, 1))


def test_citation_support_scores_only_completed_claim_reviews():
    row = _record()
    row["claims"].append(
        {"text": "An unreviewed claim.", "citations": ["S1"], "review": "pending"}
    )

    assert validate_record(row, 1) == []
    score = score_records([row])
    assert score["reviewed_claims"] == 1
    assert score["pending_claims"] == 1
    assert score["fully_supported_rate"] == 1.0


@pytest.mark.parametrize("status", ["supports", "partial", "unsupported", "pending"])
def test_citation_review_accepts_only_explicit_review_statuses(status):
    row = _record()
    row["claims"][0]["review"] = status

    assert validate_record(row, 1) == []
