"""Integrity of legacy Bengali-corpus paraphrases.

The robustness comparison is only paired if each rewrite carries exactly the gold
labels of the question it rewrites, and only honest if a person reviewed every one.
A file that drifted from eval_set.jsonl would compare two different answer keys.
"""

import json
from pathlib import Path

from evaluation.eval_harness import load_eval_set

ROWS = [
    json.loads(line)
    for line in (
        Path(__file__).resolve().parent.parent / "evaluation" / "robustness" / "paraphrased.jsonl"
    ).open()
]


def test_every_rewrite_carries_the_gold_labels_of_its_original():
    legacy_eval = Path(__file__).resolve().parent.parent / "evaluation" / "legacy_bengali_eval_set.jsonl"
    gold = {ex.query: ex.relevant_chunk_ids for ex in load_eval_set(legacy_eval)}
    assert len(ROWS) == len(gold)
    for row in ROWS:
        assert frozenset(row["relevant_chunk_ids"]) == gold[row["original"]]


def test_every_rewrite_was_reviewed_and_every_exclusion_says_why():
    assert {row["review"] for row in ROWS} <= {"ok", "drifted"}
    assert all(row.get("review_note") for row in ROWS if row["review"] == "drifted")
