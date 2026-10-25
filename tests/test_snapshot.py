"""Internal consistency checks for the retained legacy Bengali-corpus snapshot.

The regression gate (evaluation/snapshot.py) rebuilds the index from committed
vectors. If those fell out of step with the eval set or the configuration, the gate
would score something other than production — and could pass while doing it.
"""

import json
from pathlib import Path

import numpy as np

from evaluation.eval_harness import load_eval_set

ROOT = Path(__file__).resolve().parent.parent
SNAP = np.load(ROOT / "evaluation" / "index_snapshot.npz")


def test_every_gold_chunk_is_in_the_snapshot():
    """Legacy labels remain traceable to the preserved legacy snapshot."""
    ids = set(SNAP["chunk_ids"].tolist())
    legacy_eval = ROOT / "evaluation" / "legacy_bengali_eval_set.jsonl"
    gold = {g for ex in load_eval_set(legacy_eval) for g in ex.relevant_chunk_ids}
    assert gold <= ids


def test_the_snapshot_matches_the_legacy_model_and_corpus():
    baseline = json.loads((ROOT / "evaluation" / "baseline.json").read_text())
    assert str(SNAP["model"]) == "BAAI/bge-small-en-v1.5"
    assert SNAP["vectors"].ndim == 2
    assert SNAP["vectors"].shape[0] == baseline["corpus"]["chunks"]
    assert len(set(SNAP["chunk_ids"].tolist())) == len(SNAP["chunk_ids"])


def test_the_snapshot_carries_no_document_text():
    """It goes into a public repository; the papers' content must not."""
    assert set(SNAP.files) == {"chunk_ids", "vectors", "model"}


def test_vectors_are_normalised_as_the_index_expects():
    norms = np.linalg.norm(SNAP["vectors"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
