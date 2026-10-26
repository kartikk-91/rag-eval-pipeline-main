"""Score human claim-to-citation review against the passages actually retrieved.

This deliberately does not infer entailment from shared words or source titles.
Each annotation must include the exact passage text returned for its citation label,
and a reviewer must label each answer claim as supports, partial, or unsupported.
"""

import argparse
import json
from pathlib import Path

LABELS = ("supports", "partial", "unsupported", "pending")


def validate_record(row: dict, line_number: int) -> list[str]:
    errors = []
    answer_id = row.get("id", f"line {line_number}")
    passages = row.get("retrieved_passages")
    if not isinstance(passages, list) or not passages:
        return [f"{answer_id}: retrieved_passages must contain the actual retrieved text"]

    by_label = {}
    for passage in passages:
        if not isinstance(passage, dict):
            errors.append(f"{answer_id}: retrieved passages must be objects")
            continue
        label = passage.get("label")
        text = passage.get("text")
        if not label or not isinstance(text, str) or not text.strip():
            errors.append(f"{answer_id}: every retrieved passage needs a label and exact text")
            continue
        if label in by_label:
            errors.append(f"{answer_id}: duplicate retrieved passage label {label}")
        by_label[label] = passage

    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append(f"{answer_id}: claims must contain one record per factual answer claim")
        return errors
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            errors.append(f"{answer_id} claim {index}: claim record must be an object")
            continue
        status = claim.get("review")
        citations = claim.get("citations")
        if not isinstance(status, str) or status not in LABELS:
            errors.append(f"{answer_id} claim {index}: review must be one of {sorted(LABELS)}")
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append(f"{answer_id} claim {index}: claim text is required")
        if not isinstance(citations, list) or not citations:
            errors.append(f"{answer_id} claim {index}: at least one citation label is required")
            continue
        for label in citations:
            passage = by_label.get(label)
            if passage is None:
                errors.append(f"{answer_id} claim {index}: {label} is not a retrieved passage")
            elif not passage["text"].strip():
                errors.append(f"{answer_id} claim {index}: {label} has no passage text to review")
    return errors


def score_records(rows: list[dict]) -> dict:
    claims = [claim for row in rows for claim in row["claims"]]
    reviewed = [claim for claim in claims if claim["review"] != "pending"]
    counts = {label: sum(c["review"] == label for c in claims) for label in LABELS}
    return {
        "answers": len(rows),
        "claims": len(claims),
        "reviewed_claims": len(reviewed),
        "pending_claims": counts["pending"],
        "supports": counts["supports"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "fully_supported_rate": (
            round(counts["supports"] / len(reviewed), 4) if reviewed else None
        ),
        "review_basis": "human judgment of claim text against exact retrieved passage text",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_file", type=Path, help="JSONL claim-to-passage review annotations")
    args = parser.parse_args()
    rows = []
    errors = []
    for line_number, line in enumerate(args.review_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc})")
            continue
        errors.extend(validate_record(row, line_number))
        rows.append(row)
    if errors:
        print("Citation review validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 2
    if not rows:
        print("No citation review records found.")
        return 2
    print(json.dumps(score_records(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
