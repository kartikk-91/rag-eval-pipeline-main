"""Measure how much the active query wording overlaps its source-first evidence.

This is a wording diagnostic, not a retrieval score or a pass/fail quality metric.
It helps identify a benchmark where questions may borrow too many terms from the
passages they were authored from.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.eval_harness import DEFAULT_EVAL_SET, _passage_in_chunk
from rag.retriever import DenseRetriever, tokenize

OUT = Path(__file__).resolve().parent / "query_overlap_diagnostics.json"


def _overlap(informative: set[str], terms: set[str]) -> float:
    return len(informative & terms) / len(informative) if informative else 0.0


def main() -> None:
    store = DenseRetriever(load_model=False)
    try:
        chunks = store.all_chunks()
    finally:
        store._client.close()

    document_frequency = Counter(
        term for chunk in chunks for term in set(tokenize(chunk.text))
    )
    max_document_frequency = max(5, int(0.01 * len(chunks)))
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[Path(chunk.source).name.casefold()].append(chunk)

    rows = []
    with DEFAULT_EVAL_SET.open(encoding="utf-8") as stream:
        examples = [json.loads(line) for line in stream if line.strip()]
    for example in examples:
        query_terms = set(tokenize(example["query"]))
        informative = {
            term
            for term in query_terms
            if document_frequency[term] <= max_document_frequency
        }
        evidence = example["gold_evidence"]
        passage_terms = set()
        matched_gold = []
        for item in evidence:
            passage_terms.update(tokenize(item["passage_text"]))
            source = item["local_pdf"].casefold()
            matched_gold.extend(
                chunk
                for chunk in by_source.get(source, [])
                if _passage_in_chunk(item["passage_text"], chunk.text)
            )
        gold_ids = {chunk.chunk_id for chunk in matched_gold}
        same_source = by_source.get(evidence[0]["local_pdf"].casefold(), [])

        rows.append(
            {
                "id": example["id"],
                "informative_query_terms": sorted(informative),
                "gold_overlap": _overlap(informative, passage_terms),
                "best_same_paper_non_gold_overlap": max(
                    (
                        _overlap(informative, set(tokenize(chunk.text)))
                        for chunk in same_source
                        if chunk.chunk_id not in gold_ids
                    ),
                    default=0.0,
                ),
                "gold_passage_found_in_index": bool(gold_ids),
            }
        )

    n = len(rows)
    report = {
        "schema_version": 1,
        "questions": n,
        "indexed_chunks": len(chunks),
        "informative_term_document_frequency_max": max_document_frequency,
        "mean_informative_query_term_overlap_with_gold": round(
            sum(row["gold_overlap"] for row in rows) / n, 4
        ),
        "mean_best_same_paper_non_gold_overlap": round(
            sum(row["best_same_paper_non_gold_overlap"] for row in rows) / n, 4
        ),
        "all_gold_passages_found": all(row["gold_passage_found_in_index"] for row in rows),
        "interpretation": (
            "Wording diagnostic only. High overlap can favor lexical retrieval; it does not "
            "prove leakage or invalidate source-first evidence labels."
        ),
        "per_question": rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Questions: {n}; indexed chunks: {len(chunks)}")
    print(
        "Mean informative query-term overlap with gold: "
        f"{report['mean_informative_query_term_overlap_with_gold']:.3f}"
    )
    print(f"Mean best same-paper non-gold overlap: {report['mean_best_same_paper_non_gold_overlap']:.3f}")
    print(f"All gold passages found: {report['all_gold_passages_found']}")
    print(f"Wrote {OUT.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
