"""Legacy Bengali-corpus baseline comparison (disabled for the active corpus).

Exits non-zero if any metric falls more than `tolerance` below baseline. CI runs it
on every PR against an index rebuilt from evaluation/index_snapshot.npz
(docs/decisions/016). Improvements never fail.

    python -m evaluation.compare_baseline            # compare
    python -m evaluation.compare_baseline --update   # accept current numbers as the new baseline
"""

import argparse
import json
from datetime import date
from pathlib import Path

from evaluation.eval_harness import DEFAULT_EVAL_SET, evaluate, load_eval_set
from rag.retriever import DenseRetriever

BASELINE = Path(__file__).resolve().parent / "baseline.json"
K_VALUES = (1, 3, 5, 10)


def _current() -> dict:
    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    report = evaluate(DenseRetriever(), eval_set, k_values=K_VALUES)
    return {
        "mrr": round(report.mrr, 3),
        "recall_at_k": {str(k): round(report.recall_at_k[k], 3) for k in K_VALUES},
        "hit_rate_at_k": {str(k): round(report.hit_rate_at_k[k], 3) for k in K_VALUES},
        "ndcg_at_k": {str(k): round(report.ndcg_at_k[k], 3) for k in K_VALUES},
        "_n": report.n_queries,
    }


def main() -> int:
    print(
        "Refusing to compare against the legacy Bengali-corpus baseline. "
        "Run `python -m evaluation.run_eval` after the Hindi/Indic PDFs are indexed."
    )
    return 2

    # Legacy implementation retained below for provenance; it is intentionally
    # unreachable so old scores cannot be updated with the new corpus by mistake.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="write current numbers as the baseline")
    args = ap.parse_args()

    base = json.loads(BASELINE.read_text())
    now = _current()
    tol = base.get("tolerance", 0.03)

    rows, regressions = [], []
    for name, key in (("MRR", "mrr"), *((f"recall@{k}", ("recall_at_k", str(k))) for k in K_VALUES),
                      *((f"hit@{k}", ("hit_rate_at_k", str(k))) for k in K_VALUES),
                      *((f"nDCG@{k}", ("ndcg_at_k", str(k))) for k in K_VALUES)):
        if isinstance(key, tuple):
            if key[0] not in base["metrics"]:
                continue  # metric added after this baseline was recorded
            b, c = base["metrics"][key[0]][key[1]], now[key[0]][key[1]]
        else:
            b, c = base["metrics"][key], now[key]
        delta = c - b
        flag = "REGRESSION" if delta < -tol else ("improved" if delta > tol else "")
        rows.append(f"  {name:<10} {b:>6.3f} -> {c:>6.3f}  {delta:+.3f}  {flag}")
        if delta < -tol:
            regressions.append(name)

    print(f"baseline: {base['corpus']['papers']} papers, {base['eval_set']['questions']} questions "
          f"(recorded {base['recorded']})")
    print(f"current : {now['_n']} questions, tolerance ±{tol}\n")
    print("\n".join(rows))

    if args.update:
        base["metrics"] = {k: v for k, v in now.items() if not k.startswith("_")}
        base["eval_set"]["questions"] = now["_n"]
        # Stamp the date, so a baseline can't claim to be older (and more
        # established) than the run that actually produced it.
        base["recorded"] = date.today().isoformat()
        BASELINE.write_text(json.dumps(base, indent=2) + "\n")
        print("\nbaseline updated — record WHY in the commit message")
        return 0

    if regressions:
        print(f"\nREGRESSION in {len(regressions)} metric(s): {', '.join(regressions)}")
        return 1
    print("\nno regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
