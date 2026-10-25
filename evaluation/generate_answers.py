"""Produce answers for human faithfulness labelling.

Runs the real pipeline — the same retriever the eval harness scores and the same
Bedrock call the API makes — so the labels describe what actually ships, not a
reconstruction of it.

Records, per question, the answer and the exact chunks the model was given. Also
records whether retrieval found a gold chunk, because that separates two failures
a single "wrong answer" verdict would blur:

  - retrieval missed, so the model had nothing to be faithful to
  - retrieval succeeded and the model misstated what it was given

Only the second is a generation problem, and only the second is what a faithfulness
metric should be measuring.

    python -m evaluation.generate_answers --n 30 > /dev/null   # writes the JSONL
"""

import argparse
import json
import random
import time
from pathlib import Path

from evaluation.eval_harness import DEFAULT_EVAL_SET, load_eval_set
from rag.errors import PermanentGenerationError, TransientGenerationError
from rag.generate import answer_question

OUT = Path(__file__).resolve().parent / "faithfulness" / "answers.jsonl"
K = 3  # what /ask uses by default, so the labels describe the deployed behaviour

# Thirty calls in a row throttles this account, which is how the first run of this
# script died. A batch job is not a request path: nothing is waiting on a 30s
# gateway, so it can afford to pace itself and wait properly when told to slow down.
PACE_SECONDS = 1.5
BACKOFF_SECONDS = (5, 15, 45)


def _answer_with_patience(question: str) -> tuple[str, list]:
    """Call the pipeline, waiting out throttling rather than dying on it.

    Uses the same error types the API uses, from the other side: transient means
    wait and try again, permanent means stop now and say why. That split exists
    precisely so each caller can choose its own response to it.
    """
    for wait in BACKOFF_SECONDS:
        try:
            return answer_question(question, K)
        except TransientGenerationError as exc:
            print(f"      throttled ({exc}); waiting {wait}s")
            time.sleep(wait)
        except PermanentGenerationError:
            raise  # retrying cannot help; let it surface immediately
    return answer_question(question, K)  # last attempt, let it raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args()

    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    rng = random.Random(args.seed)
    chosen = rng.sample(list(eval_set), min(args.n, len(eval_set)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(chosen, start=1):
        answer, chunks = _answer_with_patience(ex.query)
        retrieved = {c.chunk_id for c in chunks}
        rows.append(
            {
                "id": i,
                "question": ex.query,
                "answer": answer,
                "chunk_ids": [c.chunk_id for c in chunks],
                "citations": [c.cite() for c in chunks],
                "contexts": [c.text for c in chunks],
                # Did the retriever surface a chunk the eval set calls relevant?
                "gold_retrieved": bool(retrieved & set(ex.relevant_chunk_ids)),
            }
        )
        print(f"  {i:>2}/{len(chosen)}  {ex.query[:60]}")

        # Written as we go: a failure at question 29 should not discard 28 answers
        # that each cost a model call.
        with OUT.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(PACE_SECONDS)

    with_gold = sum(r["gold_retrieved"] for r in rows)
    print(f"\nwrote {len(rows)} answers to {OUT.relative_to(Path.cwd())}")
    print(f"{with_gold}/{len(rows)} had a gold chunk in the {K} passages shown to the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
