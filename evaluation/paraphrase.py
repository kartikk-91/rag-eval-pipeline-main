"""Rewrite the eval questions in words their gold chunks do not use.

Decision 008 writes each question while reading its gold chunk. That keeps labels
independent of the retriever, which was the point — but a question written next to
its answer borrows the answer's vocabulary. Measured: 54% of a question's
informative terms (in <5% of chunks) appear in its gold chunk, against 21% for the
other chunks of the same paper.

A lexical retriever is rewarded for exactly that. Real visitors have not read the
passage. So the same 42 questions are rewritten as such a visitor would ask them,
and every retriever is scored again against the SAME gold labels. If a retriever's
advantage is an artifact of borrowed wording, it shrinks here.

The rules were fixed before any rewritten question was scored:
  - keep proper names (models, datasets, metrics): a real user types "SpanBERT"
  - change every other content word where a natural alternative exists
  - keep the meaning exactly, so the gold labels still apply

Each rewrite is then read by a person, and any whose meaning drifted is marked
"drifted" and excluded from scoring — not repaired, which would let the reviewer
steer the wording toward whichever retriever they expected to win.

    python -m evaluation.paraphrase        # writes evaluation/robustness/paraphrased.jsonl
"""

import json
import time
from pathlib import Path

from evaluation.eval_harness import DEFAULT_EVAL_SET, load_eval_set
from rag.generate import _bedrock

OUT = Path(__file__).resolve().parent / "robustness" / "paraphrased.jsonl"
# A different, stronger model than the generator, so the rewrite is not shaped by
# the same model whose answers are being retrieved for.
MODEL = "eu.amazon.nova-pro-v1:0"

PROMPT = (
    "Rewrite the question the way a researcher in natural language processing would "
    "ask it if they had NOT read the paper it comes from.\n"
    "- Keep every proper name exactly: model, dataset, metric and system names.\n"
    "- Change every other content word or phrase where a natural alternative exists.\n"
    "- Keep the meaning exactly. Do not add, remove or narrow what is being asked.\n"
    "Reply with the rewritten question only."
)


def main() -> int:
    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ex in eval_set:
        response = _bedrock().converse(
            modelId=MODEL,
            system=[{"text": PROMPT}],
            messages=[{"role": "user", "content": [{"text": ex.query}]}],
            inferenceConfig={"maxTokens": 120, "temperature": 0.0},
        )
        rewritten = response["output"]["message"]["content"][0]["text"].strip()
        rows.append(
            {
                "original": ex.query,
                "query": rewritten,
                "relevant_chunk_ids": sorted(ex.relevant_chunk_ids),
                "review": None,  # filled by hand: "ok" or "drifted"
            }
        )
        print(f"  {ex.query[:60]}\n  -> {rewritten[:60]}\n")
        time.sleep(1.5)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} to {OUT.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
