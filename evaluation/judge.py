"""An LLM judge for faithfulness, and the numbers that say whether to trust it.

The judge answers one question per item: is every claim in the answer supported by
the passages the model was given? It never sees the human labels.

Reporting raw agreement alone would be misleading here. The human labels are
overwhelmingly "grounded", so a judge that answered "grounded" to everything scores
close to the majority share and looks competent while detecting nothing. The exact
baseline is computed and printed beside the agreement rather than written here,
where adjudicating a label would leave it stale. So this reports three things:

  - agreement, against the always-grounded baseline it has to beat
  - Cohen's kappa, which corrects for agreement that chance alone would produce
  - recall on the ungrounded cases, which is the thing a faithfulness judge is for

Plus a planted control: answers deliberately rewritten to claim something the
passages do not say. A judge that passes those is not judging, whatever its
agreement score says.

    python -m evaluation.judge          # judge, score, write verdicts
    python -m evaluation.judge --control-only
"""

import argparse
import json
import time
from pathlib import Path

from rag.config import BEDROCK_MODEL_ID
from rag.errors import TransientGenerationError
from rag.generate import _bedrock, _format_context

HERE = Path(__file__).resolve().parent / "faithfulness"
ANSWERS = HERE / "answers.jsonl"
HUMAN = HERE / "human_labels.jsonl"
VERDICTS = HERE / "judge_labels.jsonl"

PACE_SECONDS = 1.5
BACKOFF_SECONDS = (5, 15, 45)

# Deliberately narrow. The judge is not asked whether the answer is correct, helpful
# or complete — only whether the passages support it. Widening this to "is this a
# good answer" is how a faithfulness metric quietly becomes a quality metric.
JUDGE_PROMPT = (
    "You check whether an answer is supported by the passages it was given.\n"
    "Reply with exactly one word.\n\n"
    "GROUNDED - every claim in the answer appears in the passages, or the answer "
    "declines to answer and the passages genuinely do not contain it.\n"
    "NOT_GROUNDED - the answer states something the passages do not say, or the "
    "answer declines while the passages do contain it.\n\n"
    "A claim that is true in general but absent from the passages is NOT_GROUNDED. "
    "Shared vocabulary is not support: the passages must actually make the claim."
)


class _Ctx:
    """Minimal stand-in so _format_context can be reused on stored text."""

    def __init__(self, source: str, text: str):
        self.source, self.text = source, text


def judge(
    question: str,
    answer: str,
    citations: list[str],
    contexts: list[str],
    model_id: str = BEDROCK_MODEL_ID,
) -> str:
    """Return GROUNDED or NOT_GROUNDED for one answer."""
    chunks = [_Ctx(c, t) for c, t in zip(citations, contexts, strict=True)]
    body = (
        f"Passages:\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    for wait in BACKOFF_SECONDS:
        try:
            response = _bedrock().converse(
                modelId=model_id,
                system=[{"text": JUDGE_PROMPT}],
                messages=[{"role": "user", "content": [{"text": body}]}],
                # Low temperature and a tight token budget: this is a classification,
                # not a discussion, and a verbose judge is a judge that is reasoning
                # its way somewhere other than the question asked.
                inferenceConfig={"maxTokens": 8, "temperature": 0.0},
            )
            text = response["output"]["message"]["content"][0]["text"].strip().upper()
            # "NOT" anywhere wins: the model sometimes pads the verdict.
            return "NOT_GROUNDED" if "NOT" in text else "GROUNDED"
        except TransientGenerationError:
            time.sleep(wait)
    raise TransientGenerationError("judge throttled past its patience")


def _kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa: agreement above what chance would give on these margins.

    0 means the judge is no better than guessing with the same class frequencies;
    1 means perfect. Reported because raw agreement on a heavily one-sided set
    flatters a judge that has learned nothing but the majority answer.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in {"GROUNDED", "NOT_GROUNDED"}
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _planted_controls(rows: list[dict]) -> list[dict]:
    """Answers rewritten to claim something the passages do not contain.

    Every one keeps the original question and passages and changes only the answer,
    inventing a specific, checkable detail — a number, a name, a mechanism. If the
    judge calls these grounded it is not reading the passages, and its agreement on
    the real set means nothing.
    """
    fabrications = [
        "The authors report an F1 score of 94.2 on this task, evaluated on 10,000 "
        "held-out examples.",
        "This was first proposed by Hinton and Salakhutdinov in their 2006 Science "
        "paper on dimensionality reduction.",
        "The method requires exactly 8 NVIDIA A100 GPUs and takes 36 hours to train.",
        "The paper concludes that this approach should be abandoned in favour of "
        "rule-based systems.",
        "Results were validated on the Penn Treebank and achieved state of the art "
        "by a margin of 3.1 points.",
    ]
    return [
        {**r, "answer": f, "id": f"control-{n}"}
        for n, (r, f) in enumerate(zip(rows[: len(fabrications)], fabrications, strict=True), 1)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument(
        "--model",
        default=BEDROCK_MODEL_ID,
        help="judge model; the generator's model is the default so a difference "
        "isolates capability rather than prompt",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in ANSWERS.open()]

    controls = _planted_controls(rows)
    print(f"control: {len(controls)} answers with invented claims\n")
    caught = 0
    for c in controls:
        verdict = judge(c["question"], c["answer"], c["citations"], c["contexts"], args.model)
        caught += verdict == "NOT_GROUNDED"
        print(f"  {c['id']}: {verdict}")
        time.sleep(PACE_SECONDS)
    print(f"\n  caught {caught}/{len(controls)} planted fabrications")
    if caught < len(controls):
        print("  -> a judge that passes invented claims cannot be trusted on real ones")
    if args.control_only:
        return 0

    print(f"\njudging {len(rows)} real answers with {args.model}")
    verdicts = {}
    for r in rows:
        verdicts[r["id"]] = judge(
            r["question"], r["answer"], r["citations"], r["contexts"], args.model
        )
        time.sleep(PACE_SECONDS)
    # The model goes in the file: two models were compared, and a verdict file that
    # does not say which produced it is not evidence of anything.
    VERDICTS.write_text(
        "".join(
            json.dumps({"id": i, "judge": v, "model": args.model}) + "\n"
            for i, v in sorted(verdicts.items())
        )
    )

    human = {h["id"]: h["human"] for h in (json.loads(x) for x in HUMAN.open())}
    expand = {"G": "GROUNDED", "N": "NOT_GROUNDED"}
    h = [expand[human[i]] for i in sorted(human)]
    j = [verdicts[i] for i in sorted(human)]

    n = len(h)
    agree = sum(x == y for x, y in zip(h, j, strict=True))
    majority = max(h.count("GROUNDED"), h.count("NOT_GROUNDED")) / n
    human_n = [i for i in sorted(human) if human[i] == "N"]
    caught_n = sum(verdicts[i] == "NOT_GROUNDED" for i in human_n)

    from evaluation.eval_harness import bootstrap_mean_ci

    lo, hi = bootstrap_mean_ci([float(x == y) for x, y in zip(h, j, strict=True)])

    print(f"\n{'=' * 58}")
    print(f"  agreement          {agree}/{n} = {agree / n:.0%}   95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  always-grounded    {majority:.0%}   <- the score to beat")
    print(f"  Cohen's kappa      {_kappa(h, j):+.2f}")
    print(f"  caught             {caught_n}/{len(human_n)} of the answers you marked ungrounded")
    print(f"  planted controls   {caught}/{len(controls)}")
    print("=" * 58)

    disagreements = [i for i in sorted(human) if expand[human[i]] != verdicts[i]]
    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}):")
        by_id = {r["id"]: r for r in rows}
        for i in disagreements:
            print(f"  [{i}] human {human[i]}, judge {verdicts[i]}: {by_id[i]['question'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
