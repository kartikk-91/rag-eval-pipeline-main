> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 012 — The judge detects falsehood, not unsupportedness

*R3, Step 2 — 2026-10-17*

## Observation

The project sells grounded answers with citations and measured retrieval exhaustively while
measuring groundedness not at all. An answer could cite the right paper and misstate it, and
nothing in the repository would notice.

## Method: human labels first, judge blind

Thirty questions drawn from the frozen eval set, answered by the real pipeline — same retriever
the harness scores, same Bedrock call the API makes — so the labels describe what ships.

A human labelled all thirty **before** the judge ran, and the judge never saw those labels. The
order matters: a human who has seen the machine's verdict is no longer an independent measurement,
which is the same circularity 008 was written about.

Each answer also records whether retrieval surfaced a chunk the eval set calls relevant. That
separates two failures a single "wrong answer" would blur: the model had nothing to be faithful
to, versus the model misstating what it was given.

## The instrument changed the measurement before it measured anything

The first labelling document truncated each passage to 1,100 characters. The real passages are
~1,950. The human was judging against **56% of the evidence the model received**.

Two of the first five labels flipped once the full text was shown — a 40% change on the
calibration batch. Had this not been caught, the reported agreement would have been built on
labels made against half the evidence and would have looked entirely respectable.

Calibrating on a handful of items before labelling the rest is what surfaced it.

## Result: the judge is worse than answering "grounded" every time

Both scored against the **adjudicated** labels, so the columns are comparable. Nova Lite ran
before adjudication and its verdicts were re-scored afterwards rather than re-run; a table
comparing two systems against two different ground truths is not a comparison.

| | Nova Lite | Nova Pro |
|---|---|---|
| agreement with human | 80% | **83%** |
| always-grounded baseline | 90% | **90%** |
| Cohen's kappa | −0.11 | **−0.09** |
| real unfaithful answers caught | 0/3 | **0/3** |
| planted fabrications caught | 4/5 | **5/5** |

*Provenance, noted at v1.0:* only the Nova Pro verdicts on the 30 real answers are committed
(`evaluation/faithfulness/judge_labels.jsonl`), and they reproduce that column's agreement, kappa
and 0/3. The Nova Lite column and both planted-control counts were printed by
`evaluation/judge.py` at run time and not saved; re-running it needs Bedrock and will not be
byte-identical.

Both models score **below a constant classifier**. Kappa below zero means agreement worse than
chance would give on these margins. Raw agreement alone would have hidden this: 83% sounds
respectable until you notice that replying "grounded" to everything scores 90%.

The stronger model was tested to separate "the prompt is wrong" from "the task needs more
capability". It is genuinely more capable — it caught every planted fabrication where the smaller
model missed one — and it caught **exactly the same zero** real cases. Capability is not the
bottleneck.

## Diagnosis: it is answering a different question

The planted controls were invented specifics: an F1 of 94.2, eight A100s, a fabricated citation.
Those are **false**, and the judge caught them.

The real unfaithful answers are different. Every one is a plausible, probably-true statement that
simply is not in the passages:

- *"The mGAP dataset is a multilingual extension of the GAP dataset…"*
- *"Because it is challenging to come up with generalized rules for language transformation."*
- *"Answer: LEA"*

The judge knows these are reasonable, so it calls them grounded. It is detecting **falsehood**,
not **unsupportedness** — despite a prompt that says so explicitly and warns that shared
vocabulary is not support.

That is a property of the setup, not of this prompt. A judge sharing the generator's world
knowledge cannot reliably separate "this is in the passage" from "this is correct", because the
second is the easier question and it already knows the answer.

## Adjudication, and why it is recorded

Reviewing disagreements found two human labels contradicted by the passages: item 26's answer
about polysemous word senses, and item 29's about upsampling, are both stated almost verbatim in
the text shown. They were corrected to grounded, with the reason recorded in
`human_labels.jsonl`.

The judge was right on those two. Reporting "0 of 5" would have overstated the failure using
labels there was reason to doubt. 0 of 3 is the honest number and still decisive.

## Decision: do not ship the judge; report the human-labelled rate

No automated faithfulness gate. An unvalidated judge is another model's opinion, and this one is
validated — as unusable.

What the labels do support, and what the README now reports:

**3 of 30 answers were unfaithful — 10%, 95% CI [0.00, 0.23].** Measured by a human against the
exact passages the model saw.

And the finding worth more than the judge would have been:

| | answers | unfaithful |
|---|---|---|
| retrieval found a gold chunk | 21 | **0** |
| retrieval found nothing | 9 | **3 (33%)** |

**The generator never fabricated when given correct evidence.** Every unfaithful answer followed a
retrieval failure. On this corpus, faithfulness is downstream of retrieval — which says the lever
is retrieval quality, not prompt engineering, and makes the ~6% of chunks dense retrieval cannot
reach (008) a groundedness problem as well as a coverage one.

## What would change this

- **A judge that cannot cheat.** Something that checks entailment against the passage rather than
  asking a knowledgeable model for an opinion — an NLI model scoring each claim against each
  passage would not have the generator's world knowledge to fall back on.
- **More labels.** Three unfaithful answers out of thirty gives an interval from 0% to 23%. That
  width is honest and nearly useless for tracking a regression.
- **Harder questions.** Every unfaithful answer here followed a retrieval miss. A set deliberately
  built from questions the corpus cannot answer would test fabrication directly instead of waiting
  for retrieval to fail by accident.

