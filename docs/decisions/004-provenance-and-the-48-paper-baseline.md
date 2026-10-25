> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 004 — Page provenance, and what 48 papers did to the numbers

*R1, Step 3 — 2026-10-03*

## Observation

The corpus on disk had been 48 papers since PR #2, but the index still held the original 9. Answers
cited a filename and nothing more, so "where did this come from?" could not be answered below
document level — the original point of the tool.

## Decision: annotate, never re-split

`page`, `section` and `document_id` are attached to chunks by locating each chunk's character
offset in the document text. The text and the split are untouched.

The tempting alternative — chunk per page, which makes the page trivially known — would have moved
every boundary. Chunk IDs hash `"<filename>:<index>"`, so the IDs would have survived while the
*text behind them* changed: all 10 gold labels silently repointed, the eval still running, every
number afterwards quietly meaningless. That is CLAUDE.md invariant 3, and it is the failure this
design exists to avoid.

Verified rather than assumed: all 249 original chunk texts were SHA-fingerprinted before the change
and re-checked after. **249/249 byte-identical.**

## Trade-off: page is exact, section is a heuristic

`page` is derived from per-page character spans — as reliable as the PDF extraction itself.
**1,688/1,688.**

`section` is a regex over extracted text, and three rounds of tightening were needed because the
early versions were confidently wrong:

| Caught | What it actually was |
|---|---|
| `6659 IEEE, 2013.` (×15) | a reference line matching the numbered-heading pattern |
| `1 I NTRODUCTION` | small caps extract as a split initial glyph run |
| `0.63532 The highest result obtained from CNN…` | a results-table row |
| `08 Workshop on Named Entity Recognition…` | another reference |
| `30 Rhea Sukthanker et al` | a numbered bibliography entry |

Now filtered by structure: no leading zeros, first component 1–20 (no paper here has twenty
top-level sections), at most three levels, no digits in the title, and no years, DOIs, page ranges
or "et al". **1,587/1,688**, with eight of the ten most common values being genuine headings.

Residual noise remains (`3.1 M: and while it's there it`), so **section is advisory and page is
authoritative**. `Chunk.cite()` uses only source and page. Making sections trustworthy needs a
layout-aware parser such as GROBID, not a better regex; that is deferred, not solved.

## What only running it caught

The last two defects were invisible to the test suite, because both were about what the output
*looks like* rather than whether the code runs. A single real query surfaced them:

- `Chunk.cite()` emitted citations containing a literal newline — six of the original filenames
  contain one — so a citation broke across two lines in any response. Whitespace is now collapsed
  at display time; the filename itself stays untouched, because renaming it would invalidate the
  gold labels.
- `30 Rhea Sukthanker et al` was reported as a section on page 31 of the anaphora review.

Unit tests asserted `_clean_heading` and `_plausible_section_number` behaved correctly, and they
did. The faults were in the composition. Worth remembering when the demo page lands in Step 4:
looking at real output is a different check from running tests, and it finds different bugs.

## Measured validation — and a prediction that was wrong

I expected recall@1 to fall. It did not.

| k | 9 papers | 48 papers |
|---|---|---|
| recall@1 | 0.35 | **0.35** |
| recall@3 | 0.65 | 0.55 |
| recall@5 | 0.80 | 0.70 |
| recall@10 | 0.95 | **0.75** |
| recall@20 | — | 0.85 |
| recall@50 | — | 0.95 |
| hit-rate@10 | 1.00 | 0.90 |
| hit-rate@50 | — | 1.00 |
| MRR | 0.787 | 0.770 |

Top-1 held exactly. What collapsed was **depth**: gold chunks are not lost, they are pushed down.
The coverage k=10 used to give now takes k=50.

## Consequence for R2 — the reranking hypothesis needs qualifying

R2's plan rests on: *recall@10 = 0.95 and hit-rate@10 = 1.00 say coverage is fine and ranking is the
bottleneck, so test a reranker before hybrid.* On the real corpus that premise no longer holds at
the depth a cross-encoder works over. **At k=20, recall is 0.85** — a reranker fed the top 20
cannot recover 15% of the relevant material however good it is.

The reranker experiment therefore treats candidate-pool size as a measured parameter, not an
assumption, and must report quality against pool size and latency together. Hybrid retrieval is
still not clearly required: hit-rate@50 = 1.00 means dense retrieval does surface every question's
evidence eventually.

**Caveat that limits all of the above:** n = 10. Each question is 10% of hit-rate; the moves here
are one or two questions apiece. Directionally useful, not statistically meaningful. R2 freezes a
~40-question set with bootstrap intervals *before* any comparison, and this is the evidence for why
that ordering matters.

## Next

R1 Step 4: packaging, Docker, Terraform, deploy, CI. Note for sizing: embedding 1,688 chunks takes
roughly 9 minutes on CPU here — a one-time image-build cost, not a request-path cost.

