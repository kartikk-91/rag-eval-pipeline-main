> **Historical evaluation note:** Results and corpus descriptions in this decision record apply only to the legacy corpus and labels. They are not results for the Hindi/Indic collection. The active evaluation set is documented in `evaluation/eval_set.jsonl`.

# 002 — Rebuilding the corpus from the thesis bibliography

*R1, Step 3 — 2026-09-30*

## Observation

The retrieval evaluation reported recall@1 = 0.35 over **9 papers**. The README already
attributed the weak top-1 ranking to "several closely related papers whose chunks compete for the
top slot" — but with 9 documents that competition is mild, so the measurement was being taken on a
corpus far easier than the one the tool was built for. The thesis bibliography lists 58 references;
the tool knew 7 of them.

## Hypothesis

A corpus that matches the actual literature review will (a) make the tool useful for its original
purpose and (b) make the ranking problem *harder and more honest*, since dozens of coreference
papers describe near-identical methods. The measured baseline should be expected to get worse.

## Decision

Fetch the freely available references and commit the recipe, not the PDFs. `data/` is gitignored —
redistributing papers isn't ours to do — so `scripts/corpus_manifest.csv` is the artifact: 39
verified rows that reproduce the corpus from a clone.

Existing filenames are untouched. `make_chunk_id()` hashes `filename:index`, so renaming the
original 9 would invalidate all 10 gold labels; adding files cannot (see
`tests/test_chunk_ids.py::test_chunk_ids_are_scoped_per_file`).

## Trade-off

Resolution went through four sources before two worked. Semantic Scholar returns 429 without an API
key, OpenAlex throttles anonymous search, DBLP sits behind an anti-bot challenge, and arXiv's own
search returns nothing for multi-word titles. What worked: OpenAlex (with a mailto) and
Crossref→ACL Anthology, plus URLs printed in the thesis bibliography itself.

Three bugs in the resolver are worth recording, because each would have silently corrupted the
corpus rather than failing loudly:

1. **Substring title matching** resolved *The Winograd Schema Challenge* to a different arXiv paper
   that merely contained the phrase. Matching is now exact on a normalised title.
2. **ACL Anthology paths are case-sensitive** (`P16-1009.pdf` 200, `p16-1009.pdf` 404) while
   Crossref reports DOIs lowercased. One `.lower()` call had broken every pre-2020 ACL paper.
3. **Ligatures.** Normalising by stripping non-alphanumerics deletes the `ﬁ` in `signiﬁcance`, so
   two correctly-resolved papers failed their own title check. Fixed with NFKD decomposition.
   The same ligatures are present in ingested chunk text — harmless for dense retrieval, but
   relevant if lexical matching is ever added (R4).

Because identifiers for older papers had to be guessed, verification is by **content, not status
code**: each PDF is downloaded, page 1 extracted, and the paper's own title must appear there. Six
of seven guessed identifiers verified; the seventh (`ding2023_peft` → arXiv 2203.06904) was
rejected, which is the check doing its job.

## Measured validation

39/42 fetchable references verified and fetched; corpus 9 → 48 papers. Three deliberate gaps,
recorded rather than approximated:

| Ref | Paper | Reason |
|---|---|---|
| 30 | Bagga & Baldwin 1998 (B³) | `L98-1063` 404s; LREC 1998 predates the Anthology PDF archive |
| 23 | Levesque et al. 2012 (Winograd Schema) | AAAI/KR proceedings, no open PDF |
| 12 | Ding et al. 2023 (PEFT survey) | *Nature MI* paywalled; the arXiv version carries a different title |

Six further references are paywalled and listed as `MANUAL` in `scripts/corpus_sources.py`.

## Next

Provenance (`document_id`, `page`, `section`) and a single re-index, then the re-baselined
retrieval numbers over 48 papers — published even if, as expected, they are worse than 0.35.

