> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 005 — Packaging, configuration, and the first CI

*R1, Step 4 — 2026-10-04*

## Observation

Four releases in, the project was still a folder of scripts: no `pyproject.toml`, so `pytest` and
the pre-commit hook relied on a `sys.path` insert in `conftest.py`; no linter, so the duplicate dict
key in `scripts/corpus_sources.py` was caught only by a hand-written AST check; and no CI, so the
test suite ran when someone remembered.

`rag/retriever.py` also still imported its configuration from `rag/build_index.py` — the retriever
reading a hostname out of the indexer. Decision 003 called that out and deferred the fix to here.

## Decision

**Packaging.** `pyproject.toml` declares the package, its dependencies, pytest's markers and ruff's
rules in one place. `requirements.txt` is removed rather than left to drift out of step with it;
installation is `pip install -e ".[dev]"`.

**Configuration.** `rag/config.py` holds the collection name, embedding model, vector size and
Qdrant endpoint, each overridable by environment variable so one image can point at a local
container or a deployed endpoint. Importing it costs **0.05s** against `rag.build_index`'s 3.55s,
and the indexer no longer has to be imported to learn a hostname.

**`eval/` → `evaluation/`.** `eval` shadows a builtin and reads badly as a package name. Imports
were moving anyway. Paths inside earlier decision records were updated so they still resolve; the
decisions themselves are untouched.

**Lint.** ruff with `E, F, I, UP, B` at 110 characters. `scripts/corpus_sources.py` is exempt from
line length: paper titles are copied verbatim from the bibliography and reflowing them to satisfy a
limit would damage the data. 42 findings, of which 35 were line length; the rest were fixed.

**CI.** GitHub Actions runs lint and the suite on every pull request. Integration tests skip
automatically when Qdrant is unreachable, with `-rs` so the skips are visible rather than silent.
The retrieval regression gate belongs in R3, where it can run against a real index.

## Trade-off

CI installs the full dependency set, including torch via `sentence-transformers`, so the first run
is slow; pip caching keyed on `pyproject.toml` covers later ones. The alternative — a trimmed
dependency set for CI — would mean CI exercising a different environment from the one that runs in
production, which is the class of problem decision 001 was about.

## Measured validation

`ruff check .` clean. 27 tests pass. `compare_baseline` still reproduces the committed baseline
exactly, so none of this moved retrieval behaviour.

One failure during the rename is worth recording: `tests/test_label_integrity.py` built its path as
`parent / "eval" / "eval_set.jsonl"` — assembled from parts, so a search-and-replace over the
literal string `eval/eval_set.jsonl` missed it. The label-integrity test failed and named the
problem immediately. That is the test doing exactly the job it was added for.

## Next

Docker with the index baked into its own layer, then Terraform and the deploy.

