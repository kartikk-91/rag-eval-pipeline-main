> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 003 — Agent tooling, and the 11-second import it exposed

*R1 — 2026-10-02*

## Observation

Two releases in, the repository had no `CLAUDE.md`, no skills and no hooks. Every session
rediscovered the same facts: that `.venv` is the real environment and `myenv/` is a broken
leftover, that files in `data/` must never be renamed, that the eval needs Qdrant running and a
current index. Project knowledge lived in commit messages and one test docstring.

Separately, building a pre-commit hook required the fast tests to be fast. They were not: 17.9s,
almost all of it spent importing.

## Decision

Three small, used artifacts rather than an elaborate unused set:

- **`CLAUDE.md`** — environment, four invariants whose violation corrupts data *silently*, the
  commands, and the conventions.
- **`/eval` skill** plus `evaluation/baseline.json` and `evaluation/compare_baseline.py` — the eval loop is
  about to run repeatedly, and a committed baseline turns "did that help?" into a diff. The same
  script becomes the R3 CI regression gate.
- **Two hooks, deliberately different in kind.** A Claude Code `PreToolUse` hook blocks edits to
  `data/*.pdf` (enforcing invariant 2 against the agent); a committed `.githooks/pre-commit` runs
  fast tests and an AST duplicate-key check (enforcing it against everyone, via `core.hooksPath`
  so it is shared rather than living in one clone).

**No subagents.** Nothing in this project needs parallel agents, and adding them to demonstrate
familiarity would be the same resume-driven development rejected elsewhere in this plan.

## Trade-off: the import cost

Profiling to make the hook fast found that `langchain_text_splitters` takes **11.3s** to import —
it pulls in LangChain core for one text-splitting class — and `rag/ingest.py` imported it at module
level. The chain `api/main.py → generate → retriever → build_index → ingest` meant **the API paid
11 seconds at startup for a splitter it never uses at query time**. `sentence_transformers` (~13s)
was likewise imported by `build_index` purely so `retriever` could read a hostname constant.

Both are now imported inside the functions that need them.

| | before | after |
|---|---|---|
| `import rag.ingest` | 11.3s | 0.5s |
| `import rag.build_index` | ~13s | 3.6s |
| fast test suite | 17.9s | 2.5s |

The cost is a small departure from import-at-top convention, documented at each site. The payoff is
a hook fast enough that nobody disables it — and a serving path that no longer pays 11 seconds for
nothing, which matters directly for the Lambda cold start in Step 4.

A `rag/config.py` extraction would remove the underlying coupling (the retriever importing
constants from the indexer) more cleanly than deferred imports do. It is still owed, and belongs
with the packaging work in Step 4.

## Measured validation

Fast suite 17.9s → 2.5s; full suite 13 tests green and unchanged. `compare_baseline` reproduces the
committed baseline exactly (MRR 0.787, recall@1 0.350). Both hooks were tested against a planted
failure rather than assumed: the pre-commit hook rejects a file containing a duplicate dict key,
and the corpus hook refuses a `data/*.pdf` edit with and without `jq` present.

