> **Historical record:** This decision describes an earlier implementation. Corpus-specific claims and measurements refer to the previous corpus and do not characterize the active Hindi/Indic collection. Current behavior and evaluation status are in `README.md` and `docs/audit-2026-10-25.md`.

# 001 — One retrieval path for evaluation and serving

*R1, Step 1 — 2026-09-28*

## Observation

The project claimed that a retrieval improvement measured offline would reach production,
because both sides shared a `Retriever` interface. Reading the code, they did not. Five places
issued or unpacked a Qdrant query:

| Site | Used by |
|---|---|
| `rag/retriever.py: DenseRetriever.retrieve` | eval harness, MCP `evaluate_retrieval` |
| `rag/build_index.py: search()` | `generate.py` → FastAPI `/ask` → LangGraph agent |
| `evaluation/eval_harness.py: QdrantDenseRetriever` | nothing — a third copy, dead |
| `mcp_server/server.py: search_documents` | reached into `_retriever._client` / `_model` |
| `rag/label_helper.py` | its own client + model, used to pick gold labels |

Two consequences. First, the recall/MRR numbers in the README described a code path that no user
ever hit. Second — worse, and only visible once the labelling tool was examined — gold labels were
chosen from a ranking produced by a *different* query than the one being scored.

## Hypothesis

One implementation of the query, with two views of its result, removes the divergence by
construction rather than by convention.

## Decision

`DenseRetriever.retrieve_chunks(query, k) -> list[Chunk]` is the only Qdrant query in the codebase.
`retrieve(query, k) -> list[str]` is defined as `[c.chunk_id for c in retrieve_chunks(...)]`, which
satisfies the harness's existing `Retriever` protocol unchanged. A new `rag/types.py: Chunk`
carries `chunk_id`, `source`, `text`, `score` so the Qdrant payload shape stops leaking into
callers. `build_index.search()` and the dead `QdrantDenseRetriever` are deleted; the agent, the MCP
tools and the labelling helper all go through the retriever.

## Trade-off

The alternative was to widen the harness protocol to return chunk objects. Rejected: it forces
edits to `evaluate()`, the reference retriever, the stub retriever and the self-check block, and it
weakens the harness's design principle that gold labels are chunk IDs and the harness never needs
document text. The chosen shape costs one list comprehension per eval query and one small shared
type. `Chunk` deliberately omits `document_id` / `page` / `section` until R1 Step 3 produces them —
a field that is always `None` is a lie.

## Measured validation

The refactor is behaviour-preserving: the eval reproduces the published baseline exactly.

| | recall@1 | recall@3 | recall@5 | recall@10 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| README baseline | 0.35 | 0.65 | 0.80 | 0.95 | 1.00 | 0.79 |
| After unification | 0.350 | 0.650 | 0.800 | 0.950 | 1.000 | 0.787 |

13 tests pass. `tests/test_retrieval_parity.py` asserts that the eval path and the generation path
return identical chunk IDs for the same query, that `/ask` reports exactly the retriever's top-k in
order, and — structurally — that `build_index.search` cannot come back.

## Next

R1 Step 3 extends `Chunk` with page-level provenance; R2 adds `CrossEncoderRetriever` behind the
same interface, which is now the only thing that has to be swapped to change retrieval strategy.

