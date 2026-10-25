"""MCP server exposing the RAG pipeline as tools for AI agents.

Wraps the existing shared components (retriever, generator, eval harness) so any
MCP-capable client can search the corpus, ask grounded questions, and check
retrieval quality — without bespoke integration code.

Run for local inspection:  mcp dev mcp_server/server.py
Requires the Qdrant container to be running (docker compose up -d).
"""

import sys
from pathlib import Path

# mcp dev loads this file directly, without the project root on sys.path —
# add it so the rag/ and eval/ packages resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from evaluation.eval_harness import DEFAULT_EVAL_SET, evaluate, load_eval_set
from rag.generate import answer_question
from rag.retriever import HybridRetriever

mcp = FastMCP("rag-eval-pipeline")

# One retriever instance shared across tool calls (model loads once).
_retriever = HybridRetriever()


@mcp.tool()
def search_documents(query: str, k: int = 5) -> list[dict]:
    """Search the research-paper corpus and return the top-k chunks.

    Args:
        query: Natural-language search query.
        k: Number of chunks to return (default 5).

    Returns:
        Ranked list of {chunk_id, source, text} dicts.
    """
    return [
        {"chunk_id": c.chunk_id, "source": c.source, "text": c.text}
        for c in _retriever.retrieve_chunks(query, k)
    ]


@mcp.tool()
def ask_question(question: str, k: int = 3) -> dict:
    """Answer a question grounded in the paper corpus, with source chunk IDs.

    Args:
        question: The question to answer.
        k: Number of chunks to retrieve as context (default 3).

    Returns:
        {answer, sources} where each source carries its chunk ID and a citation.
    """
    answer, chunks = answer_question(question, k)
    return {
        "answer": answer,
        "sources": [
            {"chunk_id": c.chunk_id, "citation": c.cite(), "section": c.section}
            for c in chunks
        ],
    }


@mcp.tool()
def evaluate_retrieval() -> dict:
    """Score the retriever against the hand-labeled eval set.

    Returns:
        {mrr, mrr_ci, per_k} with recall@k, hit_rate@k and ndcg@k for k in
        1, 3, 5, 10, plus a 95% bootstrap interval on MRR.

    The interval is included because the alternative is an agent reading 0.578
    as exact. On 42 questions it is not: the interval is roughly +/-0.13 wide,
    and a tool that reports the point estimate alone invites a confident wrong
    conclusion from whatever is consuming it.
    """
    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    report = evaluate(_retriever, eval_set, k_values=(1, 3, 5, 10))
    mrr_lo, mrr_hi = report.ci("mrr")
    return {
        "queries_evaluated": len(eval_set),
        "mrr": round(report.mrr, 3),
        "mrr_ci_95": [round(mrr_lo, 3), round(mrr_hi, 3)],
        "per_k": {
            str(k): {
                "recall": round(report.recall_at_k[k], 3),
                "hit_rate": round(report.hit_rate_at_k[k], 3),
                "ndcg": round(report.ndcg_at_k[k], 3),
            }
            for k in report.recall_at_k
        },
    }


if __name__ == "__main__":
    mcp.run()
