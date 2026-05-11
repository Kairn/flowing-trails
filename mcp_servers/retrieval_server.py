"""MCP server exposing the FlowingTrails retrieval service as a tool.

Standalone wrapper — not used by the orchestrator internally.
Run with: python -m mcp_servers.retrieval_server
Or configure as an MCP server in Claude Desktop / Claude Code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure project root is importable (for config, retrieval, etc.)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from otel_utils import get_tracer, setup_tracing
from retrieval.search import RetrievalResult, search

setup_tracing()
_tracer = get_tracer("mcp.retrieval")

mcp = FastMCP("flowing-trails-retrieval")


def _result_to_dict(r: RetrievalResult) -> dict:
    d = {
        "rank": r.rank,
        "score": round(r.score, 4),
        "category": r.category,
        "mood_tags": r.mood_tags,
        "energy": r.energy,
        "instrumentation": r.instrumentation,
        "bpm_hint": r.bpm_hint,
        "prompt": r.prompt,
    }
    if r.corpus_file_path is not None:
        d["corpus_file_path"] = r.corpus_file_path
    return d


@mcp.tool()
def search_corpus(query: str, top_k: int = 3) -> list[dict]:
    """Search the VGM reference corpus for tracks matching a text description.

    Uses CLAP text embeddings and Qdrant vector search to find the most
    similar reference tracks. Top-1 result includes the corpus file path
    for melody conditioning.

    Args:
        query: Natural-language description of the desired sound
               (e.g. "calm forest exploration with soft piano")
        top_k: Number of results to return (default 3)
    """
    with _tracer.start_as_current_span(
        "mcp.search_corpus",
        attributes={
            "mcp.tool": "search_corpus",
            "db.system": "qdrant",
            "retrieval.top_k": top_k,
        },
    ) as span:
        results = search(query, top_k=top_k)
        if results:
            span.set_attribute("retrieval.top_score", results[0].score)
            span.set_attribute("retrieval.result_count", len(results))
        return [_result_to_dict(r) for r in results]


if __name__ == "__main__":
    mcp.run()
