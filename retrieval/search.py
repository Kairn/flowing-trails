"""Retrieval search: text query → ranked reference tracks from Qdrant.

Plain Python module, imported by the orchestrator in-process.
CLAP text embedding + Qdrant vector search. No separate deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from clap_utils import embed_text
from config import QDRANT_COLLECTION_NAME, QDRANT_TOP_K
from qdrant_utils import make_qdrant_client

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass
class RetrievalResult:
    rank: int
    score: float
    category: str | None = None
    mood_tags: list[str] = field(default_factory=list)
    energy: str | None = None
    instrumentation: list[str] = field(default_factory=list)
    bpm_hint: int | None = None
    prompt: str | None = None
    corpus_file_path: str | None = None


def search(
    text_query: str,
    *,
    top_k: int = QDRANT_TOP_K,
    client: QdrantClient | None = None,
) -> list[RetrievalResult]:
    """Embed text_query via CLAP, search Qdrant, return ranked results.

    Top-1 includes corpus_file_path for melody conditioning.
    Ranks 2+ return metadata only (no audio path).
    """
    query_vector = embed_text(text_query).tolist()

    if client is None:
        client = make_qdrant_client()

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    results: list[RetrievalResult] = []
    for rank, hit in enumerate(hits, start=1):
        p = hit.payload or {}
        result = RetrievalResult(
            rank=rank,
            score=hit.score,
            category=p.get("category"),
            mood_tags=p.get("mood_tags") or [],
            energy=p.get("energy"),
            instrumentation=p.get("instrumentation") or [],
            bpm_hint=p.get("bpm_hint"),
            prompt=p.get("prompt"),
            corpus_file_path=p.get("corpus_file_path") if rank == 1 else None,
        )
        results.append(result)

    return results
