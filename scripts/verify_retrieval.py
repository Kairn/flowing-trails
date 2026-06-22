"""Verify retrieval search end-to-end.

Loads CLAP, embeds a test query, searches Qdrant, prints ranked results.
Runs locally (not on Modal) — requires .env with QDRANT_URL and QDRANT_API_KEY.

Usage:
    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py "calm exploration ambient forest"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from otel_utils import (
    flush_telemetry,
    get_logger,
    get_tracer,
    setup_logging,
    setup_tracing,
)

setup_tracing()
setup_logging()
log = get_logger("verify-retrieval")
tracer = get_tracer("verify-retrieval")

DEFAULT_QUERY = "epic boss battle orchestral high energy"


def main() -> None:
    from retrieval.search import search

    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    log.info("search_start", query=query)

    with tracer.start_as_current_span("verify-retrieval") as span:
        span.set_attribute("test.purpose", "retrieval-verify")
        span.set_attribute("db.system", "qdrant")
        span.set_attribute("retrieval.query", query)

        results = search(query)

        span.set_attribute("retrieval.result_count", len(results))
        if results:
            span.set_attribute("retrieval.top_score", results[0].score)

    for r in results:
        log.info(
            "result",
            rank=r.rank,
            score=round(r.score, 4),
            category=r.category,
            energy=r.energy,
            mood_tags=r.mood_tags,
            corpus_file_path=r.corpus_file_path,
            prompt=(r.prompt or "")[:80],
        )

    flush_telemetry()
    log.info(
        "search_complete",
        query=query,
        result_count=len(results),
        top_score=round(results[0].score, 4) if results else None,
    )


if __name__ == "__main__":
    main()
