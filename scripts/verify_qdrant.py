"""Verify Qdrant Cloud connection and collection setup.

Connects to the Qdrant cluster, creates the `flowing-trails-corpus`
collection if it doesn't exist (512-dim cosine), and wraps the call
in an OTel span so it appears in Grafana Tempo.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentelemetry.trace import StatusCode
from qdrant_client.models import Distance, VectorParams

from config import QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE
from qdrant_utils import make_qdrant_client
from otel_utils import (
    flush_telemetry,
    get_logger,
    get_tracer,
    setup_logging,
    setup_tracing,
)

setup_tracing()
setup_logging()
log = get_logger("verify-qdrant")
tracer = get_tracer("verify-qdrant")


def main() -> None:
    client = make_qdrant_client()

    with tracer.start_as_current_span("verify-qdrant-cloud") as span:
        span.set_attribute("test.purpose", "qdrant-verify")
        span.set_attribute("db.system", "qdrant")
        span.set_attribute("db.collection.name", QDRANT_COLLECTION_NAME)

        # Check connectivity
        log.info("Connecting to Qdrant Cloud")
        t0 = time.monotonic()
        collections = [c.name for c in client.get_collections().collections]
        connect_ms = (time.monotonic() - t0) * 1000
        span.set_attribute("test.connect_ms", round(connect_ms, 1))
        log.info(
            "Connected",
            existing_collections=collections,
            latency_ms=round(connect_ms, 1),
        )

        # Create collection if it doesn't exist
        if QDRANT_COLLECTION_NAME in collections:
            log.info("Collection already exists", collection=QDRANT_COLLECTION_NAME)
            info = client.get_collection(QDRANT_COLLECTION_NAME)
            span.set_attribute("test.collection_existed", True)
            span.set_attribute("test.points_count", info.points_count)
        else:
            log.info("Creating collection", collection=QDRANT_COLLECTION_NAME)
            t1 = time.monotonic()
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            create_ms = (time.monotonic() - t1) * 1000
            span.set_attribute("test.collection_existed", False)
            span.set_attribute("test.create_ms", round(create_ms, 1))
            log.info("Collection created", latency_ms=round(create_ms, 1))

        # Verify the collection is accessible
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        span.set_attribute("test.vector_size", info.config.params.vectors.size)
        span.set_attribute("test.distance", info.config.params.vectors.distance.value)

        span.set_status(StatusCode.OK)
        trace_id = format(span.get_span_context().trace_id, "032x")

    flush_telemetry()
    log.info(
        "Qdrant Cloud verified",
        trace_id=trace_id,
        collection=QDRANT_COLLECTION_NAME,
        vector_size=info.config.params.vectors.size,
        distance=info.config.params.vectors.distance.value,
        points_count=info.points_count,
    )


if __name__ == "__main__":
    main()
