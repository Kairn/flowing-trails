"""
Upsert CLAP embeddings into Qdrant.

Reads corpus_embeddings.json from the Modal Volume (produced by
embed_corpus.py) and batch-upserts points to the Qdrant collection.
Content-hash point IDs make re-indexing idempotent.

Usage:
    modal run retrieval/index_corpus.py
"""

import json
import sys
from pathlib import Path

import modal

from config import (
    CORPUS_EMBEDDINGS_PATH,
    MODAL_SECRET_NAME,
    QDRANT_COLLECTION_NAME,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
)

app = modal.App("flowing-trails-index-corpus")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("qdrant-client>=1.17.0")
    .add_local_python_source("config")
    .add_local_python_source("qdrant_utils")
)

UPSERT_BATCH_SIZE = 50


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=300,
)
def index_corpus() -> dict:
    import logging

    from qdrant_client.models import PointStruct

    from qdrant_utils import make_qdrant_client

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )
    log = logging.getLogger("flowing-trails.index-corpus")

    with open(CORPUS_EMBEDDINGS_PATH) as f:
        entries = json.load(f)
    log.info("Loaded %d embeddings from volume", len(entries))

    client = make_qdrant_client()

    points = []
    for entry in entries:
        payload = {
            "id": entry["id"],
            "corpus_file_path": entry["corpus_file_path"],
            "content_hash": entry["content_hash"],
            "category": entry.get("category"),
            "category_label": entry.get("category_label"),
            "mood_tags": entry.get("mood_tags"),
            "energy": entry.get("energy"),
            "instrumentation": entry.get("instrumentation"),
            "bpm_hint": entry.get("bpm_hint"),
            "prompt": entry.get("prompt"),
        }
        points.append(
            PointStruct(
                id=entry["point_id"],
                vector=entry["embedding"],
                payload=payload,
            )
        )

    upserted = 0
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)
        upserted += len(batch)
        log.info("Upserted batch %d–%d (%d total)", i, i + len(batch), upserted)

    info = client.get_collection(QDRANT_COLLECTION_NAME)
    log.info(
        "Done: %d points upserted, collection has %d points",
        upserted,
        info.points_count,
    )

    return {"upserted": upserted, "collection_points": info.points_count}


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from otel_utils import get_logger, setup_logging

    setup_logging()
    log = get_logger("index-corpus")

    log.info("Starting Qdrant indexing job")
    result = index_corpus.remote()
    log.info(
        "Indexing complete",
        upserted=result["upserted"],
        collection_points=result["collection_points"],
    )
