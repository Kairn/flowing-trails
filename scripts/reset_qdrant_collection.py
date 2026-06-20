"""Reset the Qdrant collection — delete and recreate with same config.

Run this before indexing new A/B tracks to clear out old synthetic corpus.

Usage:
    python scripts/reset_qdrant_collection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from config import QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE
from qdrant_utils import make_qdrant_client


def main() -> None:
    client = make_qdrant_client()

    info = client.get_collection(QDRANT_COLLECTION_NAME)
    print(f"Current collection: {info.points_count} points")

    from qdrant_client.models import Distance, VectorParams

    client.delete_collection(QDRANT_COLLECTION_NAME)
    print(f"Deleted collection '{QDRANT_COLLECTION_NAME}'")

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    info = client.get_collection(QDRANT_COLLECTION_NAME)
    print(
        f"Recreated collection: {info.points_count} points, {QDRANT_VECTOR_SIZE}-dim cosine"
    )


if __name__ == "__main__":
    main()
