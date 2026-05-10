"""Qdrant client factory — single source for client construction."""

import os

from qdrant_client import QdrantClient


def make_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set")
    return QdrantClient(url=url, api_key=api_key)
