"""Shared test fixtures and helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

SAMPLE_PAYLOADS = [
    {
        "category": "boss_battle",
        "mood_tags": ["intense", "dramatic"],
        "energy": "high",
        "instrumentation": ["orchestra", "choir"],
        "bpm_hint": 160,
        "prompt": "Epic orchestral boss battle theme",
        "corpus_file_path": "/corpus/boss_001.wav",
    },
    {
        "category": "exploration",
        "mood_tags": ["calm", "wonder"],
        "energy": "low",
        "instrumentation": ["piano", "strings"],
        "bpm_hint": 80,
        "prompt": "Calm exploration piano theme",
        "corpus_file_path": "/corpus/explore_001.wav",
    },
    {
        "category": "town",
        "mood_tags": ["cheerful"],
        "energy": "medium",
        "instrumentation": ["flute", "guitar"],
        "bpm_hint": 120,
        "prompt": "Cheerful town theme",
        "corpus_file_path": "/corpus/town_001.wav",
    },
]


def fake_hit(point_id: int, score: float, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def make_hits(n: int = 3) -> list[SimpleNamespace]:
    return [fake_hit(i, round(0.9 - i * 0.1, 2), SAMPLE_PAYLOADS[i]) for i in range(n)]


@pytest.fixture
def mock_clap():
    from unittest.mock import patch

    with patch("retrieval.search.embed_text") as m:
        m.return_value = np.zeros(512, dtype=np.float32)
        yield m


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=make_hits(3))
    return client


@pytest.fixture
def mock_qdrant(mock_qdrant_client):
    from unittest.mock import patch

    with patch("retrieval.search.make_qdrant_client", return_value=mock_qdrant_client):
        yield mock_qdrant_client
