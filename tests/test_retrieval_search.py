"""Unit tests for retrieval/search.py — CLAP and Qdrant are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.search import RetrievalResult, search


def _fake_hit(point_id: int, score: float, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def _make_hits(n: int = 3) -> list[SimpleNamespace]:
    payloads = [
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
    return [_fake_hit(i, round(0.9 - i * 0.1, 2), payloads[i]) for i in range(n)]


@pytest.fixture
def mock_clap():
    with patch("retrieval.search.embed_text") as m:
        m.return_value = np.zeros(512, dtype=np.float32)
        yield m


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=_make_hits(3))
    return client


class TestSearch:
    def test_returns_correct_count(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        assert len(results) == 3

    def test_result_types(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_ranks_sequential(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_descending(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top1_has_audio_path(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        assert results[0].corpus_file_path == "/corpus/boss_001.wav"

    def test_rank2_plus_no_audio_path(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        for r in results[1:]:
            assert r.corpus_file_path is None

    def test_metadata_populated(self, mock_clap, mock_qdrant):
        results = search("boss battle", client=mock_qdrant)
        top = results[0]
        assert top.category == "boss_battle"
        assert top.energy == "high"
        assert top.mood_tags == ["intense", "dramatic"]
        assert top.bpm_hint == 160

    def test_custom_top_k(self, mock_clap, mock_qdrant):
        mock_qdrant.query_points.return_value = SimpleNamespace(points=_make_hits(2))
        results = search("boss battle", top_k=2, client=mock_qdrant)
        mock_qdrant.query_points.assert_called_once()
        call_kwargs = mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs["limit"] == 2
        assert len(results) == 2

    def test_embeds_query_text(self, mock_clap, mock_qdrant):
        search("calm forest exploration", client=mock_qdrant)
        mock_clap.assert_called_once_with("calm forest exploration")

    def test_empty_results(self, mock_clap, mock_qdrant):
        mock_qdrant.query_points.return_value = SimpleNamespace(points=[])
        results = search("nonexistent", client=mock_qdrant)
        assert results == []

    def test_missing_payload_fields(self, mock_clap, mock_qdrant):
        mock_qdrant.query_points.return_value = SimpleNamespace(
            points=[_fake_hit(0, 0.5, {})]
        )
        results = search("test", client=mock_qdrant)
        r = results[0]
        assert r.category is None
        assert r.mood_tags == []
        assert r.energy is None
        assert r.corpus_file_path is None
