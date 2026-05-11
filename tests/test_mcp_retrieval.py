"""Unit tests for mcp_servers/retrieval_server.py — CLAP and Qdrant are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcp_servers.retrieval_server import _result_to_dict, search_corpus
from retrieval.search import RetrievalResult


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
    with patch("retrieval.search.make_qdrant_client", return_value=client):
        yield client


class TestResultToDict:
    def test_includes_all_metadata_fields(self):
        r = RetrievalResult(
            rank=1,
            score=0.85,
            category="boss_battle",
            mood_tags=["intense"],
            energy="high",
            instrumentation=["orchestra"],
            bpm_hint=160,
            prompt="Epic boss theme",
            corpus_file_path="/corpus/boss_001.wav",
        )
        d = _result_to_dict(r)
        assert d["rank"] == 1
        assert d["score"] == 0.85
        assert d["category"] == "boss_battle"
        assert d["mood_tags"] == ["intense"]
        assert d["energy"] == "high"
        assert d["instrumentation"] == ["orchestra"]
        assert d["bpm_hint"] == 160
        assert d["prompt"] == "Epic boss theme"
        assert d["corpus_file_path"] == "/corpus/boss_001.wav"

    def test_omits_corpus_path_when_none(self):
        r = RetrievalResult(rank=2, score=0.7)
        d = _result_to_dict(r)
        assert "corpus_file_path" not in d

    def test_score_rounded(self):
        r = RetrievalResult(rank=1, score=0.85678)
        d = _result_to_dict(r)
        assert d["score"] == 0.8568


class TestSearchCorpusTool:
    def test_returns_list_of_dicts(self, mock_clap, mock_qdrant):
        results = search_corpus("boss battle")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_result_count(self, mock_clap, mock_qdrant):
        results = search_corpus("boss battle")
        assert len(results) == 3

    def test_top1_has_corpus_path(self, mock_clap, mock_qdrant):
        results = search_corpus("boss battle")
        assert "corpus_file_path" in results[0]
        assert results[0]["corpus_file_path"] == "/corpus/boss_001.wav"

    def test_rank2_plus_no_corpus_path(self, mock_clap, mock_qdrant):
        results = search_corpus("boss battle")
        for r in results[1:]:
            assert "corpus_file_path" not in r

    def test_custom_top_k(self, mock_clap, mock_qdrant):
        mock_qdrant.query_points.return_value = SimpleNamespace(points=_make_hits(2))
        results = search_corpus("boss battle", top_k=2)
        assert len(results) == 2
        assert mock_qdrant.query_points.call_args.kwargs["limit"] == 2

    def test_empty_results(self, mock_clap, mock_qdrant):
        mock_qdrant.query_points.return_value = SimpleNamespace(points=[])
        results = search_corpus("nonexistent")
        assert results == []

    def test_scores_descending(self, mock_clap, mock_qdrant):
        results = search_corpus("boss battle")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
