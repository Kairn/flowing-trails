"""Unit tests for retrieval/search.py — CLAP and Qdrant are mocked."""

from __future__ import annotations

from types import SimpleNamespace

from retrieval.search import RetrievalResult, search

from .conftest import make_hits


class TestSearch:
    def test_returns_correct_count(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        assert len(results) == 3

    def test_result_types(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_ranks_sequential(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_descending(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top1_has_audio_path(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        assert results[0].corpus_file_path == "/corpus/boss_001.wav"

    def test_rank2_plus_no_audio_path(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        for r in results[1:]:
            assert r.corpus_file_path is None

    def test_metadata_populated(self, mock_clap, mock_qdrant_client):
        results = search("boss battle", client=mock_qdrant_client)
        top = results[0]
        assert top.category == "boss_battle"
        assert top.energy == "high"
        assert top.mood_tags == ["intense", "dramatic"]
        assert top.bpm_hint == 160

    def test_custom_top_k(self, mock_clap, mock_qdrant_client):
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=make_hits(2)
        )
        results = search("boss battle", top_k=2, client=mock_qdrant_client)
        mock_qdrant_client.query_points.assert_called_once()
        call_kwargs = mock_qdrant_client.query_points.call_args
        assert call_kwargs.kwargs["limit"] == 2
        assert len(results) == 2

    def test_embeds_query_text(self, mock_clap, mock_qdrant_client):
        search("calm forest exploration", client=mock_qdrant_client)
        mock_clap.assert_called_once_with("calm forest exploration")

    def test_empty_results(self, mock_clap, mock_qdrant_client):
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        results = search("nonexistent", client=mock_qdrant_client)
        assert results == []

    def test_missing_payload_fields(self, mock_clap, mock_qdrant_client):
        from .conftest import fake_hit

        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[fake_hit(0, 0.5, {})]
        )
        results = search("test", client=mock_qdrant_client)
        r = results[0]
        assert r.category is None
        assert r.mood_tags == []
        assert r.energy is None
        assert r.corpus_file_path is None
