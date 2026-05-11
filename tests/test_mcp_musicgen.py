"""Unit tests for mcp_servers/musicgen_server.py — Modal remote call is mocked."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

FAKE_AUDIO = b"\x00\x01\x02\x03" * 100
FAKE_RESULT = {
    "audio_bytes": FAKE_AUDIO,
    "sample_rate": 32000,
    "model": "facebook/musicgen-melody",
    "decoder": "mbd",
    "duration_seconds": 10.0,
    "latency_ms": 4200.5,
}


@pytest.fixture
def mock_modal():
    mock_service = MagicMock()
    mock_service.generate.remote.return_value = FAKE_RESULT.copy()

    mock_cls = MagicMock()
    mock_cls.return_value = mock_service

    with patch("mcp_servers.musicgen_server.modal") as m:
        m.Cls.from_name.return_value = mock_cls
        yield m, mock_cls, mock_service


class TestGenerateMusicTool:
    def test_returns_dict(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        result = generate_music("boss battle theme")
        assert isinstance(result, dict)

    def test_audio_is_base64_encoded(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        result = generate_music("boss battle theme")
        decoded = base64.b64decode(result["audio_base64"])
        assert decoded == FAKE_AUDIO

    def test_metadata_fields(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        result = generate_music("boss battle theme")
        assert result["sample_rate"] == 32000
        assert result["model"] == "facebook/musicgen-melody"
        assert result["decoder"] == "mbd"
        assert result["duration_seconds"] == 10.0
        assert result["latency_ms"] == 4200.5

    def test_no_raw_audio_bytes_in_result(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        result = generate_music("boss battle theme")
        assert "audio_bytes" not in result

    def test_calls_modal_with_correct_app(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("boss battle theme")
        m, _, _ = mock_modal
        m.Cls.from_name.assert_called_once_with(
            "flowing-trails-musicgen", "MusicGenService"
        )

    def test_passes_prompt_and_defaults(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("calm forest exploration")
        _, _, mock_service = mock_modal
        call_kwargs = mock_service.generate.remote.call_args.kwargs
        assert call_kwargs["prompt"] == "calm forest exploration"
        assert call_kwargs["duration_seconds"] == 10.0
        assert call_kwargs["seed"] is None

    def test_custom_duration(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("town theme", duration_seconds=15.0)
        _, _, mock_service = mock_modal
        call_kwargs = mock_service.generate.remote.call_args.kwargs
        assert call_kwargs["duration_seconds"] == 15.0

    def test_custom_seed(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("town theme", seed=42)
        _, _, mock_service = mock_modal
        call_kwargs = mock_service.generate.remote.call_args.kwargs
        assert call_kwargs["seed"] == 42

    def test_trace_context_injected(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("boss battle theme")
        _, _, mock_service = mock_modal
        call_kwargs = mock_service.generate.remote.call_args.kwargs
        assert "trace_context" in call_kwargs
        assert isinstance(call_kwargs["trace_context"], dict)

    def test_no_melody_passed(self, mock_modal):
        from mcp_servers.musicgen_server import generate_music

        generate_music("boss battle theme")
        _, _, mock_service = mock_modal
        call_kwargs = mock_service.generate.remote.call_args.kwargs
        assert "melody_wav" not in call_kwargs
        assert "melody_sample_rate" not in call_kwargs
