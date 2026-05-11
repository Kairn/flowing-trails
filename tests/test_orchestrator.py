"""Unit tests for orchestrator core logic (no Modal, no external services)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, mock_open, patch

_modal_stub = ModuleType("modal")
_modal_stub.App = MagicMock()
_modal_stub.Cls = MagicMock()
_modal_stub.Image = MagicMock()
_modal_stub.Secret = MagicMock()
_modal_stub.Volume = MagicMock()
_modal_stub.web_endpoint = lambda **kw: lambda fn: fn
_modal_stub.fastapi_endpoint = lambda **kw: lambda fn: fn
sys.modules.setdefault("modal", _modal_stub)

from models import MusicSpec
from orchestrator.app import (
    ComposeRequest,
    _embed_query,
    _generate_with_scoring,
    generate_music,
    parse_query,
    retrieve_melody,
)


class _StubLog:
    def info(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass


log = _StubLog()

_STUB_USAGE = MagicMock(input_tokens=100, output_tokens=50)

SAMPLE_SPEC = {
    "description": "Tense orchestral battle theme with driving strings and brass stabs",
    "genre": "JRPG battle theme",
    "mood_tags": ["tense", "epic", "aggressive"],
    "instruments": ["orchestral strings", "brass", "timpani"],
    "tempo_bpm": 155,
    "key": "D minor",
    "energy": "high",
    "duration_seconds": 12.0,
    "style_hint": "Nobuo Uematsu orchestral style, PS1 era",
}

SAMPLE_GENERATE_RESULT = {
    "audio_bytes": b"RIFF\x00\x00\x00\x00WAVEfmt ",
    "sample_rate": 32000,
    "model": "facebook/musicgen-melody",
    "decoder": "mbd",
    "duration_seconds": 10.0,
    "latency_ms": 4200.0,
}


# ── ComposeRequest validation ────────────────────────────────────────────────


def test_compose_request_minimal():
    req = ComposeRequest(description="epic boss battle theme")
    assert req.description == "epic boss battle theme"
    assert req.tempo_bpm is None
    assert req.instruments is None
    assert req.duration_seconds is None
    assert req.key is None


def test_compose_request_full():
    req = ComposeRequest(
        description="calm village music",
        tempo_bpm=90,
        instruments=["acoustic guitar", "flute"],
        duration_seconds=15.0,
        key="G major",
    )
    assert req.tempo_bpm == 90
    assert req.instruments == ["acoustic guitar", "flute"]
    assert req.duration_seconds == 15.0


# ── parse_query ──────────────────────────────────────────────────────────────


@patch("claude_client.call_claude_json", return_value=(SAMPLE_SPEC, _STUB_USAGE))
def test_parse_query_returns_music_spec(_mock):
    req = ComposeRequest(description="epic boss battle theme")
    spec = parse_query(req, log)

    assert isinstance(spec, MusicSpec)
    assert spec.genre == "JRPG battle theme"
    assert spec.energy == "high"
    assert spec.mood_tags == ["tense", "epic", "aggressive"]
    assert spec.style_hint == "Nobuo Uematsu orchestral style, PS1 era"


@patch("claude_client.call_claude_json", return_value=(SAMPLE_SPEC.copy(), _STUB_USAGE))
def test_parse_query_overrides_from_request(_mock):
    req = ComposeRequest(
        description="epic boss battle theme",
        tempo_bpm=180,
        instruments=["piano"],
        duration_seconds=20.0,
        key="C minor",
    )
    spec = parse_query(req, log)

    assert spec.tempo_bpm == 180
    assert spec.instruments == ["piano"]
    assert spec.duration_seconds == 20.0
    assert spec.key == "C minor"
    assert spec.genre == "JRPG battle theme"
    assert spec.energy == "high"


@patch("claude_client.call_claude_json")
def test_parse_query_minimal_claude_response(mock_call):
    minimal = {"description": "gentle ambient pads with soft reverb"}
    mock_call.return_value = (minimal, _STUB_USAGE)

    req = ComposeRequest(description="something calm")
    spec = parse_query(req, log)

    assert spec.description == "gentle ambient pads with soft reverb"
    assert spec.genre is None
    assert spec.mood_tags == []
    assert spec.instruments == []
    assert spec.duration_seconds == 10.0


@patch("claude_client.call_claude_json", return_value=(SAMPLE_SPEC, _STUB_USAGE))
def test_parse_query_to_prompt_roundtrip(_mock):
    req = ComposeRequest(description="epic boss battle")
    spec = parse_query(req, log)
    prompt = spec.to_prompt()

    assert "Tense orchestral battle theme" in prompt
    assert "155 bpm" in prompt
    assert "brass" in prompt
    assert "D minor" in prompt


# ── retrieve_melody ─────────────────────────────────────────────────────────


class _StubSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


@patch("retrieval.search.search")
def test_retrieve_melody_loads_top1_audio(mock_search):
    from retrieval.search import RetrievalResult

    mock_search.return_value = [
        RetrievalResult(
            rank=1,
            score=0.85,
            category="boss_battle",
            corpus_file_path="boss_001.wav",
        ),
        RetrievalResult(rank=2, score=0.72, category="exploration"),
    ]

    spec = MusicSpec(description="epic battle theme", mood_tags=["intense"])
    span = _StubSpan()

    with patch("builtins.open", mock_open(read_data=b"RIFF_WAV_DATA")):
        melody_bytes, sr = retrieve_melody(spec, span, log)

    assert melody_bytes == b"RIFF_WAV_DATA"
    assert sr == 32000
    assert span.attrs["retrieval.result_count"] == 2
    assert span.attrs["retrieval.top_score"] == 0.85
    assert span.attrs["retrieval.melody_loaded"] is True
    mock_search.assert_called_once_with(spec.clap_text())


@patch("retrieval.search.search")
def test_retrieve_melody_no_results(mock_search):
    mock_search.return_value = []

    spec = MusicSpec(description="something unique")
    span = _StubSpan()

    melody_bytes, sr = retrieve_melody(spec, span, log)

    assert melody_bytes is None
    assert sr is None
    assert span.attrs["retrieval.result_count"] == 0


@patch("retrieval.search.search")
def test_retrieve_melody_file_missing(mock_search):
    from retrieval.search import RetrievalResult

    mock_search.return_value = [
        RetrievalResult(
            rank=1, score=0.6, category="town", corpus_file_path="missing.wav"
        ),
    ]

    spec = MusicSpec(description="town theme")
    span = _StubSpan()

    with patch("builtins.open", side_effect=FileNotFoundError):
        melody_bytes, sr = retrieve_melody(spec, span, log)

    assert melody_bytes is None
    assert sr is None
    assert span.attrs["retrieval.melody_loaded"] is False


@patch("retrieval.search.search")
def test_retrieve_melody_no_corpus_path(mock_search):
    from retrieval.search import RetrievalResult

    mock_search.return_value = [
        RetrievalResult(rank=1, score=0.4, category="ambient", corpus_file_path=None),
    ]

    spec = MusicSpec(description="ambient pad")
    span = _StubSpan()

    melody_bytes, sr = retrieve_melody(spec, span, log)

    assert melody_bytes is None
    assert sr is None


# ── generate_music ───────────────────────────────────────────────────────────


@patch("modal.Cls.from_name")
def test_generate_music_without_melody(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    spec = MusicSpec(description="epic battle theme", duration_seconds=10.0)
    result = generate_music(spec, None, None, log)

    assert result == SAMPLE_GENERATE_RESULT["audio_bytes"]
    mock_from_name.assert_called_once_with("flowing-trails-musicgen", "MusicGenService")
    call_kwargs = mock_instance.generate.remote.call_args.kwargs
    assert "epic battle theme" in call_kwargs["prompt"]
    assert call_kwargs["duration_seconds"] == 10.0
    assert call_kwargs["melody_wav"] is None
    assert call_kwargs["melody_sample_rate"] is None


@patch("modal.Cls.from_name")
def test_generate_music_with_melody(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)
    melody_data = b"RIFF_MELODY_WAV"
    result = generate_music(spec, melody_data, 32000, log)

    assert result == SAMPLE_GENERATE_RESULT["audio_bytes"]
    call_kwargs = mock_instance.generate.remote.call_args.kwargs
    assert call_kwargs["melody_wav"] == melody_data
    assert call_kwargs["melody_sample_rate"] == 32000


@patch("modal.Cls.from_name")
def test_generate_music_passes_full_prompt(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    spec = MusicSpec(
        description="dark dungeon theme",
        genre="ambient exploration",
        mood_tags=["eerie", "tense"],
        instruments=["strings", "choir"],
        tempo_bpm=80,
        key="C minor",
        energy="low",
        style_hint="Koji Kondo style",
    )
    result = generate_music(spec, None, None, log)

    assert result is not None
    prompt = mock_instance.generate.remote.call_args.kwargs["prompt"]
    assert "dark dungeon theme" in prompt
    assert "ambient exploration" in prompt
    assert "strings" in prompt
    assert "80 bpm" in prompt


# ── _generate_with_scoring ──────────────────────────────────────────────────


class _StubTracer:
    """Minimal tracer that yields _StubSpan contexts."""

    def start_as_current_span(self, name):
        return _StubSpanContext()


class _StubSpanContext(_StubSpan):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


_QUERY_VEC = None  # set per test via fixture


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_accepts_on_first_try(mock_score, mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.return_value = 0.45

    import numpy as np

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="epic battle theme", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, _StubTracer(), log
    )

    assert attempts == 1
    assert score == 0.45
    assert audio == SAMPLE_GENERATE_RESULT["audio_bytes"]
    assert mock_instance.generate.remote.call_count == 1


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_retries_then_accepts(mock_score, mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.15, 0.40]

    import numpy as np

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, _StubTracer(), log
    )

    assert attempts == 2
    assert score == 0.40
    assert mock_instance.generate.remote.call_count == 2


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_exhausts_returns_best(mock_score, mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.10, 0.20]

    import numpy as np

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="retro town", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, _StubTracer(), log
    )

    assert attempts == 2
    assert score == 0.20
    assert audio is not None


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_keeps_best_across_attempts(mock_score, mock_from_name):
    """Second attempt scores lower — should still return the first (better) audio."""
    result_a = {**SAMPLE_GENERATE_RESULT, "audio_bytes": b"AUDIO_A"}
    result_b = {**SAMPLE_GENERATE_RESULT, "audio_bytes": b"AUDIO_B"}
    mock_instance = MagicMock()
    mock_instance.generate.remote.side_effect = [result_a, result_b]
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.25, 0.10]

    import numpy as np

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="dungeon ambience", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, _StubTracer(), log
    )

    assert attempts == 2
    assert score == 0.25
    assert audio == b"AUDIO_A"


@patch("modal.Cls.from_name")
def test_scoring_loop_generate_returns_none(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = {
        **SAMPLE_GENERATE_RESULT,
        "audio_bytes": None,
    }
    mock_from_name.return_value.return_value = mock_instance

    import numpy as np

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="silence", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, _StubTracer(), log
    )

    assert attempts == 1
    assert audio is None
    assert score == -1.0
