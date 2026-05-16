"""Unit tests for orchestrator core logic (no Modal, no external services)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

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
    refine_spec,
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
    "model": "facebook/musicgen-melody-large",
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
    assert req.use_melody_conditioning is False
    assert req.cfg_coeff is None
    assert req.top_k is None
    assert req.temperature is None


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
    assert span.attrs["db.system"] == "qdrant"
    assert span.attrs["db.operation.name"] == "query"
    assert span.attrs["db.collection.name"] == "flowing-trails-corpus"
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
    result = generate_music(spec, None, None, {}, log)

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
    result = generate_music(spec, melody_data, 32000, {}, log)

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
    result = generate_music(spec, None, None, {}, log)

    assert result is not None
    prompt = mock_instance.generate.remote.call_args.kwargs["prompt"]
    assert "dark dungeon theme" in prompt
    assert "ambient exploration" in prompt
    assert "strings" in prompt
    assert "80 bpm" in prompt


@patch("modal.Cls.from_name")
def test_generate_music_forwards_gen_params(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    spec = MusicSpec(description="battle theme", duration_seconds=10.0)
    gen_params = {"cfg_coeff": 5.0, "top_k": 128, "temperature": 0.8}
    generate_music(spec, None, None, gen_params, log)

    call_kwargs = mock_instance.generate.remote.call_args.kwargs
    assert call_kwargs["cfg_coeff"] == 5.0
    assert call_kwargs["top_k"] == 128
    assert call_kwargs["temperature"] == 0.8


@patch("modal.Cls.from_name")
def test_generate_music_omits_none_gen_params(mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    spec = MusicSpec(description="town theme", duration_seconds=10.0)
    gen_params = {"cfg_coeff": None, "top_k": 200, "temperature": None}
    generate_music(spec, None, None, gen_params, log)

    call_kwargs = mock_instance.generate.remote.call_args.kwargs
    assert "cfg_coeff" not in call_kwargs
    assert call_kwargs["top_k"] == 200
    assert "temperature" not in call_kwargs


# ── _generate_with_scoring ──────────────────────────────────────────────────


class _StubSpanContext(_StubSpan):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _StubTracer:
    """Minimal tracer that yields _StubSpan contexts and records them by name."""

    def __init__(self):
        self.spans: dict[str, list[_StubSpanContext]] = {}

    def start_as_current_span(self, name):
        span = _StubSpanContext()
        self.spans.setdefault(name, []).append(span)
        return span


_QUERY_VEC = None  # set per test via fixture


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_accepts_on_first_try(mock_score, mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.return_value = 0.45

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="epic battle theme", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, {}, _StubTracer(), log
    )

    assert attempts == 1
    assert score == 0.45
    assert audio == SAMPLE_GENERATE_RESULT["audio_bytes"]
    assert mock_instance.generate.remote.call_count == 1


REFINED_SPEC = {
    "description": "Warm ambient pads with gentle reverb and soft bell tones",
    "genre": "ambient exploration",
    "mood_tags": ["calm", "warm", "peaceful"],
    "instruments": ["synth pads", "bells"],
    "tempo_bpm": 85,
    "key": "C major",
    "energy": "low",
    "duration_seconds": 10.0,
    "style_hint": None,
}


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_retries_then_accepts(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.15, 0.40]
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)
    mock_embed.return_value = np.zeros(512, dtype=np.float32)

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, {}, _StubTracer(), log
    )

    assert attempts == 2
    assert score == 0.40
    assert mock_instance.generate.remote.call_count == 2
    mock_claude.assert_called_once()
    mock_embed.assert_called_once()


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_exhausts_returns_best(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.10, 0.20]
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)
    mock_embed.return_value = np.zeros(512, dtype=np.float32)

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="retro town", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, {}, _StubTracer(), log
    )

    assert attempts == 2
    assert score == 0.20
    assert audio is not None


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_keeps_best_across_attempts(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    """Second attempt scores lower — should still return the first (better) audio."""
    result_a = {**SAMPLE_GENERATE_RESULT, "audio_bytes": b"AUDIO_A"}
    result_b = {**SAMPLE_GENERATE_RESULT, "audio_bytes": b"AUDIO_B"}
    mock_instance = MagicMock()
    mock_instance.generate.remote.side_effect = [result_a, result_b]
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.25, 0.10]
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)
    mock_embed.return_value = np.zeros(512, dtype=np.float32)

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="dungeon ambience", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, {}, _StubTracer(), log
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

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="silence", duration_seconds=10.0)

    audio, score, attempts = _generate_with_scoring(
        spec, None, None, query_vec, {}, _StubTracer(), log
    )

    assert attempts == 1
    assert audio is None
    assert score == -1.0


# ── refine_spec ─────────────────────────────────────────────────────────────


@patch("claude_client.call_claude_json")
def test_refine_spec_returns_revised_spec(mock_claude):
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)

    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)
    history = [{"spec": spec.model_dump(), "score": 0.15}]

    result = refine_spec(spec, 0.15, history, _StubTracer(), log)

    assert isinstance(result, MusicSpec)
    assert result.description == REFINED_SPEC["description"]
    assert result.mood_tags == ["calm", "warm", "peaceful"]
    mock_claude.assert_called_once()


@patch("claude_client.call_claude_json")
def test_refine_spec_enforces_readonly_fields(mock_claude):
    """Even if Claude changes style_hint or duration, they get overwritten."""
    tampered = {
        **REFINED_SPEC,
        "style_hint": "tampered hint",
        "duration_seconds": 30.0,
    }
    mock_claude.return_value = (tampered, _STUB_USAGE)

    spec = MusicSpec(
        description="epic battle",
        duration_seconds=12.0,
        style_hint="Nobuo Uematsu orchestral style",
    )
    history = [{"spec": spec.model_dump(), "score": 0.18}]

    result = refine_spec(spec, 0.18, history, _StubTracer(), log)

    assert result.style_hint == "Nobuo Uematsu orchestral style"
    assert result.duration_seconds == 12.0


@patch("claude_client.call_claude_json")
def test_refine_spec_passes_history_as_json(mock_claude):
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)

    spec = MusicSpec(description="town theme", duration_seconds=10.0)
    history = [
        {"spec": {"description": "town theme"}, "score": 0.12},
    ]

    refine_spec(spec, 0.12, history, _StubTracer(), log)

    call_args = mock_claude.call_args
    import json

    user_msg = json.loads(call_args.kwargs["user_message"])
    assert "history" in user_msg
    assert len(user_msg["history"]) == 1
    assert user_msg["history"][0]["score"] == 0.12


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_uses_refined_spec_for_generation(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    """After refinement, the next generate call uses the refined spec."""
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.15, 0.40]
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)
    mock_embed.return_value = np.zeros(512, dtype=np.float32)

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)

    _generate_with_scoring(spec, None, None, query_vec, {}, _StubTracer(), log)

    second_call_prompt = mock_instance.generate.remote.call_args_list[1].kwargs[
        "prompt"
    ]
    assert "Warm ambient pads" in second_call_prompt


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_scoring_loop_no_refine_on_accept(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    """When first attempt passes threshold, refiner is never called."""
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.return_value = 0.45

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="epic battle", duration_seconds=10.0)

    _generate_with_scoring(spec, None, None, query_vec, {}, _StubTracer(), log)

    mock_claude.assert_not_called()
    mock_embed.assert_not_called()


# ── OTel span attributes ──────────────────────────────────────────────────


@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_gen_span_has_score_attribute(mock_score, mock_from_name):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.return_value = 0.42

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="battle theme", duration_seconds=10.0)
    tracer = _StubTracer()

    _generate_with_scoring(spec, None, None, query_vec, {}, tracer, log)

    gen_spans = tracer.spans["music_generate"]
    assert len(gen_spans) == 1
    assert gen_spans[0].attrs["generate.score"] == 0.42
    assert gen_spans[0].attrs["generate.attempt"] == 1


@patch("clap_utils.embed_text")
@patch("claude_client.call_claude_json")
@patch("modal.Cls.from_name")
@patch("scoring.score_generation")
def test_gen_span_score_on_each_attempt(
    mock_score, mock_from_name, mock_claude, mock_embed
):
    mock_instance = MagicMock()
    mock_instance.generate.remote.return_value = SAMPLE_GENERATE_RESULT
    mock_from_name.return_value.return_value = mock_instance

    mock_score.side_effect = [0.15, 0.41]
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)
    mock_embed.return_value = np.zeros(512, dtype=np.float32)

    query_vec = np.zeros(512, dtype=np.float32)
    spec = MusicSpec(description="town theme", duration_seconds=10.0)
    tracer = _StubTracer()

    _generate_with_scoring(spec, None, None, query_vec, {}, tracer, log)

    gen_spans = tracer.spans["music_generate"]
    assert len(gen_spans) == 2
    assert gen_spans[0].attrs["generate.score"] == 0.15
    assert gen_spans[1].attrs["generate.score"] == 0.41


@patch("claude_client.call_claude_json")
def test_refine_span_has_score_delta(mock_claude):
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)

    spec = MusicSpec(description="calm exploration", duration_seconds=10.0)
    history = [
        {"spec": spec.model_dump(), "score": 0.10},
        {"spec": spec.model_dump(), "score": 0.18},
    ]
    tracer = _StubTracer()

    refine_spec(spec, 0.18, history, tracer, log)

    refine_spans = tracer.spans["spec_refine"]
    assert len(refine_spans) == 1
    assert refine_spans[0].attrs["refine.prior_score"] == 0.18
    assert refine_spans[0].attrs["refine.score_delta"] == 0.08


@patch("claude_client.call_claude_json")
def test_refine_span_no_delta_on_first_attempt(mock_claude):
    mock_claude.return_value = (REFINED_SPEC.copy(), _STUB_USAGE)

    spec = MusicSpec(description="dungeon theme", duration_seconds=10.0)
    history = [{"spec": spec.model_dump(), "score": 0.12}]
    tracer = _StubTracer()

    refine_spec(spec, 0.12, history, tracer, log)

    refine_spans = tracer.spans["spec_refine"]
    assert "refine.score_delta" not in refine_spans[0].attrs
