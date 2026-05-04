"""Unit tests for orchestrator core logic (no Modal, no external services)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

_modal_stub = ModuleType("modal")
_modal_stub.App = MagicMock()
_modal_stub.Image = MagicMock()
_modal_stub.Secret = MagicMock()
_modal_stub.web_endpoint = lambda **kw: lambda fn: fn
sys.modules.setdefault("modal", _modal_stub)

from models import MusicSpec
from orchestrator.app import ComposeRequest, generate_music, parse_query


class _StubLog:
    def info(self, *a, **kw):
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


# ── generate_music (still a stub) ────────────────────────────────────────────


def test_generate_music_stub_returns_none():
    spec = MusicSpec(description="test")
    result = generate_music(spec, log)
    assert result is None
