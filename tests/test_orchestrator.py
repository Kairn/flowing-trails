"""Unit tests for orchestrator core logic (no Modal, no external services)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Stub out modal before importing orchestrator — Modal isn't importable locally
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


def test_parse_query_returns_music_spec():
    req = ComposeRequest(
        description="dark dungeon ambient",
        tempo_bpm=70,
        instruments=["strings"],
        duration_seconds=20.0,
        key="D minor",
    )
    spec = parse_query(req, log)
    assert isinstance(spec, MusicSpec)
    assert spec.description == "dark dungeon ambient"
    assert spec.tempo_bpm == 70
    assert spec.instruments == ["strings"]
    assert spec.duration_seconds == 20.0
    assert spec.key == "D minor"


def test_parse_query_defaults():
    req = ComposeRequest(description="retro chiptune")
    spec = parse_query(req, log)
    assert spec.instruments == []
    assert spec.duration_seconds == 10.0
    assert spec.tempo_bpm is None


def test_generate_music_stub_returns_none():
    spec = MusicSpec(description="test")
    result = generate_music(spec, log)
    assert result is None


def test_parse_query_to_prompt_roundtrip():
    req = ComposeRequest(
        description="soaring orchestral theme",
        tempo_bpm=140,
        instruments=["brass", "timpani"],
        key="Bb major",
    )
    spec = parse_query(req, log)
    prompt = spec.to_prompt()
    assert "soaring orchestral theme" in prompt
    assert "140 bpm" in prompt
    assert "brass" in prompt
    assert "Bb major" in prompt
