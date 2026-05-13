"""CI eval — CLAP-only scoring of golden reference tracks against their specs.

No LLM calls, no MusicGen, no external services. Runs entirely on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import DEFAULT_SIMILARITY_THRESHOLD
from models import MusicSpec
from scoring import score_generation

GOLDEN_DIR = Path(__file__).parent
GOLDEN_PROMPTS = GOLDEN_DIR / "golden_prompts.json"


def _load_golden_prompts() -> list[dict]:
    with open(GOLDEN_PROMPTS) as f:
        return json.load(f)


def _golden_ids() -> list[str]:
    return [p["id"] for p in _load_golden_prompts()]


@pytest.fixture(scope="module")
def golden_prompts() -> list[dict]:
    return _load_golden_prompts()


@pytest.fixture(scope="module")
def clap_embeddings(golden_prompts: list[dict]) -> dict:
    from clap_utils import embed_text

    embeddings = {}
    for entry in golden_prompts:
        spec = MusicSpec(**entry["spec"])
        embeddings[entry["id"]] = embed_text(spec.clap_text())
    return embeddings


@pytest.mark.ci
@pytest.mark.parametrize("prompt_id", _golden_ids())
def test_golden_score(
    prompt_id: str, golden_prompts: list[dict], clap_embeddings: dict
):
    entry = next(p for p in golden_prompts if p["id"] == prompt_id)
    wav_path = GOLDEN_DIR / f"{prompt_id}.wav"
    assert wav_path.exists(), f"Missing golden WAV: {wav_path}"

    audio_bytes = wav_path.read_bytes()
    query_vector = clap_embeddings[prompt_id]
    score = score_generation(audio_bytes, query_vector)

    assert score >= DEFAULT_SIMILARITY_THRESHOLD, (
        f"{prompt_id}: score {score:.4f} < threshold {DEFAULT_SIMILARITY_THRESHOLD} "
        f"(baseline was {entry['baseline_score']})"
    )
