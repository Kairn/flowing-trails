"""CI eval — CLAP-only scoring of golden reference tracks against their specs.

No LLM calls, no MusicGen, no external services. Runs entirely on CPU.

This is a scoring-pipeline regression gate: each golden WAV is re-scored against
its frozen spec and must stay within a margin of its recorded baseline. Scoring is
deterministic, so a passing run means resample → CLAP embed → cosine still behave
identically; a real break (wrong resample, embedding change, CLAP version drift)
craters the score well past the margin. The gate is intentionally decoupled from
the production acceptance threshold so model recalibration never turns CI red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import MusicSpec
from scoring import score_generation

# Allowed downward drift from a track's recorded baseline before CI fails.
# Scoring is deterministic (recomputed == baseline locally), so this only needs
# to absorb cross-environment float noise while still catching real regressions.
GOLDEN_REGRESSION_MARGIN = 0.05

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

    floor = entry["baseline_score"] - GOLDEN_REGRESSION_MARGIN
    assert score >= floor, (
        f"{prompt_id}: score {score:.4f} < floor {floor:.4f} "
        f"(baseline {entry['baseline_score']} − margin {GOLDEN_REGRESSION_MARGIN})"
    )
