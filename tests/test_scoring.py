"""Tests for scoring.score_generation."""

from __future__ import annotations

import io
import struct
import wave
from unittest.mock import patch

import numpy as np
import pytest

from config import MUSICGEN_SAMPLE_RATE


def _make_wav(
    sr: int = MUSICGEN_SAMPLE_RATE, duration: float = 1.0, channels: int = 1
) -> bytes:
    """Create a valid 16-bit PCM WAV buffer with a sine tone."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if channels == 1:
            wf.writeframes(samples.tobytes())
        else:
            interleaved = np.column_stack([samples] * channels).flatten()
            wf.writeframes(interleaved.tobytes())
    return buf.getvalue()


def _fake_embed_audio(waveform):
    """Return a deterministic unit vector based on waveform energy."""
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    return vec


@pytest.fixture
def mock_clap_scoring():
    with patch("scoring.embed_audio", side_effect=_fake_embed_audio) as m:
        yield m


class TestScoreGeneration:
    def test_perfect_match(self, mock_clap_scoring):
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0
        audio = _make_wav()

        score = score_generation(audio, query)

        assert score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self, mock_clap_scoring):
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[1] = 1.0
        audio = _make_wav()

        score = score_generation(audio, query)

        assert score == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self, mock_clap_scoring):
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[0] = -1.0
        audio = _make_wav()

        score = score_generation(audio, query)

        assert score == pytest.approx(-1.0, abs=1e-5)

    def test_returns_float(self, mock_clap_scoring):
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0
        audio = _make_wav()

        score = score_generation(audio, query)

        assert isinstance(score, float)

    def test_stereo_downmix(self, mock_clap_scoring):
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0
        audio = _make_wav(channels=2)

        score = score_generation(audio, query)

        assert score == pytest.approx(1.0, abs=1e-5)

    def test_calls_resample(self, mock_clap_scoring):
        from scoring import score_generation

        with patch(
            "scoring.resample_to_clap", wraps=__import__("scoring").resample_to_clap
        ) as mock_resample:
            query = np.zeros(512, dtype=np.float32)
            query[0] = 1.0
            audio = _make_wav()

            score_generation(audio, query)

            mock_resample.assert_called_once()
            call_kwargs = mock_resample.call_args
            assert call_kwargs[1]["orig_sr"] == MUSICGEN_SAMPLE_RATE

    def test_non_standard_sample_rate(self, mock_clap_scoring):
        """WAV at 44100 Hz should still work — resample uses detected SR."""
        from scoring import score_generation

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0
        audio = _make_wav(sr=44100)

        score = score_generation(audio, query)

        assert -1.0 <= score <= 1.0


class TestCosineSimilarity:
    def test_identical(self):
        from scoring import _cosine_similarity

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_zero_vector(self):
        from scoring import _cosine_similarity

        a = np.array([1.0, 0.0], dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        assert _cosine_similarity(a, z) == pytest.approx(0.0)

    def test_known_angle(self):
        from scoring import _cosine_similarity

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        expected = 1.0 / np.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-5)
