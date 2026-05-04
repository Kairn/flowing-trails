"""Verify CLAP model loading, embedding, and resampling.

Loads laion/clap-htsat-unfused on CPU, embeds a test text and a synthetic
audio clip, prints shapes and cosine similarity, and exports an OTel span
to Grafana Tempo.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentelemetry.trace import StatusCode

from clap_utils import embed_audio, embed_text, resample_to_clap
from config import CLAP_SAMPLE_RATE, MUSICGEN_SAMPLE_RATE, QDRANT_VECTOR_SIZE
from otel_utils import (
    flush_telemetry,
    get_logger,
    get_tracer,
    setup_logging,
    setup_tracing,
)

setup_tracing()
setup_logging()
log = get_logger("verify-clap")
tracer = get_tracer("verify-clap")

TEST_TEXT = "epic orchestral boss battle music with choir and brass"


def make_synthetic_audio(
    duration_s: float = 3.0, freq_hz: float = 440.0
) -> torch.Tensor:
    """Generate a sine wave at MusicGen's native sample rate."""
    t = torch.linspace(0, duration_s, int(MUSICGEN_SAMPLE_RATE * duration_s))
    return torch.sin(2 * torch.pi * freq_hz * t)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    with tracer.start_as_current_span("verify-clap") as span:
        span.set_attribute("test.purpose", "m0-t10-clap-verify")
        span.set_attribute("test.clap_model", "laion/clap-htsat-unfused")

        # 1. Text embedding
        log.info("Embedding test text", text=TEST_TEXT)
        t0 = time.monotonic()
        text_vec = embed_text(TEST_TEXT)
        text_ms = (time.monotonic() - t0) * 1000
        span.set_attribute("test.text_embed_ms", round(text_ms, 1))
        log.info(
            "Text embedding done", shape=text_vec.shape, latency_ms=round(text_ms, 1)
        )

        assert text_vec.shape == (
            QDRANT_VECTOR_SIZE,
        ), f"Expected ({QDRANT_VECTOR_SIZE},), got {text_vec.shape}"
        assert text_vec.dtype == np.float32

        # 2. Resample synthetic audio from 32kHz → 48kHz
        raw_audio = make_synthetic_audio()
        log.info(
            "Resampling synthetic audio",
            orig_sr=MUSICGEN_SAMPLE_RATE,
            target_sr=CLAP_SAMPLE_RATE,
            orig_samples=raw_audio.shape[0],
        )
        t1 = time.monotonic()
        resampled = resample_to_clap(raw_audio)
        resample_ms = (time.monotonic() - t1) * 1000
        expected_samples = int(3.0 * CLAP_SAMPLE_RATE)
        assert (
            abs(resampled.shape[0] - expected_samples) <= 1
        ), f"Expected ~{expected_samples} samples, got {resampled.shape[0]}"
        span.set_attribute("test.resample_ms", round(resample_ms, 1))
        span.set_attribute("test.resampled_samples", resampled.shape[0])
        log.info(
            "Resample done",
            samples=resampled.shape[0],
            latency_ms=round(resample_ms, 1),
        )

        # 3. Audio embedding
        t2 = time.monotonic()
        audio_vec = embed_audio(resampled)
        audio_ms = (time.monotonic() - t2) * 1000
        span.set_attribute("test.audio_embed_ms", round(audio_ms, 1))
        log.info(
            "Audio embedding done", shape=audio_vec.shape, latency_ms=round(audio_ms, 1)
        )

        assert audio_vec.shape == (
            QDRANT_VECTOR_SIZE,
        ), f"Expected ({QDRANT_VECTOR_SIZE},), got {audio_vec.shape}"
        assert audio_vec.dtype == np.float32

        # 4. Cross-modal similarity (sine wave vs. text — expect low but non-zero)
        sim = cosine_similarity(text_vec, audio_vec)
        span.set_attribute("test.cosine_similarity", round(sim, 4))
        log.info("Cross-modal cosine similarity", similarity=round(sim, 4))

        span.set_status(StatusCode.OK)
        trace_id = format(span.get_span_context().trace_id, "032x")

    flush_telemetry()
    log.info(
        "CLAP verification complete",
        trace_id=trace_id,
        text_shape=text_vec.shape,
        audio_shape=audio_vec.shape,
        similarity=round(sim, 4),
    )


if __name__ == "__main__":
    main()
