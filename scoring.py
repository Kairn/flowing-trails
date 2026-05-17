"""Post-generation CLAP scoring — compares generated audio against the query intent."""

from __future__ import annotations

import io
import wave

import numpy as np
import torch
from numpy.typing import NDArray

from clap_utils import embed_audio, resample_to_clap


def _load_wav(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode WAV bytes into a (1, T) float32 tensor and sample rate."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    else:
        dtype = {2: np.int16, 4: np.int32}[sampwidth]
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        samples /= float(np.iinfo(dtype).max)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(samples).unsqueeze(0), sr


def score_generation(audio_bytes: bytes, query_vector: NDArray[np.float32]) -> float:
    """Score generated audio against a CLAP text embedding.

    Returns cosine similarity in [-1, 1].
    """
    waveform, sr = _load_wav(audio_bytes)

    audio_np = resample_to_clap(waveform, orig_sr=sr)
    audio_embedding = embed_audio(audio_np)

    cos_sim = _cosine_similarity(audio_embedding, query_vector)
    return float(cos_sim)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> np.floating:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return np.float32(0.0)
    return dot / norm
