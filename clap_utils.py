"""CLAP embedding utilities for text and audio.

Wraps laion-clap to provide simple embed_text / embed_audio functions
and a resample helper for MusicGen → CLAP sample-rate conversion.
"""

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from laion_clap import CLAP_Module
from numpy.typing import NDArray

from config import CLAP_SAMPLE_RATE, MUSICGEN_SAMPLE_RATE

load_dotenv()

_model: CLAP_Module | None = None


def load_clap_model() -> CLAP_Module:
    """Load CLAP htsat-unfused on CPU (cached after first call)."""
    global _model
    if _model is not None:
        return _model
    _model = CLAP_Module(enable_fusion=False, device="cpu")
    _model.load_ckpt()
    return _model


def embed_text(text: str) -> NDArray[np.float32]:
    """Embed a single text string → (512,) float32 vector."""
    model = load_clap_model()
    embedding = model.get_text_embedding([text])
    return embedding[0]


def embed_audio(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    """Embed a mono audio waveform (48 kHz expected) → (512,) float32 vector.

    The waveform should already be resampled to 48 kHz. Use resample_to_clap()
    to convert from MusicGen's 32 kHz output before calling this.
    """
    model = load_clap_model()
    embedding = model.get_audio_embedding_from_data([waveform])
    return embedding[0]


def resample_to_clap(
    waveform: torch.Tensor,
    orig_sr: int = MUSICGEN_SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Resample a waveform from orig_sr to CLAP's 48 kHz and return numpy array.

    Parameters
    ----------
    waveform : torch.Tensor
        1-D or (1, T) mono waveform tensor.
    orig_sr : int
        Source sample rate (default: MusicGen's 32 kHz).
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, CLAP_SAMPLE_RATE)
    return resampled.numpy()
