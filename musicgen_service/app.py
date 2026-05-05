"""MusicGen inference service — Modal GPU endpoint.

Loads facebook/musicgen-melody (or fine-tuned variant via MODEL_TAG env var)
with Multi-Band Diffusion decoder. Accepts a text prompt and returns WAV bytes.
"""

from __future__ import annotations

import modal

from config import (
    GPU_CONFIG,
    HF_MUSICGEN_REPO,
    MODAL_SECRET_NAME,
    MUSICGEN_APP_NAME,
    MUSICGEN_BASE_MODEL,
    MUSICGEN_SAMPLE_RATE,
)

app = modal.App(MUSICGEN_APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "pkg-config",
        "ffmpeg",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libavfilter-dev",
    )
    .pip_install(
        "numpy<2",
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers>=4.40.0",
        "audiocraft",
        "huggingface_hub",
        "soundfile",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
    )
    .add_local_python_source("config")
    .add_local_python_source("otel_utils")
)


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=300,
    container_idle_timeout=120,
)
class MusicGenService:
    @modal.enter()
    def load_model(self):
        import os
        import logging

        import torch
        from audiocraft.models import MusicGen, MultiBandDiffusion  # type: ignore

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
        )
        self.log = logging.getLogger("flowing-trails.musicgen")

        model_tag = os.environ.get("MODEL_TAG", "")
        if model_tag:
            # Fine-tuned model: download by tag from our HF repo, load locally.
            # HF_MUSICGEN_REPO env var must be the full namespace/repo path
            # (e.g. "username/flowing-trails-musicgen").
            from huggingface_hub import snapshot_download

            hf_repo = os.environ.get("HF_MUSICGEN_REPO", HF_MUSICGEN_REPO)
            model_id = f"{hf_repo}@{model_tag}"
            local_dir = snapshot_download(repo_id=hf_repo, revision=model_tag)
            self.log.info("Loading fine-tuned model: %s", model_id)
            self.model = MusicGen.get_pretrained(local_dir)
        else:
            model_id = MUSICGEN_BASE_MODEL
            self.log.info("Loading base model: %s", model_id)
            self.model = MusicGen.get_pretrained(model_id)

        self.log.info("Loading MultiBandDiffusion decoder")
        self.mbd = MultiBandDiffusion.get_mbd_musicgen()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.log.info("Models loaded on %s", self.device)

    @modal.method()
    def generate(
        self,
        prompt: str,
        duration_seconds: float = 10.0,
        melody_wav: bytes | None = None,
        melody_sample_rate: int | None = None,
        seed: int | None = None,
    ) -> dict:
        import io
        import time

        import soundfile as sf
        import torch

        self.log.info("Generating: %.60s (%.1fs)", prompt, duration_seconds)
        t0 = time.monotonic()

        self.model.set_generation_params(duration=duration_seconds)

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            if melody_wav is not None and melody_sample_rate is not None:
                melody_tensor, sr = _load_wav_bytes(melody_wav, melody_sample_rate)
                melody_tensor = melody_tensor.to(self.device)
                wav = self.model.generate_with_chroma([prompt], melody_tensor, sr)
            else:
                wav = self.model.generate([prompt])

            # Re-encode through compression model to get tokens for MBD
            tokens, scale = self.model.compression_model.encode(wav)
            wav_mbd = self.mbd.tokens_to_wav(tokens)

        # Encode as WAV bytes
        audio_np = wav_mbd[0].cpu().numpy().T  # [samples, channels]
        buf = io.BytesIO()
        sf.write(
            buf,
            audio_np,
            samplerate=MUSICGEN_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        audio_bytes = buf.getvalue()

        latency_ms = (time.monotonic() - t0) * 1000
        self.log.info(
            "Generated %.1fs audio in %.0fms (MBD decoder)",
            duration_seconds,
            latency_ms,
        )

        return {
            "audio_bytes": audio_bytes,
            "sample_rate": MUSICGEN_SAMPLE_RATE,
            "model": self.model_id,
            "decoder": "mbd",
            "duration_seconds": duration_seconds,
            "latency_ms": round(latency_ms, 1),
        }


def _load_wav_bytes(wav_bytes: bytes, target_sr: int):
    """Decode WAV bytes into a torch tensor [1, channels, samples]."""
    import io

    import soundfile as sf
    import torch

    buf = io.BytesIO(wav_bytes)
    audio_np, sr = sf.read(buf)
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)
    tensor = torch.from_numpy(audio_np.T).unsqueeze(0).float()  # [1, C, T]
    return tensor, sr
