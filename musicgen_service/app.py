"""MusicGen inference service — Modal GPU endpoint.

Loads facebook/musicgen-melody-large (or fine-tuned variant via MODEL_TAG env var)
with Multi-Band Diffusion decoder. Accepts a text prompt and returns WAV bytes.
"""

from __future__ import annotations

from contextlib import contextmanager

import modal

from config import (
    AUDIOCRAFT_SHA,
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
        "git",
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
    )
    .run_commands(
        f"git clone https://github.com/facebookresearch/audiocraft.git /tmp/audiocraft"
        f" && cd /tmp/audiocraft && git checkout {AUDIOCRAFT_SHA}",
        "cd /tmp/audiocraft && sed -i"
        " -e 's/torch==2.1.0/torch>=2.1.0/'"
        " -e 's/torchaudio>=2.0.0,<2.1.2/torchaudio>=2.0.0/'"
        " -e 's/xformers<0.0.23/xformers/'"
        " -e '/torchvision/d'"
        " -e '/torchtext/d'"
        " -e '/gradio/d'"
        " requirements.txt"
        " && pip install .",
    )
    .pip_install(
        "transformers>=4.40.0",
        "huggingface_hub",
        "soundfile",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "python-dotenv",
        "structlog",
    )
    .add_local_python_source("config")
    .add_local_python_source("otel_utils")
)


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=300,
    scaledown_window=120,
    retries=modal.Retries(max_retries=1),
)
class MusicGenService:
    @modal.enter()
    def load_model(self):
        import os
        import logging

        import functools
        import torch

        # audiocraft checkpoints use omegaconf globals that torch 2.6+ rejects
        # under weights_only=True — patch for Meta's own trusted checkpoints.
        _original_load = torch.load
        torch.load = functools.partial(_original_load, weights_only=False)
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

        torch.load = _original_load
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
        cfg_coeff: float | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> dict:
        import io
        import time

        import soundfile as sf
        import torch

        from otel_utils import restored_context, setup_tracing

        setup_tracing()
        ctx_mgr = restored_context(trace_context) if trace_context else _noop_context()

        self.log.info("Generating: %.60s (%.1fs)", prompt, duration_seconds)
        t0 = time.monotonic()

        with ctx_mgr:
            gen_params = {"duration": duration_seconds}
            if cfg_coeff is not None:
                gen_params["cfg_coeff"] = cfg_coeff
            if top_k is not None:
                gen_params["top_k"] = top_k
            if temperature is not None:
                gen_params["temperature"] = temperature
            self.model.set_generation_params(**gen_params)

            if seed is not None:
                torch.manual_seed(seed)

            with torch.no_grad():
                if melody_wav is not None and melody_sample_rate is not None:
                    melody_tensor, sr = _load_wav_bytes(melody_wav)
                    melody_tensor = melody_tensor.to(self.device)
                    wav = self.model.generate_with_chroma([prompt], melody_tensor, sr)
                else:
                    wav = self.model.generate([prompt])

                # Re-encode through compression model to get tokens for MBD
                encoded = self.model.compression_model.encode(wav)
                entry = encoded[0]
                codes = entry[0] if isinstance(entry, (tuple, list)) else entry
                wav_mbd = self.mbd.tokens_to_wav(codes)

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


@contextmanager
def _noop_context():
    yield


def _load_wav_bytes(wav_bytes: bytes):
    """Decode WAV bytes into a torch tensor [1, channels, samples] and sample rate."""
    import io

    import soundfile as sf
    import torch

    buf = io.BytesIO(wav_bytes)
    audio_np, sr = sf.read(buf)
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)
    tensor = torch.from_numpy(audio_np.T).unsqueeze(0).float()  # [1, C, T]
    return tensor, sr
