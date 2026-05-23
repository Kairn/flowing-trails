"""Training runner — Modal GPU app for audiocraft fine-tuning.

Image: CUDA 12.1 + Python 3.10 + torch 2.1.0 + audiocraft v1.3.0 (pinned SHA).
Volume: persistent Dora XP directory for checkpoint survival across preemptions.
"""

from __future__ import annotations

import modal

from config import (
    MODAL_SECRET_NAME,
    TRAINING_APP_NAME,
    TRAINING_DATA_PATH,
    TRAINING_GPU_CONFIG,
    TRAINING_VOLUME_MOUNT_PATH,
    TRAINING_VOLUME_NAME,
)

AUDIOCRAFT_SHA = "72cb16f9fb239e9cf03f7bd997198c7d7a67a01c"
AUDIOCRAFT_ROOT = "/opt/audiocraft"

app = modal.App(TRAINING_APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "clang",
        "git",
        "pkg-config",
        "ffmpeg",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libsndfile1-dev",
    )
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "numpy==1.26.4",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers<0.0.23",
        "huggingface_hub>=0.20",
        "python-dotenv>=1.0",
        "soundfile",
    )
    .run_commands(
        f"git clone https://github.com/facebookresearch/audiocraft.git {AUDIOCRAFT_ROOT}",
        f"cd {AUDIOCRAFT_ROOT} && git checkout {AUDIOCRAFT_SHA}",
        f"cd {AUDIOCRAFT_ROOT} && pip install -e .",
    )
    .env({"AUDIOCRAFT_DORA_DIR": TRAINING_VOLUME_MOUNT_PATH})
    .add_local_python_source("config")
)

training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME, create_if_missing=True)


@app.cls(
    gpu=TRAINING_GPU_CONFIG,
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    volumes={TRAINING_VOLUME_MOUNT_PATH: training_volume},
    timeout=3600 * 6,
)
class TrainingRunner:
    @modal.enter()
    def setup(self):
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
        )
        self.log = logging.getLogger("flowing-trails.training")
        self.log.info("Training runner ready")

    @modal.method()
    def check_env(self) -> dict:
        """Verify GPU, torch, audiocraft, and volume are accessible."""
        import os

        import torch

        has_audiocraft = os.path.isdir(AUDIOCRAFT_ROOT)
        has_data_dir = os.path.isdir(TRAINING_DATA_PATH)
        dora_dir = os.environ.get("AUDIOCRAFT_DORA_DIR", "")

        info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "torch_version": torch.__version__,
            "audiocraft_root": AUDIOCRAFT_ROOT,
            "audiocraft_present": has_audiocraft,
            "dora_dir": dora_dir,
            "data_dir_exists": has_data_dir,
        }
        self.log.info("Environment: %s", info)
        return info
