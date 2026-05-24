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
CONFIGS_DIR = "/opt/training_configs"

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
        "transformers<4.46",
        "python-dotenv>=1.0",
        "soundfile",
    )
    .run_commands(
        f"git clone https://github.com/facebookresearch/audiocraft.git {AUDIOCRAFT_ROOT}",
        f"cd {AUDIOCRAFT_ROOT} && git checkout {AUDIOCRAFT_SHA}",
        f"cd {AUDIOCRAFT_ROOT} && pip install -e .",
        f"mkdir -p {AUDIOCRAFT_ROOT}/config/dset {AUDIOCRAFT_ROOT}/config/teams {CONFIGS_DIR}",
    )
    .env(
        {
            "AUDIOCRAFT_DORA_DIR": TRAINING_VOLUME_MOUNT_PATH,
            "AUDIOCRAFT_CONFIG": f"{AUDIOCRAFT_ROOT}/config/teams/default.yaml",
            "AUDIOCRAFT_CLUSTER": "default",
            "USER": "training",
        }
    )
    .add_local_file(
        "training/configs/dset_vgm.yaml",
        f"{AUDIOCRAFT_ROOT}/config/dset/vgm.yaml",
        copy=True,
    )
    .add_local_file(
        "training/configs/teams_default.yaml",
        f"{AUDIOCRAFT_ROOT}/config/teams/default.yaml",
        copy=True,
    )
    .add_local_file(
        "training/configs/poc_small.yaml",
        f"{CONFIGS_DIR}/poc_small.yaml",
        copy=True,
    )
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
    def train(self, config_name: str) -> dict:
        """Run training via Dora with the specified config."""
        import subprocess

        import yaml

        self._rebuild_manifest()

        config_path = f"{CONFIGS_DIR}/{config_name}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        cmd = _build_dora_cmd(config)
        self.log.info("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, cwd=AUDIOCRAFT_ROOT)

        training_volume.commit()
        self.log.info("Volume committed. Return code: %d", result.returncode)

        return {"returncode": result.returncode, "config": config_name}

    def _rebuild_manifest(self):
        """Rebuild manifest from volume data so paths are container-valid."""
        from pathlib import Path

        from audiocraft.data.audio_dataset import find_audio_files, save_audio_meta  # type: ignore

        data_dir = Path(TRAINING_DATA_PATH)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Training data not found at {data_dir}. "
                "Upload first: make train-upload"
            )

        wavs = find_audio_files(
            data_dir, [".wav"], progress=True, resolve=True, minimal=True, workers=1
        )
        if not wavs:
            raise FileNotFoundError(f"No WAV files found in {data_dir}")

        for m in wavs:
            m.weight = 1.0

        manifest_path = data_dir / "data.jsonl.gz"
        save_audio_meta(manifest_path, wavs)
        training_volume.commit()
        self.log.info("Manifest rebuilt: %s (%d entries)", manifest_path, len(wavs))

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


def _build_dora_cmd(config: dict) -> list[str]:
    """Convert training config YAML to dora run CLI arguments."""
    cmd = ["dora", "run"]
    for key, value in config.items():
        if isinstance(value, dict):
            cmd.extend(_flatten_overrides(value, key))
        else:
            cmd.append(f"{key}={_format_value(value)}")
    return cmd


def _flatten_overrides(d: dict, prefix: str) -> list[str]:
    """Flatten nested dict to Hydra dot-separated overrides."""
    items = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            items.extend(_flatten_overrides(value, full_key))
        else:
            items.append(f"{full_key}={_format_value(value)}")
    return items


def _format_value(value: object) -> str:
    """Format a Python value for Hydra CLI (lowercase booleans)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
