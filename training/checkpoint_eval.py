"""Checkpoint evaluation — generate audio from Dora checkpoints for A/B listening.

Loads specific epoch checkpoints from the training volume, generates 30s audio
for each eval prompt, and saves WAVs to /dora/eval/{variant}/ on the volume.
Also runs vanilla musicgen-melody-large as baseline.

Usage:
    modal run training/checkpoint_eval.py::CheckpointEvaluator.evaluate_all \
        --xp-sig 48bdbba4 --epochs '[3, 6, 10]'
"""

from __future__ import annotations

import modal

from config import (
    AUDIOCRAFT_SHA,
    ENCODEC_PRETRAINED,
    MODAL_SECRET_NAME,
    MUSICGEN_BASE_MODEL,
    MUSICGEN_SAMPLE_RATE,
    TRAINING_VOLUME_MOUNT_PATH,
    TRAINING_VOLUME_NAME,
)

AUDIOCRAFT_ROOT = "/opt/audiocraft"
EVAL_PROMPTS_PATH = "/opt/eval_prompts.json"
EVAL_OUTPUT_DIR = f"{TRAINING_VOLUME_MOUNT_PATH}/eval"
DURATION_SECONDS = 30.0

app = modal.App("flowing-trails-checkpoint-eval")

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
        "soundfile",
    )
    .run_commands(
        f"git clone https://github.com/facebookresearch/audiocraft.git {AUDIOCRAFT_ROOT}",
        f"cd {AUDIOCRAFT_ROOT} && git checkout {AUDIOCRAFT_SHA}",
        f"cd {AUDIOCRAFT_ROOT} && pip install -e .",
    )
    .add_local_file(
        "eval/checkpoint_eval_prompts.json",
        EVAL_PROMPTS_PATH,
        copy=True,
    )
    .add_local_python_source("config")
)

training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME, create_if_missing=True)


@app.cls(
    gpu="a10g",
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    volumes={TRAINING_VOLUME_MOUNT_PATH: training_volume},
    timeout=3600 * 2,
)
class CheckpointEvaluator:
    @modal.enter()
    def setup(self):
        import functools
        import logging

        import torch

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
        )
        self.log = logging.getLogger("flowing-trails.checkpoint-eval")

        _original_load = torch.load
        torch.load = functools.partial(_original_load, weights_only=False)

        self.log.info("Checkpoint evaluator ready")

    @modal.method()
    def evaluate_all(
        self, xp_sig: str, epochs: str, include_vanilla: bool = False
    ) -> dict:
        """Generate audio for all prompts across checkpoints (and optionally vanilla).

        Args:
            xp_sig: Dora experiment signature (e.g. '48bdbba4').
            epochs: Comma-separated epoch numbers (e.g. '3,6,10').
            include_vanilla: Also generate from vanilla musicgen-melody-large.
        """
        import json

        epoch_list = [int(e.strip()) for e in epochs.split(",")]

        with open(EVAL_PROMPTS_PATH) as f:
            prompts = json.load(f)

        self.log.info(
            "Evaluating %d epochs%s across %d prompts",
            len(epoch_list),
            " + vanilla" if include_vanilla else "",
            len(prompts),
        )

        results = {}

        if include_vanilla:
            results["vanilla"] = self._evaluate_vanilla(prompts)

        for epoch in epoch_list:
            epoch_result = self._evaluate_epoch(xp_sig, epoch, prompts)
            results[f"epoch_{epoch}"] = epoch_result

        training_volume.commit()
        self.log.info("All evaluations complete. Results saved to %s", EVAL_OUTPUT_DIR)
        return results

    def _evaluate_vanilla(self, prompts: list[dict]) -> dict:
        """Generate from vanilla musicgen-melody-large."""
        from audiocraft.models import MusicGen  # type: ignore

        self.log.info("Loading vanilla model: %s", MUSICGEN_BASE_MODEL)
        model = MusicGen.get_pretrained(MUSICGEN_BASE_MODEL)
        model.set_generation_params(duration=DURATION_SECONDS)

        result = self._generate_all(model, prompts, "vanilla")

        del model
        self._free_gpu()
        return result

    def _evaluate_epoch(self, xp_sig: str, epoch: int, prompts: list[dict]) -> dict:
        """Generate from a specific epoch checkpoint."""
        import shutil
        import tempfile
        from pathlib import Path

        from audiocraft.models import MusicGen  # type: ignore
        from audiocraft.utils.export import export_lm, export_pretrained_compression_model  # type: ignore

        ckpt_path = (
            Path(TRAINING_VOLUME_MOUNT_PATH) / "xps" / xp_sig / f"checkpoint_{epoch}.th"
        )
        if not ckpt_path.is_file():
            self.log.error("Checkpoint not found: %s", ckpt_path)
            return {"error": f"Checkpoint not found: {ckpt_path}"}

        self.log.info("Exporting checkpoint to loadable format: %s", ckpt_path)
        stage_dir = Path(tempfile.mkdtemp(prefix="ckpt-eval-"))
        export_lm(ckpt_path, stage_dir / "state_dict.bin")
        export_pretrained_compression_model(
            ENCODEC_PRETRAINED, stage_dir / "compression_state_dict.bin"
        )

        self.log.info("Loading exported model (epoch %d)", epoch)
        model = MusicGen.get_pretrained(str(stage_dir))
        model.set_generation_params(duration=DURATION_SECONDS)

        variant = f"epoch_{epoch}"
        result = self._generate_all(model, prompts, variant)

        del model
        self._free_gpu()
        shutil.rmtree(stage_dir, ignore_errors=True)
        return result

    def _generate_all(self, model, prompts: list[dict], variant: str) -> dict:
        """Generate audio for all prompts and save to volume."""
        import io
        from pathlib import Path

        import soundfile as sf
        import torch

        out_dir = Path(EVAL_OUTPUT_DIR) / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for i, prompt in enumerate(prompts, 1):
            prompt_id = prompt["id"]
            description = prompt["description"]
            wav_path = out_dir / f"{prompt_id}.wav"

            self.log.info(
                "[%s] [%d/%d] %s: %s",
                variant,
                i,
                len(prompts),
                prompt_id,
                description,
            )

            with torch.no_grad():
                wav = model.generate([description])

            audio_np = wav[0].cpu().numpy().T
            sf.write(str(wav_path), audio_np, samplerate=MUSICGEN_SAMPLE_RATE)

            generated.append(prompt_id)
            self.log.info("  Saved: %s", wav_path)

        training_volume.commit()
        self.log.info("[%s] Generated %d tracks", variant, len(generated))
        return {"variant": variant, "count": len(generated), "prompts": generated}

    def _free_gpu(self):
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
