"""
Compute CLAP audio embeddings for the entire corpus.

Reads WAV files from the Modal Volume, resamples to 48 kHz, computes
512-dim CLAP embeddings, and writes embeddings + content hashes to
corpus_embeddings.json on the volume.

Usage:
    modal run retrieval/embed_corpus.py
"""

import hashlib
import json
from pathlib import Path

import modal

from config import (
    CLAP_SAMPLE_RATE,
    CORPUS_EMBEDDINGS_PATH,
    CORPUS_MANIFEST_PATH,
    MODAL_SECRET_NAME,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
)

app = modal.App("flowing-trails-embed-corpus")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "numpy<2",
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "torchvision>=0.19.0",
        "laion_clap",
        "soundfile",
    )
    .add_local_python_source("config")
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=1800,
)
def embed_corpus() -> dict:
    import logging

    import soundfile as sf
    import torch
    import torchaudio
    from laion_clap import CLAP_Module

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )
    log = logging.getLogger("flowing-trails.embed-corpus")

    with open(CORPUS_MANIFEST_PATH) as f:
        manifest = json.load(f)
    log.info("Loaded manifest: %d entries", len(manifest))

    log.info("Loading CLAP model (htsat-unfused, CPU)")
    model = CLAP_Module(enable_fusion=False, device="cpu")
    model.load_ckpt()
    log.info("CLAP model loaded")

    embeddings = []
    failed = []

    for i, item in enumerate(manifest, 1):
        clip_id = item["id"]
        audio_path = f"{VOLUME_MOUNT_PATH}/{item['corpus_file_path']}"
        try:
            with open(audio_path, "rb") as f:
                raw_bytes = f.read()
            content_hash = hashlib.sha256(raw_bytes).hexdigest()
            point_id = int(content_hash[:16], 16)

            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio)
            if sr != CLAP_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sr, CLAP_SAMPLE_RATE
                )

            embedding = model.get_audio_embedding_from_data([waveform.numpy()])

            embeddings.append(
                {
                    "id": clip_id,
                    "corpus_file_path": item["corpus_file_path"],
                    "content_hash": content_hash,
                    "point_id": point_id,
                    "embedding": embedding[0].tolist(),
                    "category": item.get("category"),
                    "category_label": item.get("category_label"),
                    "mood_tags": item.get("mood_tags"),
                    "energy": item.get("energy"),
                    "instrumentation": item.get("instrumentation"),
                    "bpm_hint": item.get("bpm_hint"),
                    "prompt": item.get("prompt"),
                }
            )
            log.info("Embedded %s (%d/%d)", clip_id, i, len(manifest))

        except Exception as e:
            log.error("Failed %s: %s", clip_id, e)
            failed.append({"id": clip_id, "error": str(e)})

    with open(CORPUS_EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f)
    volume.commit()

    log.info("Done: %d embedded, %d failed", len(embeddings), len(failed))

    result = {"embedded": len(embeddings), "failed": len(failed)}
    if failed:
        result["failed_ids"] = [f["id"] for f in failed]
    return result


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from otel_utils import get_logger, setup_logging

    setup_logging()
    log = get_logger("embed-corpus")

    log.info("Starting corpus embedding job")
    result = embed_corpus.remote()
    log.info(
        "Embedding complete",
        embedded=result["embedded"],
        failed=result["failed"],
    )
    if result.get("failed_ids"):
        log.warning("Failed clip IDs", ids=result["failed_ids"])
