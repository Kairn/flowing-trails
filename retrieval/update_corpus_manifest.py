"""
Update corpus_manifest.json on the Modal Volume with corrected metadata
from corpus_prompts.json — without regenerating audio.

Reads the existing manifest to confirm which clips exist, then rewrites
it with current metadata from corpus_prompts.json.

Usage:
    modal run retrieval/update_corpus_manifest.py
"""

import json
from pathlib import Path

import modal

from config import MODAL_SECRET_NAME, VOLUME_MOUNT_PATH, VOLUME_NAME

app = modal.App("flowing-trails-manifest-update")

volume = modal.Volume.from_name(VOLUME_NAME)

image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("config")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=60,
)
def update_manifest(prompts: list[dict]) -> dict:
    import os

    manifest_path = f"{VOLUME_MOUNT_PATH}/corpus_manifest.json"

    # Read existing manifest to get the set of actually-generated clips
    existing = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for entry in json.load(f):
                existing[entry["id"]] = entry["corpus_file_path"]

    # Rebuild manifest: only include clips that exist on the volume
    updated = []
    skipped = []
    for item in prompts:
        clip_id = item["id"]
        if clip_id in existing:
            updated.append({**item, "corpus_file_path": existing[clip_id]})
        else:
            skipped.append(clip_id)

    with open(manifest_path, "w") as f:
        json.dump(updated, f, indent=2)

    volume.commit()

    return {"updated": len(updated), "skipped": len(skipped), "skipped_ids": skipped}


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from otel_utils import get_logger, setup_logging

    setup_logging()
    log = get_logger("manifest-update")

    prompts_path = Path(__file__).parent / "corpus_prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"{prompts_path} not found. "
            "Run `python retrieval/generate_corpus_prompts.py` first."
        )

    with open(prompts_path) as f:
        prompts = json.load(f)

    log.info("Updating manifest metadata", prompt_count=len(prompts))
    result = update_manifest.remote(prompts)
    log.info(
        "Manifest updated",
        updated=result["updated"],
        skipped=result["skipped"],
    )
    if result["skipped_ids"]:
        log.warning("Clips in prompts but not on volume", ids=result["skipped_ids"])
