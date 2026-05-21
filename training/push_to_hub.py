"""Push MusicGen weights to HF Hub with a git tag.

Two source modes:
  --from-hf REPO    Download state_dict.bin + compression_state_dict.bin from
                    an existing HF repo and re-upload to our repo under TAG.
  --from-checkpoint PATH
                    Export a Dora training checkpoint via audiocraft's
                    export_lm, create the compression pointer, and upload.

Files are staged in a temporary directory and cleaned up after upload.

Usage:
  python training/push_to_hub.py --tag base-melody-large-roundtrip \
    --from-hf facebook/musicgen-melody-large

  python training/push_to_hub.py --tag vgm-melody-v1 \
    --from-checkpoint /path/to/checkpoint.th
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import torch
from huggingface_hub import HfApi, hf_hub_download

from config import HF_MUSICGEN_REPO

LM_FILENAME = "state_dict.bin"
COMPRESSION_FILENAME = "compression_state_dict.bin"
ENCODEC_PRETRAINED = "facebook/encodec_32khz"


def stage_from_hf(source_repo: str, stage_dir: Path) -> None:
    """Download exported weights from an existing HF repo into stage_dir."""
    for filename in (LM_FILENAME, COMPRESSION_FILENAME):
        print(f"Downloading {source_repo}/{filename} ...")
        path = hf_hub_download(repo_id=source_repo, filename=filename)
        shutil.copy2(path, stage_dir / filename)
        print(f"  staged → {stage_dir / filename}")


def stage_from_checkpoint(checkpoint_path: str, stage_dir: Path) -> None:
    """Export a Dora training checkpoint into the HF-ready format."""
    from audiocraft.utils.export import export_lm, export_pretrained_compression_model  # type: ignore

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        sys.exit(f"Checkpoint not found: {ckpt}")

    print(f"Exporting LM from {ckpt} ...")
    export_lm(ckpt, stage_dir / LM_FILENAME)
    print(f"  staged → {stage_dir / LM_FILENAME}")

    print(f"Creating compression pointer ({ENCODEC_PRETRAINED}) ...")
    export_pretrained_compression_model(
        ENCODEC_PRETRAINED, stage_dir / COMPRESSION_FILENAME
    )
    print(f"  staged → {stage_dir / COMPRESSION_FILENAME}")


def upload(stage_dir: Path, tag: str, dest_repo: str) -> None:
    """Upload staged files to HF Hub and create a git tag."""
    api = HfApi()

    api.create_repo(repo_id=dest_repo, private=True, exist_ok=True)

    print(f"Uploading to {dest_repo} ...")
    commit = api.upload_folder(
        repo_id=dest_repo,
        folder_path=str(stage_dir),
        commit_message=f"Push weights: {tag}",
    )
    print(f"  commit: {commit.commit_url}")

    print(f"Creating tag '{tag}' ...")
    api.create_tag(
        repo_id=dest_repo,
        tag=tag,
        tag_message=f"Weights snapshot: {tag}",
        exist_ok=True,
    )
    print(f"  tagged: {dest_repo}@{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push MusicGen weights to HF Hub")
    parser.add_argument("--tag", required=True, help="Git tag to create on the HF repo")
    parser.add_argument(
        "--dest-repo",
        default=HF_MUSICGEN_REPO,
        help=f"Target HF repo (default: {HF_MUSICGEN_REPO})",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-hf", metavar="REPO", help="Mirror weights from this HF repo"
    )
    source.add_argument(
        "--from-checkpoint",
        metavar="PATH",
        help="Export from a Dora training checkpoint",
    )

    args = parser.parse_args()

    stage_dir = Path(tempfile.mkdtemp(prefix="ft-hub-"))
    try:
        if args.from_hf:
            stage_from_hf(args.from_hf, stage_dir)
        else:
            stage_from_checkpoint(args.from_checkpoint, stage_dir)

        for f in (LM_FILENAME, COMPRESSION_FILENAME):
            if not (stage_dir / f).exists():
                sys.exit(f"Staging failed: {f} not found in {stage_dir}")

        upload(stage_dir, args.tag, args.dest_repo)
        print("Done.")
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        print(f"Cleaned up staging dir: {stage_dir}")


if __name__ == "__main__":
    main()
