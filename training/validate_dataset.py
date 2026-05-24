"""M6-T5: Validate manifest loads under audiocraft MusicDataset.

Pass condition: manifest loads, every entry has required sidecar fields,
MusicDataset.__getitem__ iterates without error for all entries.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MUSICGEN_SAMPLE_RATE

ROOT = Path(__file__).parent
os.environ.setdefault("AUDIOCRAFT_CONFIG", str(ROOT / "audiocraft_env.yaml"))

from audiocraft.data.audio_dataset import load_audio_meta  # type: ignore
from audiocraft.data.music_dataset import MusicDataset  # type: ignore

PREPARED_DIR = ROOT / "prepared"
MANIFEST_PATH = PREPARED_DIR / "data.jsonl.gz"

REQUIRED_SIDECAR_FIELDS = [
    "description",
    "genre",
    "bpm",
    "key",
    "moods",
    "instrument",
]


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        print("Run: make train-manifest", file=sys.stderr)
        sys.exit(1)

    meta = load_audio_meta(MANIFEST_PATH)
    print(f"Loaded manifest: {len(meta)} entries")

    for m in meta:
        sidecar = Path(m.path).with_suffix(".json")
        if not sidecar.exists():
            print(f"  FAIL: missing sidecar for {m.path}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(sidecar.read_text())
        missing = [f for f in REQUIRED_SIDECAR_FIELDS if f not in data]
        if missing:
            print(f"  FAIL: {sidecar.name} missing fields: {missing}", file=sys.stderr)
            sys.exit(1)
        print(f"  OK: {Path(m.path).name} — sidecar valid")

    ds = MusicDataset(
        meta,
        segment_duration=None,
        sample_rate=MUSICGEN_SAMPLE_RATE,
        channels=1,
        shuffle=False,
        num_samples=len(meta),
        info_fields_required=False,
    )
    print(f"\nMusicDataset created: {len(ds)} items")

    for i in range(len(meta)):
        wav, info = ds[i]
        print(
            f"  [{i}] {Path(info.meta.path).name} — "
            f'wav={list(wav.shape)}, desc="{info.description}"'
        )

    print(f"\nPASS: all {len(meta)} entries loaded successfully")


if __name__ == "__main__":
    main()
