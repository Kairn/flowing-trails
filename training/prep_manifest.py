"""M6-T5: Build audiocraft manifest from prepared WAVs.

Scans training/prepared/ for WAV files, writes data.jsonl.gz manifest
(AudioDataset.from_meta expects this file). Sampling is duration-proportional
by default via audiocraft's sample_on_duration; no extra weight needed.
"""

import sys
from pathlib import Path

from audiocraft.data.audio_dataset import find_audio_files, save_audio_meta  # type: ignore

ROOT = Path(__file__).parent
PREPARED_DIR = ROOT / "prepared"
MANIFEST_PATH = PREPARED_DIR / "data.jsonl.gz"


def main() -> None:
    if not PREPARED_DIR.exists():
        print(f"ERROR: {PREPARED_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    meta = find_audio_files(
        PREPARED_DIR, [".wav"], progress=True, resolve=True, minimal=True, workers=1
    )
    if not meta:
        print("ERROR: no WAV files found in prepared/", file=sys.stderr)
        sys.exit(1)

    for m in meta:
        m.weight = None

    save_audio_meta(MANIFEST_PATH, meta)
    print(f"\nWrote manifest: {MANIFEST_PATH} ({len(meta)} entries)")


if __name__ == "__main__":
    main()
