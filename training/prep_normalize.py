"""M6-T3: Audio normalize + machine metadata extraction.

Reads source mp3s listed in source/labels.json, normalizes each to
32kHz mono 16-bit PCM WAV at -14 LUFS via ffmpeg, then extracts
bpm/key/duration via librosa. Outputs:
  - prepared/<stem>.wav  (normalized audio)
  - prepared/machine_metadata.json  (bpm, key, duration per track)
"""

import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "source"
PREPARED_DIR = ROOT / "prepared"

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (starting from C)
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def detect_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)

    best_corr = -2.0
    best_key = "C major"

    for shift in range(12):
        major_rotated = np.roll(MAJOR_PROFILE, shift)
        minor_rotated = np.roll(MINOR_PROFILE, shift)

        corr_major = np.corrcoef(chroma_avg, major_rotated)[0, 1]
        corr_minor = np.corrcoef(chroma_avg, minor_rotated)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{PITCH_CLASSES[shift]} major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{PITCH_CLASSES[shift]} minor"

    return best_key


def normalize_audio(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "32000",
        "-ac",
        "1",
        "-af",
        "loudnorm=I=-14:TP=-1:LRA=11",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_metadata(wav_path: Path) -> dict:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.asarray(tempo).flat[0])))
    key = detect_key(y, sr)
    return {"bpm": bpm, "key": key, "duration": round(duration, 2)}


def main() -> None:
    PREPARED_DIR.mkdir(exist_ok=True)

    labels = json.loads((SOURCE_DIR / "labels.json").read_text())
    metadata = {}
    errors = []

    for entry in labels:
        filename = entry["filename"]
        stem = Path(filename).stem
        src = SOURCE_DIR / filename
        dst = PREPARED_DIR / f"{stem}.wav"

        print(f"[normalize] {filename}")
        try:
            normalize_audio(src, dst)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e.stderr.decode()}", file=sys.stderr)
            errors.append(filename)
            continue

        print(f"[metadata]  {stem}.wav")
        meta = extract_metadata(dst)
        metadata[stem] = meta
        print(f"  bpm={meta['bpm']}  key={meta['key']}  duration={meta['duration']}s")

    out_path = PREPARED_DIR / "machine_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote {len(metadata)} entries to {out_path}")

    if errors:
        print(f"ERRORS: {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
