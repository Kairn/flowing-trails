"""M6-T4: Chroma stability scoring + template descriptions + sidecar JSON.

Reads prepared WAVs + labels + machine metadata. For each track:
  - Computes chroma stability score via audiocraft ChromaExtractor
  - Generates template description from human labels
  - Writes sidecar JSON at prepared/<stem>.wav.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MUSICGEN_SAMPLE_RATE

import torch
import torchaudio
from audiocraft.modules.chroma import ChromaExtractor  # type: ignore

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "source"
PREPARED_DIR = ROOT / "prepared"

ENERGY_ADJ = {
    "high": "High-energy",
    "medium": "Mid-energy",
    "low": "Low-energy",
}


def compute_chroma_score(extractor: ChromaExtractor, wav_path: Path) -> float:
    """Fraction of adjacent frames sharing the same dominant chroma bin."""
    wav, sr = torchaudio.load(wav_path)
    assert sr == MUSICGEN_SAMPLE_RATE
    with torch.no_grad():
        chroma = extractor(wav.unsqueeze(0))  # (1, T, 12)
    indices = chroma.argmax(dim=-1).squeeze(0)  # (T,)
    if indices.shape[0] < 2:
        return 1.0
    stable = (indices[1:] == indices[:-1]).float().mean().item()
    return round(stable, 4)


def join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_description(label: dict) -> str:
    energy_adj = ENERGY_ADJ[label["energy"]]
    moods = join_list(label["mood_tags"])
    return f"{energy_adj} JRPG {label['scene_type']} theme, {moods}"


def build_sidecar(label: dict, meta: dict, chroma_score: float) -> dict:
    return {
        "description": build_description(label),
        "genre": label["genre"],
        "bpm": str(meta["bpm"]),
        "key": meta["key"],
        "moods": label["mood_tags"],
        "instrument": join_list(label["dominant_instruments"]),
        "keywords": label["notes"] or "",
        "duration": meta["duration"],
        "sample_rate": MUSICGEN_SAMPLE_RATE,
        "chroma_score": chroma_score,
    }


def main() -> None:
    labels = json.loads((SOURCE_DIR / "labels.json").read_text())
    metadata = json.loads((PREPARED_DIR / "machine_metadata.json").read_text())
    extractor = ChromaExtractor(sample_rate=MUSICGEN_SAMPLE_RATE, argmax=True)

    for entry in labels:
        stem = Path(entry["filename"]).stem
        wav_path = PREPARED_DIR / f"{stem}.wav"

        print(f"[chroma]    {stem}.wav")
        score = compute_chroma_score(extractor, wav_path)
        print(f"  chroma_score={score}")

        meta = metadata[stem]
        sidecar = build_sidecar(entry, meta, score)

        sidecar_path = PREPARED_DIR / f"{stem}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"  description: {sidecar['description']}")

    print(f"\nWrote {len(labels)} sidecar JSONs")


if __name__ == "__main__":
    main()
