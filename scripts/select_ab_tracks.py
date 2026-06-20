"""Select top chroma-stable training tracks for the retrieval A/B test.

Reads sidecar JSONs from training/prepared/, selects top 50 by chroma score
with proportional category representation, and outputs:
  - eval/ab_corpus_manifest.json (for embed_corpus/index_corpus pipeline)
  - eval/ab_tracks/ directory with symlinks to selected WAVs

Usage:
    python scripts/select_ab_tracks.py [--count 50]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

PREPARED_DIR = Path(__file__).resolve().parent.parent / "training" / "prepared"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
MANIFEST_OUT = EVAL_DIR / "ab_corpus_manifest.json"
TRACKS_OUT = EVAL_DIR / "ab_tracks"

MIN_CHROMA = 0.80

CATEGORY_RE = re.compile(r"JRPG (\w+) theme")


def extract_category(description: str) -> str:
    m = CATEGORY_RE.search(description)
    return m.group(1) if m else "other"


def extract_energy(description: str) -> str:
    desc_lower = description.lower()
    if desc_lower.startswith("high"):
        return "high"
    if desc_lower.startswith("low"):
        return "low"
    return "medium"


def load_candidates() -> list[dict]:
    candidates = []
    for sidecar in sorted(PREPARED_DIR.glob("track_*.json")):
        data = json.loads(sidecar.read_text())
        chroma = data.get("chroma_score", 0)
        if chroma < MIN_CHROMA:
            continue

        stem = sidecar.stem
        wav_path = PREPARED_DIR / f"{stem}.wav"
        if not wav_path.exists():
            continue

        desc = data.get("description", "")
        candidates.append(
            {
                "stem": stem,
                "wav_path": str(wav_path),
                "chroma_score": chroma,
                "category": extract_category(desc),
                "energy": extract_energy(desc),
                "description": desc,
                "moods": data.get("moods", []),
                "instrument": data.get("instrument", ""),
                "bpm": data.get("bpm"),
                "key": data.get("key"),
                "duration": data.get("duration"),
            }
        )
    return candidates


def select_tracks(candidates: list[dict], count: int) -> list[dict]:
    """Select top tracks with proportional category representation."""
    from collections import Counter

    cat_counts = Counter(c["category"] for c in candidates)
    total_eligible = len(candidates)

    cat_quotas: dict[str, int] = {}
    allocated = 0
    for cat, n in cat_counts.most_common():
        quota = max(1, round(count * n / total_eligible))
        cat_quotas[cat] = quota
        allocated += quota

    while allocated > count:
        biggest = max(cat_quotas, key=lambda c: cat_quotas[c])
        cat_quotas[biggest] -= 1
        allocated -= 1

    while allocated < count:
        biggest_pool = max(cat_counts, key=lambda c: cat_counts[c])
        cat_quotas[biggest_pool] += 1
        allocated += 1

    by_cat: dict[str, list[dict]] = {}
    for c in candidates:
        by_cat.setdefault(c["category"], []).append(c)
    for cat_list in by_cat.values():
        cat_list.sort(key=lambda x: x["chroma_score"], reverse=True)

    selected = []
    for cat, quota in cat_quotas.items():
        selected.extend(by_cat[cat][:quota])

    selected.sort(key=lambda x: x["chroma_score"], reverse=True)
    return selected


def build_manifest(selected: list[dict]) -> list[dict]:
    """Build corpus manifest compatible with embed_corpus.py / index_corpus.py."""
    manifest = []
    for i, track in enumerate(selected, 1):
        instruments = [
            s.strip()
            for s in track["instrument"].replace(" and ", ", ").split(",")
            if s.strip()
        ]
        bpm_hint = None
        if track.get("bpm"):
            try:
                bpm_hint = int(track["bpm"])
            except (ValueError, TypeError):
                pass

        manifest.append(
            {
                "id": f"ab-{i:03d}",
                "corpus_file_path": f"ab_tracks/{track['stem']}.wav",
                "category": track["category"],
                "category_label": track["category"].title(),
                "mood_tags": track["moods"],
                "energy": track["energy"],
                "instrumentation": instruments,
                "bpm_hint": bpm_hint,
                "prompt": track["description"],
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tracks for A/B test")
    parser.add_argument(
        "--count", type=int, default=50, help="Number of tracks to select"
    )
    args = parser.parse_args()

    candidates = load_candidates()
    print(f"Found {len(candidates)} candidates with chroma >= {MIN_CHROMA}")

    from collections import Counter

    cat_dist = Counter(c["category"] for c in candidates)
    print("Category distribution (eligible):")
    for cat, n in cat_dist.most_common():
        print(f"  {cat}: {n}")

    selected = select_tracks(candidates, args.count)
    print(f"\nSelected {len(selected)} tracks:")

    sel_dist = Counter(s["category"] for s in selected)
    for cat, n in sel_dist.most_common():
        print(f"  {cat}: {n}")

    chroma_vals = [s["chroma_score"] for s in selected]
    print(f"\nChroma range: {min(chroma_vals):.3f} – {max(chroma_vals):.3f}")

    import shutil

    TRACKS_OUT.mkdir(parents=True, exist_ok=True)
    for track in selected:
        src = Path(track["wav_path"])
        dst = TRACKS_OUT / f"{track['stem']}.wav"
        shutil.copy2(src, dst)

    manifest = build_manifest(selected)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nManifest written to {MANIFEST_OUT}")
    print(f"Tracks copied to {TRACKS_OUT}/")
    print(f"\nNext steps:")
    print(f"  1. Upload ab_tracks/ WAVs + ab_corpus_manifest.json to corpus volume")
    print(f"  2. Run embed_corpus (pointed at ab_corpus_manifest.json)")
    print(f"  3. Reset Qdrant collection + run index_corpus")


if __name__ == "__main__":
    main()
