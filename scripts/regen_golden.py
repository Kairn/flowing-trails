"""Regenerate the golden CI tracks against the deployed compose endpoint.

Re-runs each golden brief through /compose on the current production model, overwrites
each `<id>.wav` in eval/golden/, re-parses the spec, and recaptures `baseline_score`
as CLAP(audio)·CLAP(spec.clap_text()) computed locally — so the CI regression gate
(test_ci_eval.py) reproduces the stored baseline exactly.

Duration is pinned to 10s to match the original fixture (and keep generation cheap).
Specs are refreshed from the live parser; the briefs (request_description) are the
stable identity of the set and are preserved.

Usage:
    COMPOSE_URL=https://... make regen-golden
    # or directly:
    python scripts/regen_golden.py <compose-url> [--model MODEL]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clap_utils import embed_text
from models import MusicSpec
from scoring import score_generation

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "eval" / "golden"
GOLDEN_PROMPTS = GOLDEN_DIR / "golden_prompts.json"

DURATION_SECONDS = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate golden CI tracks")
    parser.add_argument("compose_url", help="Deployed compose endpoint URL")
    parser.add_argument(
        "--model", default=None, help="Claude model override for query parsing"
    )
    args = parser.parse_args()

    entries = json.loads(GOLDEN_PROMPTS.read_text())

    for i, entry in enumerate(entries, 1):
        pid = entry["id"]
        body = {
            "description": entry["request_description"],
            "melody_source": "none",
            "duration_seconds": DURATION_SECONDS,
        }
        if args.model:
            body["model"] = args.model

        print(f"[{i:2d}/{len(entries)}] {pid:<16} generating...", flush=True)
        resp = requests.post(args.compose_url, json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        audio = base64.b64decode(data["audio_b64"])
        (GOLDEN_DIR / f"{pid}.wav").write_bytes(audio)

        spec = MusicSpec(**data["spec"])
        baseline = float(score_generation(audio, embed_text(spec.clap_text())))

        entry["spec"] = spec.model_dump()
        entry["baseline_score"] = round(baseline, 4)
        print(
            f"      baseline={entry['baseline_score']:.4f}  "
            f"attempts={data.get('attempts')}  bytes={len(audio):,}"
        )

    GOLDEN_PROMPTS.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"\nUpdated {len(entries)} entries in {GOLDEN_PROMPTS}")
    print("Now run `make eval` to confirm the regenerated set is green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
