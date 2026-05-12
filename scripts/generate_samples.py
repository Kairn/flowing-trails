"""Generate sample audio files by calling the deployed /compose endpoint.

Usage:
    python scripts/generate_samples.py <compose-endpoint-url>

Reads prompts from samples/prompts.json, POSTs each to the endpoint,
and writes .wav files + manifest.json to samples/.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import requests

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
PROMPTS_FILE = SAMPLES_DIR / "prompts.json"
MANIFEST_FILE = SAMPLES_DIR / "manifest.json"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_samples.py <compose-endpoint-url>")
        sys.exit(1)

    endpoint_url = sys.argv[1]
    prompts = json.loads(PROMPTS_FILE.read_text())
    manifest: list[dict] = []

    for entry in prompts:
        slug = entry["slug"]
        request_body = entry["request"]

        print(f"[{slug}] Sending request...")
        resp = requests.post(endpoint_url, json=request_body, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        audio_b64 = data.get("audio_b64")
        if not audio_b64:
            print(f"[{slug}] No audio returned, skipping.")
            continue

        audio_data = base64.b64decode(audio_b64)
        filename = f"{slug}.wav"
        output_path = SAMPLES_DIR / filename
        output_path.write_bytes(audio_data)

        manifest.append(
            {
                "slug": slug,
                "filename": filename,
                "brief": request_body["description"],
                "spec": data.get("spec"),
                "score": data.get("score"),
                "attempts": data.get("attempts"),
                "trace_id": data.get("trace_id"),
                "size_bytes": len(audio_data),
                "use_melody_conditioning": request_body.get(
                    "use_melody_conditioning", True
                ),
            }
        )

        print(f"[{slug}] Saved {filename} ({len(audio_data):,} bytes)")

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone. {len(manifest)} samples written to {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
