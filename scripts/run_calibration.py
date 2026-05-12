"""Run calibration prompts against the deployed /compose endpoint.

Collects CLAP similarity scores, attempt counts, and latencies for threshold analysis.

Usage:
    python scripts/run_calibration.py <compose-endpoint-url>
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import requests

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
PROMPTS_FILE = EVAL_DIR / "calibration_prompts.json"
RESULTS_FILE = EVAL_DIR / "calibration_results.json"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_calibration.py <compose-endpoint-url>")
        sys.exit(1)

    endpoint_url = sys.argv[1]
    prompts = json.loads(PROMPTS_FILE.read_text())
    results: list[dict] = []
    scores: list[float] = []

    print(f"Running {len(prompts)} calibration prompts against {endpoint_url}\n")

    for i, entry in enumerate(prompts, 1):
        prompt_id = entry["id"]
        category = entry["category"]
        request_body = entry["request"]

        print(f"[{i}/{len(prompts)}] {prompt_id} ({category})... ", end="", flush=True)

        t0 = time.monotonic()
        try:
            resp = requests.post(endpoint_url, json=request_body, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.monotonic() - t0

            score = data.get("score", 0.0)
            attempts = data.get("attempts", 0)
            has_audio = data.get("audio_b64") is not None

            scores.append(score)
            print(f"score={score:.4f}  attempts={attempts}  {elapsed:.1f}s")

            result = {
                "id": prompt_id,
                "category": category,
                "description": request_body["description"],
                "score": score,
                "attempts": attempts,
                "latency_s": round(elapsed, 2),
                "has_audio": has_audio,
                "trace_id": data.get("trace_id"),
                "spec": data.get("spec"),
            }

            if has_audio:
                audio_bytes = base64.b64decode(data["audio_b64"])
                wav_path = EVAL_DIR / f"calibration_{prompt_id}.wav"
                wav_path.write_bytes(audio_bytes)
                result["wav_file"] = wav_path.name
                result["audio_size_bytes"] = len(audio_bytes)

            results.append(result)

        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            results.append(
                {
                    "id": prompt_id,
                    "category": category,
                    "description": request_body["description"],
                    "score": None,
                    "attempts": None,
                    "latency_s": round(elapsed, 2),
                    "has_audio": False,
                    "error": str(e),
                }
            )

    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")

    succeeded = [r for r in results if r["score"] is not None]
    failed = [r for r in results if r["score"] is None]

    print(f"\n{'='*60}")
    print(f"Done. {len(succeeded)}/{len(results)} succeeded, {len(failed)} failed.")
    if scores:
        scores.sort()
        print(f"Score range: {min(scores):.4f} – {max(scores):.4f}")
        print(f"Median: {scores[len(scores)//2]:.4f}")
        print(f"Mean:   {sum(scores)/len(scores):.4f}")
    print(f"\nResults written to {RESULTS_FILE}")
    print(f"Next step: python scripts/analyze_thresholds.py")


if __name__ == "__main__":
    main()
