"""Retrieval A/B test: 20 prompts × 3 arms (text-only, top-1 retrieval, random melody).

Hits the deployed /compose endpoint with each prompt under each melody_source mode,
collects CLAP scores and WAV files for listening comparison.

Usage:
    python scripts/run_ab_test.py <compose-endpoint-url> [--model MODEL]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
PROMPTS_FILE = EVAL_DIR / "ab_prompts.json"
RESULTS_FILE = EVAL_DIR / "ab_results.json"
WAVS_DIR = EVAL_DIR / "ab_wavs"

ARMS = ["none", "retrieval", "random"]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def run_one(endpoint_url: str, entry: dict, arm: str, model: str) -> dict:
    request_body = dict(entry["request"], melody_source=arm, model=model)

    t0 = time.monotonic()
    try:
        resp = requests.post(endpoint_url, json=request_body, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0

        score = data.get("score", 0.0)
        result = {
            "id": entry["id"],
            "category": entry["category"],
            "arm": arm,
            "score": score,
            "attempts": data.get("attempts", 0),
            "latency_s": round(elapsed, 2),
            "trace_id": data.get("trace_id"),
        }

        if data.get("audio_b64"):
            audio_bytes = base64.b64decode(data["audio_b64"])
            wav_name = f"{entry['id']}_{arm}.wav"
            wav_path = WAVS_DIR / wav_name
            wav_path.write_bytes(audio_bytes)
            result["wav_file"] = wav_name

        return result

    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "id": entry["id"],
            "category": entry["category"],
            "arm": arm,
            "score": None,
            "attempts": None,
            "latency_s": round(elapsed, 2),
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval A/B test")
    parser.add_argument("compose_url", help="Deployed compose endpoint URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text())
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_runs = len(prompts) * len(ARMS)
    run_num = 0

    print(f"A/B test: {len(prompts)} prompts × {len(ARMS)} arms = {total_runs} runs")
    print(f"Endpoint: {args.compose_url}")
    print(f"Model: {args.model}\n")

    for entry in prompts:
        for arm in ARMS:
            run_num += 1
            label = f"[{run_num:2d}/{total_runs}] {entry['id']:<20s} {arm:<10s}"
            print(f"{label} ", end="", flush=True)

            result = run_one(args.compose_url, entry, arm, args.model)
            results.append(result)

            if result["score"] is not None:
                print(
                    f"score={result['score']:.4f}  "
                    f"attempts={result['attempts']}  "
                    f"{result['latency_s']:.1f}s"
                )
            else:
                print(
                    f"ERROR  {result.get('error', '')[:60]}  {result['latency_s']:.1f}s"
                )

    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {RESULTS_FILE}")
    print(f"WAVs written to {WAVS_DIR}/")
    print(f"\nNext: python scripts/analyze_ab_test.py")


if __name__ == "__main__":
    main()
