"""Full eval suite — run all 25 prompts end-to-end against the deployed compose endpoint.

Sends each calibration prompt, collects CLAP scores, and reports pass/fail
against DEFAULT_SIMILARITY_THRESHOLD. Defaults to Haiku for Claude steps.

Usage:
    python scripts/run_full_eval.py <compose-endpoint-url> [--model MODEL]

Exit codes:
    0 — all prompts passed
    1 — one or more prompts failed or errored
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
PROMPTS_FILE = EVAL_DIR / "calibration_prompts.json"
RESULTS_FILE = EVAL_DIR / "eval_results.json"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
SIMILARITY_THRESHOLD = 0.40


def run_prompt(endpoint_url: str, entry: dict, model: str) -> dict:
    prompt_id = entry["id"]
    request_body = dict(entry["request"], model=model)

    t0 = time.monotonic()
    try:
        resp = requests.post(endpoint_url, json=request_body, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0

        score = data.get("score", 0.0)
        passed = score >= SIMILARITY_THRESHOLD

        result = {
            "id": prompt_id,
            "category": entry["category"],
            "description": entry["request"]["description"],
            "score": score,
            "passed": passed,
            "attempts": data.get("attempts", 0),
            "latency_s": round(elapsed, 2),
            "trace_id": data.get("trace_id"),
            "spec": data.get("spec"),
            "model": model,
        }

        if data.get("audio_b64"):
            audio_bytes = base64.b64decode(data["audio_b64"])
            wav_path = EVAL_DIR / f"{prompt_id}.wav"
            wav_path.write_bytes(audio_bytes)
            result["wav_file"] = wav_path.name

        return result

    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "id": prompt_id,
            "category": entry["category"],
            "description": entry["request"]["description"],
            "score": None,
            "passed": False,
            "attempts": None,
            "latency_s": round(elapsed, 2),
            "error": str(e),
            "model": model,
        }


def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"] and r["score"] is not None)
    errored = sum(1 for r in results if r["score"] is None)
    scores = [r["score"] for r in results if r["score"] is not None]

    print(f"\n{'=' * 68}")
    print(
        f"FULL EVAL RESULTS: {passed}/{total} passed  "
        f"({failed} failed, {errored} errored)"
    )
    print(f"Threshold: {SIMILARITY_THRESHOLD}")

    if scores:
        scores.sort()
        mean = sum(scores) / len(scores)
        p25 = scores[len(scores) // 4]
        median = scores[len(scores) // 2]
        print(
            f"Scores — min: {min(scores):.4f}  p25: {p25:.4f}  "
            f"median: {median:.4f}  mean: {mean:.4f}  max: {max(scores):.4f}"
        )

    latencies = [r["latency_s"] for r in results if r["score"] is not None]
    if latencies:
        print(
            f"Latency — min: {min(latencies):.1f}s  "
            f"mean: {sum(latencies)/len(latencies):.1f}s  "
            f"max: {max(latencies):.1f}s"
        )

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'─' * 68}")
        print("FAILURES:")
        for r in failures:
            if r["score"] is not None:
                print(f"  ✗ {r['id']}: score {r['score']:.4f} < {SIMILARITY_THRESHOLD}")
            else:
                print(f"  ✗ {r['id']}: ERROR — {r.get('error', 'unknown')}")

    print(f"{'=' * 68}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full E2E eval suite")
    parser.add_argument("compose_url", help="Deployed compose endpoint URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model for parse/refine (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text())
    results: list[dict] = []

    print(f"Running {len(prompts)} prompts against {args.compose_url}")
    print(f"Claude model: {args.model}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}\n")

    for i, entry in enumerate(prompts, 1):
        prompt_id = entry["id"]
        print(f"[{i:2d}/{len(prompts)}] {prompt_id:<20s} ", end="", flush=True)

        result = run_prompt(args.compose_url, entry, args.model)
        results.append(result)

        if result["score"] is not None:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{status}  score={result['score']:.4f}  "
                f"attempts={result['attempts']}  {result['latency_s']:.1f}s"
            )
        else:
            print(f"ERROR  {result.get('error', '')[:60]}  {result['latency_s']:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {RESULTS_FILE}")

    print_summary(results)

    all_passed = all(r["passed"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
