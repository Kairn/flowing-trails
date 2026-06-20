"""Analyze retrieval A/B test results.

Reads eval/ab_results.json, prints per-arm statistics, per-prompt comparison,
and a summary for the listening decision.

Usage:
    python scripts/analyze_ab_test.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_FILE = EVAL_DIR / "ab_results.json"

ARMS = ["none", "retrieval", "random"]
ARM_LABELS = {
    "none": "Text-only",
    "retrieval": "Top-1 retrieval",
    "random": "Random melody",
}


def main() -> None:
    results = json.loads(RESULTS_FILE.read_text())

    by_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    by_prompt: dict[str, dict[str, float | None]] = {}

    for r in results:
        arm = r["arm"]
        score = r["score"]
        prompt_id = r["id"]

        if score is not None:
            by_arm[arm].append(score)

        by_prompt.setdefault(prompt_id, {})[arm] = score

    print("=" * 72)
    print("RETRIEVAL A/B TEST — CLAP SCORE SUMMARY")
    print("=" * 72)

    print(
        f"\n{'Arm':<20s} {'N':>4s} {'Mean':>7s} {'Median':>7s} {'P25':>7s} {'Min':>7s} {'Max':>7s}"
    )
    print("-" * 72)

    for arm in ARMS:
        scores = by_arm[arm]
        if not scores:
            print(
                f"{ARM_LABELS[arm]:<20s} {'0':>4s}    —       —       —       —       —"
            )
            continue
        scores_sorted = sorted(scores)
        n = len(scores)
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        p25 = scores_sorted[n // 4]
        print(
            f"{ARM_LABELS[arm]:<20s} {n:>4d} {mean:>7.4f} {median:>7.4f} "
            f"{p25:>7.4f} {min(scores):>7.4f} {max(scores):>7.4f}"
        )

    none_scores = by_arm["none"]
    retrieval_scores = by_arm["retrieval"]
    if none_scores and retrieval_scores:
        delta = statistics.mean(retrieval_scores) - statistics.mean(none_scores)
        print(f"\nRetrieval vs Text-only mean delta: {delta:+.4f}")

    random_scores = by_arm["random"]
    if none_scores and random_scores:
        delta = statistics.mean(random_scores) - statistics.mean(none_scores)
        print(f"Random vs Text-only mean delta:    {delta:+.4f}")

    print(f"\n{'─' * 72}")
    print("PER-PROMPT COMPARISON")
    print(f"{'─' * 72}")
    print(
        f"{'Prompt':<20s} {'Text-only':>10s} {'Retrieval':>10s} {'Random':>10s} {'Best':>12s}"
    )
    print("-" * 72)

    wins = {"none": 0, "retrieval": 0, "random": 0}

    for prompt_id in sorted(by_prompt.keys()):
        scores = by_prompt[prompt_id]
        cols = []
        for arm in ARMS:
            s = scores.get(arm)
            cols.append(f"{s:.4f}" if s is not None else "  ERR")

        valid = {arm: scores[arm] for arm in ARMS if scores.get(arm) is not None}
        if valid:
            best_arm = max(valid, key=lambda a: valid[a])
            wins[best_arm] += 1
            best_label = ARM_LABELS[best_arm]
        else:
            best_label = "—"

        print(
            f"{prompt_id:<20s} {cols[0]:>10s} {cols[1]:>10s} {cols[2]:>10s} {best_label:>12s}"
        )

    print(f"\n{'─' * 72}")
    print("WIN COUNTS (highest CLAP score per prompt)")
    print(f"{'─' * 72}")
    for arm in ARMS:
        print(f"  {ARM_LABELS[arm]:<20s} {wins[arm]:>3d} / {len(by_prompt)}")

    print(f"\n{'=' * 72}")
    print("DECISION GUIDE")
    print(f"{'=' * 72}")
    print(
        "If retrieval wins on both CLAP scores AND listening quality → proceed to M7-T6"
    )
    print("If retrieval ≈ text-only or worse → skip M7-T6, keep text-only as default")
    print(
        "If random ≈ retrieval → retrieval is not adding semantic value, just conditioning noise"
    )


if __name__ == "__main__":
    main()
