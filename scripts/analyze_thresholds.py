"""Analyze calibration results and produce eval/thresholds.json.

Reads eval/calibration_results.json, computes percentile-based thresholds,
and writes eval/thresholds.json with the recommended accept threshold.

Usage:
    python scripts/analyze_thresholds.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_FILE = EVAL_DIR / "calibration_results.json"
THRESHOLDS_FILE = EVAL_DIR / "thresholds.json"


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}")
        print("Run calibration first: python scripts/run_calibration.py <url>")
        raise SystemExit(1)

    results = json.loads(RESULTS_FILE.read_text())
    valid = [r for r in results if r.get("score") is not None]

    if len(valid) < 10:
        print(
            f"Only {len(valid)} valid results — need at least 10 for reliable thresholds."
        )
        raise SystemExit(1)

    all_scores = sorted(r["score"] for r in valid)

    p10 = percentile(all_scores, 10)
    p25 = percentile(all_scores, 25)
    p50 = percentile(all_scores, 50)
    p75 = percentile(all_scores, 75)
    p90 = percentile(all_scores, 90)
    mean = statistics.mean(all_scores)
    stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    # Accept threshold: p25 — accepts the top 75% of generations without retry.
    # Below p25 triggers a retry, which should improve most borderline cases.
    accept_threshold = round(p25, 4)

    by_category: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_category[r["category"]].append(r["score"])

    category_stats = {}
    for cat in sorted(by_category):
        cat_scores = sorted(by_category[cat])
        category_stats[cat] = {
            "count": len(cat_scores),
            "min": round(min(cat_scores), 4),
            "max": round(max(cat_scores), 4),
            "mean": round(statistics.mean(cat_scores), 4),
            "median": round(percentile(cat_scores, 50), 4),
        }

    thresholds = {
        "model": "facebook/musicgen-melody-large",
        "calibration_samples": len(valid),
        "accept_threshold": accept_threshold,
        "distribution": {
            "min": round(min(all_scores), 4),
            "p10": round(p10, 4),
            "p25": round(p25, 4),
            "p50_median": round(p50, 4),
            "p75": round(p75, 4),
            "p90": round(p90, 4),
            "max": round(max(all_scores), 4),
            "mean": round(mean, 4),
            "stdev": round(stdev, 4),
        },
        "by_category": category_stats,
    }

    THRESHOLDS_FILE.write_text(json.dumps(thresholds, indent=2) + "\n")

    print("Calibration Analysis")
    print("=" * 60)
    print(f"Samples:           {len(valid)}")
    print(f"Score range:       {min(all_scores):.4f} – {max(all_scores):.4f}")
    print(f"Mean +/- stdev:    {mean:.4f} +/- {stdev:.4f}")
    print(f"")
    print(f"  p10:  {p10:.4f}")
    print(f"  p25:  {p25:.4f}  <-- accept threshold")
    print(f"  p50:  {p50:.4f}")
    print(f"  p75:  {p75:.4f}")
    print(f"  p90:  {p90:.4f}")
    print()
    print("By category:")
    for cat, stats in category_stats.items():
        print(
            f"  {cat:15s}  n={stats['count']}  "
            f"mean={stats['mean']:.4f}  "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
        )

    print(f"\nRecommended accept_threshold: {accept_threshold}")
    print(f"Written to {THRESHOLDS_FILE}")
    print(
        f"\nNext: update DEFAULT_SIMILARITY_THRESHOLD in config.py to {accept_threshold}"
    )


if __name__ == "__main__":
    main()
