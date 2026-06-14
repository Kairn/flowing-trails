"""Download audio and build labels.json from labeled export CSVs.

Reads all CSVs from training/exports/, filters to include=TRUE rows,
downloads audio via yt-dlp, assigns anonymous sequential names, and
writes training/source/labels.json.

Usage:
    python prep_from_exports.py [--workers N]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
EXPORTS_DIR = ROOT / "exports"
SOURCE_DIR = ROOT / "source"

DEFAULT_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 5
BOOL_TRUE = {"true", "1", "yes"}


def parse_list_field(value: str) -> list[str]:
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in BOOL_TRUE


def load_all_exports() -> list[dict]:
    rows = []
    csv_files = sorted(EXPORTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {EXPORTS_DIR}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if parse_bool(row.get("include", "")):
                    rows.append(row)
                    count += 1
        if count:
            print(f"[load] {csv_path.name}: {count} included")

    print(f"\n{len(rows)} tracks marked include=TRUE across {len(csv_files)} files")
    return rows


def download_track(url: str, output_path: Path) -> bool:
    template = str(output_path.parent / f"{output_path.stem}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "320K",
        "--embed-metadata",
        "--no-playlist",
        "-o",
        template,
        url,
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return False


def build_label(row: dict, filename: str) -> dict:
    return {
        "filename": filename,
        "scene_type": row.get("scene_type", "").strip().lower() or None,
        "energy": row.get("energy", "").strip().lower() or None,
        "mood_tags": parse_list_field(row.get("mood_tags", "")),
        "dominant_instruments": parse_list_field(row.get("instruments", "")),
        "genre": row.get("genre", "").strip().lower() or None,
        "composer": row.get("composer", "").strip() or None,
        "notes": row.get("notes", "").strip() or None,
    }


def _download_job(item: dict) -> dict:
    """Single download unit for thread pool."""
    i = item["index"]
    row = item["row"]
    output_path = item["output_path"]
    url = row["link"].strip()
    title = row.get("title", "").strip()

    if output_path.exists():
        return {"ok": True, "skipped": True, "item": item}

    ok = download_track(url, output_path)
    return {"ok": ok, "skipped": False, "item": item}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    SOURCE_DIR.mkdir(exist_ok=True)
    rows = load_all_exports()

    if not rows:
        print("No tracks to process.")
        return

    seen_urls: set[str] = set()
    unique_rows: list[dict] = []
    for row in rows:
        url = row["link"].strip()
        if url in seen_urls:
            print(f"  [skip] duplicate: {row.get('title', url)}")
            continue
        seen_urls.add(url)
        unique_rows.append(row)

    total = len(unique_rows)
    print(f"{total} unique tracks to download (workers={args.workers})\n")

    jobs = []
    for i, row in enumerate(unique_rows, start=1):
        filename = f"track_{i:04d}.mp3"
        jobs.append(
            {
                "index": i,
                "row": row,
                "output_path": SOURCE_DIR / filename,
                "filename": filename,
            }
        )

    results: dict[int, dict] = {}
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            item = result["item"]
            i = item["index"]
            title = item["row"].get("title", "").strip()
            filename = item["filename"]

            if result["ok"]:
                tag = "exists" if result["skipped"] else "done"
                print(f"[{i}/{total}] [{tag}] {filename}  {title}")
                results[i] = result
            else:
                url = item["row"]["link"].strip()
                print(f"[{i}/{total}] [FAIL]  {filename}  {title}")
                errors.append({"index": i, "url": url, "title": title})

    labels = []
    for job in jobs:
        if job["index"] in results:
            labels.append(build_label(job["row"], job["filename"]))

    labels_path = SOURCE_DIR / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"\nWrote {len(labels)} entries to {labels_path}")

    if errors:
        print(f"\n{len(errors)} ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  [{err['index']}] {err['title'] or err['url']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
