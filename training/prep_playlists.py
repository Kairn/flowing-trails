"""Generate blank labeling CSVs from YouTube playlist URLs.

Reads playlist URLs from training/playlists.txt (one per line).

Usage:
    python prep_playlists.py [path/to/playlists.txt]
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "playlists.txt"
EXPORTS_DIR = ROOT / "exports"

COLUMNS = [
    "link",
    "title",
    "scene_type",
    "energy",
    "mood_tags",
    "genre",
    "instruments",
    "composer",
    "notes",
    "include",
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "playlist"


def get_playlist_info(playlist_url: str) -> tuple[str, list[dict]]:
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--yes-playlist", "-J", playlist_url],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    playlist_title = data.get("title", "unknown-playlist")

    entries = []
    for entry in data.get("entries", []):
        video_id = entry.get("id", "")
        title = entry.get("title", "")
        url = f"https://www.youtube.com/watch?v={video_id}"
        entries.append({"link": url, "title": title})

    return playlist_title, entries


def write_csv(playlist_title: str, entries: list[dict]) -> Path:
    EXPORTS_DIR.mkdir(exist_ok=True)
    filename = f"{slugify(playlist_title)}.csv"
    out_path = EXPORTS_DIR / filename

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for entry in entries:
            row = {col: "" for col in COLUMNS}
            row["link"] = entry["link"]
            row["title"] = entry["title"]
            row["include"] = "FALSE"
            writer.writerow(row)

    return out_path


def main() -> None:
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not input_file.exists():
        print(f"Not found: {input_file}", file=sys.stderr)
        print(f"Create it with one playlist URL per line.", file=sys.stderr)
        sys.exit(1)

    urls = [
        line.strip()
        for line in input_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        print(f"No URLs found in {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"[input] {len(urls)} playlists from {input_file}\n")

    for url in urls:
        print(f"[playlist] {url}")
        try:
            playlist_title, entries = get_playlist_info(url)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e.stderr}", file=sys.stderr)
            continue
        out_path = write_csv(playlist_title, entries)
        print(f"  → {out_path.name} ({len(entries)} tracks)")

    print(f"\nCSVs written to {EXPORTS_DIR}/")


if __name__ == "__main__":
    main()
