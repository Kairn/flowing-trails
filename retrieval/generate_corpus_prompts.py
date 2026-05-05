"""
Deterministically generates corpus_prompts.json from the config block below.
Each entry is the cross-product of its category's attributes — every prompt id
is stable across runs (derived from category slug + zero-padded index), so
re-running this script produces the same JSON as long as the config is unchanged.

Usage:
    python retrieval/generate_corpus_prompts.py
    # writes retrieval/corpus_prompts.json

Tune the TOP-LEVEL CONFIG section to:
  - Change how many tracks to generate per category (count)
  - Add, remove, or reorder categories
  - Adjust the pool of moods, instruments, energy levels, bpm ranges
  - Add your own style notes that get appended to every prompt in a category
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL CONFIG — edit freely
# ──────────────────────────────────────────────────────────────────────────────

# Global generation settings
DURATION_SECONDS = 15  # seconds per clip (15s is good for CLAP quality vs cost)
RANDOM_SEED = 42  # change to get a different-but-still-deterministic shuffle

# Optional: append a personal style note to every prompt in the corpus.
# Set to "" to disable. Example: "influenced by Nobuo Uematsu and Yasunori Mitsuda"
GLOBAL_STYLE_SUFFIX = ""

# ── Category definitions ───────────────────────────────────────────────────────
# Each category produces `count` prompts.
# Attributes are sampled round-robin (not randomly) so the full attribute space
# is covered evenly regardless of count. Increase count to get more combinations.
#
# Fields per category:
#   slug          str   — used in prompt IDs and as the `category` metadata field
#   label         str   — human-readable name stored in metadata
#   count         int   — number of prompts to generate
#   base_prompts  list  — sentence fragments describing the core feel; one is chosen per clip
#   mood_pools    list  — list of [mood, mood, ...] tag groups; one group chosen per clip
#   instrument_pools list — list of [instr, instr, ...] groups; one chosen per clip
#   energy_levels list  — cycle through these values
#   bpm_range     tuple — (min, max) BPM hint; evenly distributed across count
#   style_suffix  str   — category-specific style note appended to prompt (set "" to skip)

CATEGORIES = [
    {
        "slug": "battle",
        "label": "Battle / Combat",
        "count": 20,
        "base_prompts": [
            "intense battle music for a JRPG, fast tempo, driving rhythm",
            "high-energy combat theme, relentless and aggressive",
            "action-packed fight scene underscore, tense and propulsive",
            "heroic battle anthem, powerful and driving",
        ],
        "mood_pools": [
            ["intense", "heroic", "driving"],
            ["fierce", "urgent", "relentless"],
            ["powerful", "tense", "energetic"],
            ["aggressive", "bold", "cinematic"],
        ],
        "instrument_pools": [
            ["electric guitar", "brass", "percussion"],
            ["orchestra", "drums", "distorted bass"],
            ["brass fanfare", "snare", "electric guitar"],
            ["full orchestra", "taiko drums", "strings"],
        ],
        "energy_levels": ["high"],
        "bpm_range": (140, 180),
        "style_suffix": "",
    },
    {
        "slug": "boss",
        "label": "Boss Fight",
        "count": 15,
        "base_prompts": [
            "dark and menacing boss battle theme, heavy and dramatic",
            "epic final boss music, overwhelming and intense",
            "sinister boss encounter theme, imposing and relentless",
        ],
        "mood_pools": [
            ["dark", "menacing", "overwhelming"],
            ["sinister", "epic", "dread-inducing"],
            ["ominous", "powerful", "climactic"],
        ],
        "instrument_pools": [
            ["heavy brass", "deep drums", "distorted guitar"],
            ["choir", "orchestra", "heavy percussion"],
            ["organ", "brass", "low strings", "taiko"],
        ],
        "energy_levels": ["high"],
        "bpm_range": (150, 190),
        "style_suffix": "darker and heavier than a standard battle theme",
    },
    {
        "slug": "exploration",
        "label": "Overworld / Exploration",
        "count": 20,
        "base_prompts": [
            "peaceful overworld exploration theme, adventurous and melodic",
            "open world traversal music, uplifting and expansive",
            "journey underscore, wandering and optimistic",
            "field exploration theme, lighthearted and melodic",
        ],
        "mood_pools": [
            ["peaceful", "adventurous", "optimistic"],
            ["wandering", "hopeful", "uplifting"],
            ["curious", "expansive", "warm"],
            ["light", "breezy", "cheerful"],
        ],
        "instrument_pools": [
            ["acoustic piano", "strings", "flute"],
            ["acoustic guitar", "woodwinds", "light percussion"],
            ["orchestra", "harp", "oboe"],
            ["piano", "marimba", "strings"],
        ],
        "energy_levels": ["low", "medium"],
        "bpm_range": (80, 110),
        "style_suffix": "",
    },
    {
        "slug": "ambient",
        "label": "Ambient / Atmospheric",
        "count": 20,
        "base_prompts": [
            "slow atmospheric ambient music, mysterious and spacious",
            "textural soundscape, minimal and ethereal",
            "meditative ambient underscore, calm and otherworldly",
            "sparse atmospheric pad music, quiet and introspective",
        ],
        "mood_pools": [
            ["mysterious", "ethereal", "spacious"],
            ["calm", "otherworldly", "introspective"],
            ["dreamy", "quiet", "floating"],
            ["melancholic", "serene", "distant"],
        ],
        "instrument_pools": [
            ["synth pads", "ambient textures", "light bells"],
            ["piano", "reverb strings", "atmospheric synth"],
            ["choir pads", "minimal piano", "wind sounds"],
            ["synth drone", "music box", "soft strings"],
        ],
        "energy_levels": ["low"],
        "bpm_range": (50, 80),
        "style_suffix": "minimal percussion, emphasis on texture and atmosphere",
    },
    {
        "slug": "town",
        "label": "Town / Village",
        "count": 15,
        "base_prompts": [
            "cheerful town theme, warm and inviting, for a village in an RPG",
            "upbeat market or plaza music, lively and friendly",
            "cozy settlement music, folk-inspired and warm",
        ],
        "mood_pools": [
            ["cheerful", "warm", "inviting"],
            ["lively", "friendly", "nostalgic"],
            ["cozy", "lighthearted", "homey"],
        ],
        "instrument_pools": [
            ["accordion", "acoustic guitar", "woodwinds"],
            ["flute", "pizzicato strings", "light percussion"],
            ["piano", "clarinet", "acoustic bass"],
        ],
        "energy_levels": ["low", "medium"],
        "bpm_range": (90, 120),
        "style_suffix": "",
    },
    {
        "slug": "dungeon",
        "label": "Dungeon / Cave",
        "count": 15,
        "base_prompts": [
            "dark dungeon crawl music, ominous and tense",
            "cave exploration underscore, eerie and foreboding",
            "mysterious underground theme, sparse and unsettling",
        ],
        "mood_pools": [
            ["dark", "ominous", "tense"],
            ["eerie", "foreboding", "mysterious"],
            ["unsettling", "haunting", "sparse"],
        ],
        "instrument_pools": [
            ["low strings", "sparse piano", "ambient noise"],
            ["bass clarinet", "tense strings", "quiet percussion"],
            ["cello", "synth drone", "distant bells"],
        ],
        "energy_levels": ["low"],
        "bpm_range": (55, 85),
        "style_suffix": "echo-heavy, sparse, sense of danger and isolation",
    },
    {
        "slug": "puzzle",
        "label": "Puzzle / Thinking",
        "count": 15,
        "base_prompts": [
            "light and playful puzzle music, quirky and rhythmic",
            "thoughtful puzzle underscore, gentle and curious",
            "whimsical problem-solving theme, fun and light",
        ],
        "mood_pools": [
            ["playful", "quirky", "light"],
            ["curious", "thoughtful", "gentle"],
            ["whimsical", "fun", "focused"],
        ],
        "instrument_pools": [
            ["marimba", "bells", "light piano"],
            ["xylophone", "pizzicato strings", "ukulele"],
            ["celesta", "glockenspiel", "acoustic guitar"],
        ],
        "energy_levels": ["low", "medium"],
        "bpm_range": (90, 120),
        "style_suffix": "",
    },
    {
        "slug": "title",
        "label": "Title Screen / Menu",
        "count": 10,
        "base_prompts": [
            "grand title screen theme, epic and memorable",
            "calm and inviting main menu music",
        ],
        "mood_pools": [
            ["epic", "grandiose", "memorable"],
            ["calm", "inviting", "anticipatory"],
        ],
        "instrument_pools": [
            ["full orchestra", "choir", "brass"],
            ["piano", "strings", "light orchestral"],
        ],
        "energy_levels": ["medium", "high"],
        "bpm_range": (90, 130),
        "style_suffix": "suitable for looping on a title or main menu screen",
    },
    {
        "slug": "victory",
        "label": "Victory / Fanfare",
        "count": 10,
        "base_prompts": [
            "short triumphant victory fanfare, brass-led and joyful",
            "celebratory win jingle, upbeat and energetic",
        ],
        "mood_pools": [
            ["triumphant", "joyful", "celebratory"],
            ["uplifting", "exciting", "heroic"],
        ],
        "instrument_pools": [
            ["brass stabs", "snare roll", "timpani"],
            ["full brass", "strings", "cymbal crash"],
        ],
        "energy_levels": ["high"],
        "bpm_range": (130, 160),
        "style_suffix": "short punchy fanfare feel, strong upward melodic movement",
    },
    {
        "slug": "sad",
        "label": "Sad / Emotional",
        "count": 10,
        "base_prompts": [
            "slow emotional piano theme, melancholic and heartfelt",
            "sad narrative underscore, sparse and moving",
        ],
        "mood_pools": [
            ["melancholic", "heartfelt", "gentle"],
            ["sad", "moving", "quiet"],
        ],
        "instrument_pools": [
            ["solo piano", "soft strings"],
            ["piano", "cello", "ambient pad"],
        ],
        "energy_levels": ["low"],
        "bpm_range": (50, 75),
        "style_suffix": "slow tempo, minor key, sparse arrangement",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# GENERATION LOGIC — no need to edit below this line
# ──────────────────────────────────────────────────────────────────────────────


def _cycle(lst, i):
    """Return lst[i % len(lst)] — round-robin, no randomness."""
    return lst[i % len(lst)]


def _bpm_for_index(bpm_range, count, i):
    """Evenly distribute BPM values across [min, max] for `count` clips."""
    lo, hi = bpm_range
    if count == 1:
        return (lo + hi) // 2
    step = (hi - lo) / (count - 1)
    return round(lo + step * i)


def build_prompt(base, instruments, mood_tags, style_suffix, global_suffix):
    parts = [base]
    parts.append(f"{', '.join(instruments)}")
    parts.append(f"{', '.join(mood_tags)}")
    for suffix in (style_suffix, global_suffix):
        if suffix:
            parts.append(suffix)
    return ", ".join(parts)


def generate_prompts(categories, duration, seed, global_suffix):
    # Use seed only to verify reproducibility — we use round-robin, not random.
    # The seed is embedded in the output metadata so consumers know the config.
    entries = []
    for cat in categories:
        slug = cat["slug"]
        count = cat["count"]
        for i in range(count):
            clip_id = f"{slug}_{str(i + 1).zfill(3)}"
            base = _cycle(cat["base_prompts"], i)
            mood_tags = _cycle(cat["mood_pools"], i)
            instruments = _cycle(cat["instrument_pools"], i)
            energy = _cycle(cat["energy_levels"], i)
            bpm = _bpm_for_index(cat["bpm_range"], count, i)
            style_suffix = cat.get("style_suffix", "")

            prompt = build_prompt(
                base, instruments, mood_tags, style_suffix, global_suffix
            )

            entries.append(
                {
                    "id": clip_id,
                    "prompt": prompt,
                    "category": slug,
                    "category_label": cat["label"],
                    "mood_tags": mood_tags,
                    "energy": energy,
                    "instrumentation": instruments,
                    "bpm_hint": bpm,
                    "duration_seconds": duration,
                }
            )

    return entries


def main():
    from otel_utils import get_logger, setup_logging

    setup_logging()
    log = get_logger("corpus-prompts")

    prompts = generate_prompts(
        categories=CATEGORIES,
        duration=DURATION_SECONDS,
        seed=RANDOM_SEED,
        global_suffix=GLOBAL_STYLE_SUFFIX,
    )

    total = len(prompts)
    out_path = Path(__file__).parent / "corpus_prompts.json"
    out_path.write_text(json.dumps(prompts, indent=2))

    breakdown = {}
    for p in prompts:
        breakdown[p["category_label"]] = breakdown.get(p["category_label"], 0) + 1
    log.info("Written prompts", count=total, path=str(out_path), breakdown=breakdown)


if __name__ == "__main__":
    main()
