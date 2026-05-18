# Labels Glossary — Human Labeling Reference

Comprehensive vocabulary for every discrete-label field in `training/source/labels.json`.
Use this when labeling tracks so you don't miss a category you didn't know existed. If a
track feels borderline between two labels, pick the dominant feel — labels capture intent,
not exhaustive description.

Schema lives in `DESIGN_TRAINING.md §Labels and Metadata`. Free-form fields (`composer`,
`notes`) have no glossary; write whatever helps.

For a fully-filled syntax template across varied scene types, see
`training/source/labels_example.json` (checked into the repo as a public reference).

---

## Source-Audio Requirements (before labeling)

- **No vocals** in source tracks. Wordless choir is fine. Strip or skip any track with sung lyrics.
- **Min 192 kbps** mp3 (or any ffmpeg-readable format at equivalent quality).
- Any sample rate — ffmpeg normalizes to 32 kHz mono in prep.
- Filenames: `track_001.mp3`, `track_002.mp3`, … (rename your originals).
- Place files in `training/source/` (gitignored, kept private).

---

## `scene_type` (pick one)

What gameplay or narrative context does this track score?

| Value         | Definition                                                                   | Example                                                  |
| ------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| `battle`      | Regular combat / random encounters.                                          | FF7 *Let the Battles Begin*, Chrono Trigger battle theme |
| `boss`        | Major / climactic combat. Stakes feel higher than `battle`.                  | FF7 *One-Winged Angel*, FF6 *Dancing Mad*                |
| `town`        | Settlements, peaceful populated locations.                                   | FF6 *Tina*, FF7 *Costa del Sol*                          |
| `exploration` | World map, traversal, journeying between locations.                          | Chrono Trigger world map, FF8 *Blue Fields*              |
| `dungeon`     | Caves, ruins, hostile non-combat zones, infiltration.                        | Chrono Trigger *Mt. Woe*, FF7 *Lifestream*               |
| `emotional`   | Cutscene / character-driven scenes. Story moments, not ambience.             | FF7 *Aerith's Theme*, FF8 *Eyes on Me* (instrumental)    |
| `ambient`     | Atmospheric backdrop without strong melodic identity. Texture-led.           | FF7 *Underneath the Rotting Pizza*, save-point hum       |
| `credits`     | End credits, ending cinematic sequences.                                     | FF9 *Memoria*, Chrono Trigger ending suite               |
| `menu`        | Title screens, pause menus, character menus, equipment screens.              | FF7 prelude/title, persona menu themes                   |

---

## `energy` (pick one)

How much physical/sonic momentum is in the track?

| Value     | Definition                                                                    |
| --------- | ----------------------------------------------------------------------------- |
| `low`     | Relaxed, slow tempo, sparse arrangement. Typically <90 BPM or very minimal.   |
| `medium`  | Moderate movement, present rhythm. Most exploration/town themes sit here.     |
| `high`    | Driving, dense, fast or punchy. Battle, chase, climax cues.                   |

---

## `mood_tags` (pick 2–4, array)

Emotional flavor. Pick the 2–4 most dominant. Order doesn't matter.

| Value         | Definition                                                                         | Distinguishing notes                                   |
| ------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `tense`       | Anxious, anticipatory, unresolved.                                                 | Pre-boss buildup, suspense scenes.                     |
| `triumphant`  | Victory, achievement, success.                                                     | Fanfare, win cues. Past-tense win.                     |
| `melancholic` | Sad, sorrowful, weighted.                                                          | Heavier than `nostalgic`; grief vs wistfulness.        |
| `mysterious`  | Enigmatic, intriguing, unknown.                                                    | Curiosity-tinged. Lighter than `dark`.                 |
| `peaceful`    | Calm, serene, content.                                                             | Small village themes, rest scenes.                     |
| `whimsical`   | Playful, light, charming.                                                          | Chocobo theme, comic relief.                           |
| `epic`        | Grand, large-scale, sweeping.                                                      | Main themes, large-scale conflict.                     |
| `dark`        | Foreboding, menacing, oppressive.                                                  | Villain themes, evil locations. Heavier than `mysterious`. |
| `hopeful`     | Optimistic, forward-looking, gentle uplift.                                        | Pre-victory, dawn scenes. Less concluded than `triumphant`. |
| `nostalgic`   | Wistful, memory-tinged, bittersweet.                                               | Childhood town themes. Lighter than `melancholic`.     |
| `urgent`      | Driving forward, time-pressure, momentum.                                          | Countdown/escape sequences. More action than `tense`.  |

---

## `dominant_instruments` (pick 1–4, array)

What instruments carry the track? "Dominant" means audibly leading, not just present.

| Value         | Definition                                                       |
| ------------- | ---------------------------------------------------------------- |
| `piano`       | Acoustic or electric piano carrying melody or harmony.           |
| `strings`     | Violin / viola / cello / contrabass section. Bowed strings.      |
| `brass`       | Trumpet, trombone, French horn, tuba.                            |
| `woodwinds`   | Flute, clarinet, oboe, bassoon.                                  |
| `choir`       | Wordless or non-language vocal ensemble (no lyrics).             |
| `synth`       | Electronic synthesizers — analog or digital, lead or pad.        |
| `percussion`  | Drums, mallets, ethnic percussion when prominent (not just kit). |
| `guitar`      | Acoustic or electric guitar, lead or rhythm.                     |

---

## `genre` (pick one)

The overall stylistic idiom.

| Value         | Definition                                                                   |
| ------------- | ---------------------------------------------------------------------------- |
| `orchestral`  | Full classical instrumentation. Strings, brass, woodwinds, percussion.       |
| `chiptune`    | 8-bit / 16-bit synthesis. NES / SNES / Game Boy idiom.                       |
| `synthwave`   | 80s-style electronic, retro synth lead and arpeggios.                        |
| `jazz`        | Jazz idiom — extended chords, swing feel, improvisational phrasing.          |
| `rock`        | Rock-band instrumentation: drums + electric guitar + bass core.              |
| `ambient`     | Atmospheric / textural. Low rhythmic emphasis, soundscape-driven.            |

---

## Free-form fields (no glossary)

| Field      | Notes                                                                              |
| ---------- | ---------------------------------------------------------------------------------- |
| `composer` | Composer name (string) or `null` if unknown.                                       |
| `notes`    | Free-form description. Used as `keywords` in the training sidecar. Or `null`.      |
