# Labels Glossary — Human Labeling Reference

Comprehensive vocabulary for every discrete-label field in `training/source/labels.json`.
Use this when labeling tracks so you don't miss a category you didn't know existed. If a
track feels borderline between two labels, pick the dominant feel — labels capture intent,
not exhaustive description.

Schema lives in `DESIGN_TRAINING.md §Labels and Metadata`. Free-form fields (`notes`)
have no glossary; write whatever helps.

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

| Value         | Definition                                                                                                                                          | Example                                                              |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `battle`      | Regular combat / random encounters.                                                                                                                 | FF7 *Let the Battles Begin*, Chrono Trigger battle theme             |
| `boss`        | Major / climactic combat. Stakes feel higher than `battle`.                                                                                         | FF7 *One-Winged Angel*, FF6 *Dancing Mad*                            |
| `town`        | Safe, social, or pleasant locations — villages, castles, airships, festivals, peaceful interiors. Any scene where the player is at ease.            | FF6 *Tina*, FF7 *Costa del Sol*, Chrono Trigger *Millennial Fair*    |
| `exploration` | World map, open traversal, journeying between locations.                                                                                            | Chrono Trigger world map, FF8 *Blue Fields*                          |
| `dungeon`     | Hostile or dangerous non-combat zones — caves, ruins, military bases, enemy fortresses, final dungeons. Includes high-energy combat-zone traversal. | Chrono Trigger *Mt. Woe*, FF7 *Mako Reactor*, FF6 *Narshe Cliffs*    |
| `cutscene`    | Scripted story moments — character scenes, plot beats, revelations, farewells, celebrations. Mood_tags differentiate emotional tone.                | FF7 *Aerith's Theme*, FF8 *Eyes on Me* (instrumental), FF9 *Memoria* |
| `menu`        | Title screens, pause menus, character menus, equipment screens.                                                                                     | FF7 prelude/title, Persona menu themes                               |

---

## `energy` (pick one)

How much physical/sonic momentum is in the track?

| Value    | Definition                                                                  |
| -------- | --------------------------------------------------------------------------- |
| `low`    | Relaxed, slow tempo, sparse arrangement. Typically <90 BPM or very minimal. |
| `medium` | Moderate movement, present rhythm. Most exploration/town themes sit here.   |
| `high`   | Driving, dense, fast or punchy. Battle, chase, climax cues.                 |

---

## `mood_tags` (pick 2–4, array)

Emotional flavor. Pick the 2–4 most dominant. Order doesn't matter.

| Value         | Definition                                  | Distinguishing notes                                        |
| ------------- | ------------------------------------------- | ----------------------------------------------------------- |
| `tense`       | Anxious, anticipatory, unresolved.          | Pre-boss buildup, suspense scenes.                          |
| `triumphant`  | Victory, achievement, success.              | Fanfare, win cues. Past-tense win.                          |
| `melancholic` | Sad, sorrowful, weighted.                   | Heavier than `nostalgic`; grief vs wistfulness.             |
| `mysterious`  | Enigmatic, intriguing, unknown.             | Curiosity-tinged. Lighter than `dark`.                      |
| `peaceful`    | Calm, serene, content.                      | Small village themes, rest scenes.                          |
| `whimsical`   | Playful, light, charming.                   | Chocobo theme, comic relief.                                |
| `epic`        | Grand, large-scale, sweeping.               | Main themes, large-scale conflict.                          |
| `dark`        | Foreboding, menacing, oppressive.           | Villain themes, evil locations. Heavier than `mysterious`.  |
| `hopeful`     | Optimistic, forward-looking, gentle uplift. | Pre-victory, dawn scenes. Less concluded than `triumphant`. |
| `nostalgic`   | Wistful, memory-tinged, bittersweet.        | Childhood town themes. Lighter than `melancholic`.          |
| `urgent`      | Driving forward, time-pressure, momentum.   | Countdown/escape sequences. More action than `tense`.       |

---

## `dominant_instruments` (pick 1–4, array)

What instruments carry the track? "Dominant" means audibly leading, not just present.

| Value        | Definition                                                       |
| ------------ | ---------------------------------------------------------------- |
| `piano`      | Acoustic or electric piano carrying melody or harmony.           |
| `strings`    | Violin / viola / cello / contrabass section. Bowed strings.      |
| `brass`      | Trumpet, trombone, French horn, tuba.                            |
| `woodwinds`  | Flute, clarinet, oboe, bassoon.                                  |
| `choir`      | Wordless or non-language vocal ensemble (no lyrics).             |
| `synth`      | Electronic synthesizers — analog or digital, lead or pad.        |
| `percussion` | Drums, mallets, ethnic percussion when prominent (not just kit). |
| `guitar`     | Acoustic or electric guitar, lead or rhythm.                     |

---

## `genre` (pick one)

The overall stylistic idiom.

| Value        | Definition                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `orchestral` | Full classical instrumentation. Strings, brass, woodwinds, percussion.                                                             |
| `chiptune`   | 8-bit / 16-bit synthesis. NES / SNES / Game Boy idiom.                                                                             |
| `synthwave`  | 80s-style electronic, retro synth lead and arpeggios.                                                                              |
| `electronic` | Modern electronic production — programmed drums, synth leads, quantized loops. Not retro (synthwave) or purely textural (ambient). |
| `jazz`       | Jazz idiom — extended chords, swing feel, improvisational phrasing.                                                                |
| `rock`       | Rock-band instrumentation: drums + electric guitar + bass core.                                                                    |
| `folk`       | Folk or ethnic instrumentation — Celtic, Japanese traditional, world music elements.                                               |
| `ambient`    | Atmospheric / textural. Low rhythmic emphasis, soundscape-driven.                                                                  |

---

## Free-form fields (no glossary)

| Field   | Notes                                                                         |
| ------- | ----------------------------------------------------------------------------- |
| `notes` | Free-form description. Used as `keywords` in the training sidecar. Or `null`. |

---

## Ear Training Guide — Identifying Instruments

Label by **what the sound imitates**, not what physically produced it. If a PS1 sampled trumpet
plays a fanfare, that's `brass`. Only use `synth` when the sound is distinctly electronic and
not imitating any acoustic instrument.

**Practical test:** "If a real orchestra performed this part, what section would play it?"

### Easy identifications (start here)

- **Piano** — almost always identifiable. Percussive attack, sustain, wide range.
- **Guitar** — plucked or strummed, acoustic warmth or electric crunch. Distinctive attack.
- **Percussion** — rhythmic hits, kit drums, mallets, ethnic drums. Hard to miss.
- **Choir** — human voices, sustained vowels, layered harmonies. Distinct from all instruments.

### Strings vs. brass vs. woodwinds

These three are the hardest to separate in VGM. Focus on **attack character** (how the note starts):

| Instrument    | Attack                              | Typical role                                      |
| ------------- | ----------------------------------- | ------------------------------------------------- |
| **Strings**   | Smooth swell (bowed) or sharp pluck | Sustaining harmony underneath, lush wide pads     |
| **Brass**     | Punchy, buzzy, announces itself     | Blasting the main melody, fanfares, stabs         |
| **Woodwinds** | Airy, breathy onset                 | Solo melody lines, one voice singing over the mix |

Key distinctions:
- **Strings vs. brass:** Brass is bright, forward, and cuts through the mix. Strings are smoother, blend into harmony, sustain more naturally. In a boss theme, the thing blasting the melody in your face is usually brass; the thing swelling underneath is usually strings.
- **Woodwinds vs. strings:** Woodwinds sound like a single solo voice (one melody line). Strings come in sections (multiple voices, richer/wider sound). A lone singing melody over quiet backing is often flute or oboe. A lush wide pad underneath is almost always strings.

### When to use `synth`

Only when the sound has no acoustic equivalent — sweeping filter pads, electronic arpeggios, clearly digital leads. If you can name what real instrument it's trying to be, use that instrument instead.

### When unsure

Label only the 1–2 instruments you're confident about. Accurate sparse labels are better than noisy complete ones.

---

## Ear Training Guide — Distinguishing Genres

### Quick decision tree

```
Distorted electric guitar as a main element?
  → Yes: rock
  → No ↓

Rhythm swings (bouncy, uneven, triplet feel)?
  → Yes: jazz
  → No ↓

Sounds synthetic/programmed rather than acoustic?
  → Yes: electronic (or synthwave/ambient — see below)
  → No: probably orchestral (or folk if ethnic instruments dominate)
```

### Genre signatures

**Rock** — you feel it in your chest.
- Distorted electric guitar + heavy straight drum beat (kick-snare-kick-snare in 4/4)
- Rigid, driving rhythm you can headbang to
- Simple harmony: power chords, minor keys, 3–4 chords repeating
- Thick wall-of-sound; guitar distortion fills the frequency spectrum
- *Dead giveaway:* distorted guitar chugging on power chords

**Jazz** — sounds sophisticated and loose.
- Swing feel: notes aren't evenly spaced, they have a lilting long-short-long-short bounce
- Ride cymbal with "ding-da-ding-da-ding" pattern; drums are conversational, not mechanical
- Extended chords (7ths, 9ths) that would sound "wrong" in rock but resolve naturally here
- Clean separated instruments with space between them; piano comping (rhythmic chord stabs)
- *Dead giveaway:* walking bass line (bass playing a new note every beat, stepwise) + swing rhythm

**Electronic** — the sounds themselves are synthetic.
- Quantized rhythm: perfectly on-grid, no human looseness, loop-based patterns
- Synth leads and pads that don't imitate real instruments
- Programmed drums: too perfect, too crispy to be a human drummer
- Sidechain pumping, sweeping filters, glitchy textures
- *Dead giveaway:* programmed drums + clearly synthesized lead sounds

**Orchestral** — full ensemble, cinematic.
- Strings, brass, woodwinds, and percussion working together
- Dynamic range: quiet passages build to loud climaxes
- No electronic production artifacts; sounds like a live ensemble (even if sampled)

**Synthwave vs. electronic:** Synthwave has a specific 80s retro aesthetic — gated reverb drums, neon-vibes, retro arpeggios (think Hotline Miami). If it's electronic but doesn't feel retro, use `electronic`.

**Ambient vs. electronic:** Ambient is texture-led with low rhythmic emphasis — soundscapes, drones, atmosphere. If it has a clear beat and melodic synth leads, it's `electronic` not `ambient`.

**Folk:** Prominent ethnic or traditional instruments (Celtic fiddle, shamisen, pan flute, hand drums) that define the track's character rather than just adding color to an orchestral arrangement.
