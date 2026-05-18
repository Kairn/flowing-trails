# FlowingTrails — Training and Custom Retrieval Design

## What This Document Covers
The fine-tuning pipeline, data preparation, checkpoint management, and retrieval index migration
for the custom VGM model. All architectural decisions locked; LR, epochs, and weight_decay
deferred to training session.

---

## Vision
Fine-tune `facebook/musicgen-melody-large` (3.3B) on a personal JRPG collection (~1000 tracks)
to produce a model with strong stylistic affinity for that collection. The model has an
intentional stylistic bias — defined personality shaped by a specific collection. Diversity
within the collection (battle, town, ambient, emotional) prevents mode collapse without
diluting the style signature.

The same collection feeds both fine-tuning and the Qdrant retrieval index.

---

## Platform Decisions

| Concern          | Platform                   |
| ---------------- | -------------------------- |
| Training compute | Modal A100-80GB (spot)     |
| Training storage | Modal Volume               |
| Data preparation | Local (CPU only)           |
| Model registry   | Hugging Face Hub (private) |

### Local Environment

Data prep and manifest validation depend on audiocraft v1.3.0 (torch 2.1, numpy 1.26.4),
which conflicts with the main project venv (torch 2.6). A separate venv at `training/.venv/`
holds the training-specific stack with CPU-only torch to avoid pulling CUDA locally.

### Source Data

Source audio and human labels live in `training/source/` (gitignored — private tracks
treated as secrets). Structure:

```
training/source/
├── labels.json         (human labels, one object per track)
├── track_001.mp3
├── track_002.mp3
└── ...
```

Prep pipeline reads from `training/source/`, outputs to `training/prepared/` (also
gitignored). Prepared data is uploaded to Modal Volume for training.

---

## Source Audio Requirements

| Requirement     | Value                                            |
| --------------- | ------------------------------------------------ |
| Format          | mp3 (or any ffmpeg-readable format)              |
| Minimum bitrate | 192 kbps (lower has audible artifacts post-prep) |
| Sample rate     | Any (44.1 / 48 kHz mix — normalized in prep)     |
| Vocals          | Excluded manually before labeling                |
| Choir           | Retained (wordless orchestral texture)           |

Non-uniform bitrate and sample rate is fine across the source set. The data prep step
normalizes every file to a uniform target (32 kHz mono 16-bit PCM, -14 LUFS).

---

## Labels and Metadata

### Human Labels (JSON array, one object per track)

| Field                  | Type                                     | Example                                                          |
| ---------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| `filename`             | string                                   | `"track_001.mp3"`                                                |
| `scene_type`           | string (enum, see glossary)              | `"battle"`                                                       |
| `energy`               | string (`low` / `medium` / `high`)       | `"high"`                                                         |
| `mood_tags`            | array of strings, 2–4 values (see glossary) | `["tense", "urgent", "triumphant"]`                           |
| `dominant_instruments` | array of strings, 1–4 values (see glossary) | `["brass", "strings", "percussion"]`                          |
| `genre`                | string (enum, see glossary)              | `"orchestral"`                                                   |
| `composer`             | string or null                           | `"Motoi Sakuraba"`                                               |
| `notes`                | string or null                           | `"driving 6/8 battle theme"` (used as `keywords`)                |

Full vocabulary for every enum field lives in `training/LABELS_GLOSSARY.md`. Mood vocabulary
extended beyond the original 7 to 11 values (`dark`, `hopeful`, `nostalgic`, `urgent` added)
for richer per-track captions during training-time augmentation.

Example record:

```json
{
  "filename": "track_001.mp3",
  "scene_type": "battle",
  "energy": "high",
  "mood_tags": ["tense", "urgent", "triumphant"],
  "dominant_instruments": ["brass", "strings", "percussion"],
  "genre": "orchestral",
  "composer": "Motoi Sakuraba",
  "notes": "driving 6/8 battle theme"
}
```

### Machine-Generated (during data prep)

| Field           | Source                                              |
| --------------- | --------------------------------------------------- |
| `bpm`           | `librosa.beat.beat_track`                           |
| `key`           | librosa chroma + Krumhansl-Schmuckler profile       |
| `duration`      | ffprobe                                             |
| `chroma_stable` | `audiocraft.modules.chroma.ChromaExtractor` (argmax) — eligibility flag for retrieval index |

### Training Description (template-generated)

Short tag-like sentence per track. No LLM required.

Template:
```
"{energy_adj} JRPG {scene_type} theme, {moods}"
```
- `energy_adj`: high-energy / mid-energy / low-energy
- `moods`: 2-mood form `"X and Y"`, 3+-mood form `"X, Y, and Z"`

Examples:
- `"High-energy JRPG battle theme, tense and triumphant"`
- `"Low-energy JRPG exploration theme, peaceful and melancholic"`
- `"Mid-energy JRPG dungeon theme, mysterious, dark, and ominous"`

Structured fields (genre, bpm, key, moods, instrument, keywords) live in the sidecar JSON
and are spliced into the description by audiocraft's training-time condition-merging
augmentation (p=0.25 merge, p=0.5 description-dropout, p=0.3 word-dropout). This produces
per-epoch caption variance from a single per-track description.

### Sidecar JSON (one per WAV, at `<wav_path>.json`)

```json
{
  "description": "High-energy JRPG battle theme, tense and triumphant",
  "genre": "orchestral",
  "bpm": "138",
  "key": "D minor",
  "moods": ["tense", "triumphant"],
  "instrument": "brass and strings",
  "keywords": "battle, urgent, sakuraba",
  "artist": "Motoi Sakuraba",
  "duration": 178.5,
  "sample_rate": 32000,
  "chroma_stable": true
}
```

---

## Data Preparation Pipeline

Runs locally, CPU-only. Input: source audio + label JSON. Output: normalized WAVs +
sidecar JSONs + training manifest. Upload to Modal Volume.

### Steps

1. **Audio normalize (ffmpeg, per track):**
   ```
   ffmpeg -i in.mp3 -ar 32000 -ac 1 \
     -af loudnorm=I=-14:TP=-1:LRA=11 \
     -c:a pcm_s16le out.wav
   ```
   Output: 32 kHz mono 16-bit PCM, **full-length**, -14 LUFS.

2. **Machine metadata extraction:** bpm, key, duration.

3. **Chroma stability scoring:** per-track stability score → `chroma_stable` flag.

4. **Description template:** assemble from labels.

5. **Sidecar JSON:** write per WAV.

6. **Manifest build:**
   ```
   python -m audiocraft.data.audio_dataset <wav_folder> egs/vgm/data.jsonl.gz
   ```

7. **Upload** WAVs + sidecars + manifest to Modal Volume.

### No Pre-Chunking

audiocraft's `AudioDataset` random-crops 30s windows on-the-fly per epoch. Pre-chunking
to fixed 30s segments would lose this variance and inflate disk usage. Feed full-length
normalized WAVs; let the dataloader handle cropping.

---

## Training Pipeline

### audiocraft Pin

| Dependency | Version                                                  |
| ---------- | -------------------------------------------------------- |
| audiocraft | v1.3.0 (SHA `72cb16f9fb239e9cf03f7bd997198c7d7a67a01c`)  |
| torch      | 2.1.0+cu121                                              |
| numpy      | 1.26.4 (hard pin)                                        |
| xformers   | <0.0.23                                                  |
| Python     | 3.10                                                     |
| Modal base | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`            |

Training image is **separate** from inference image (inference uses torch 2.6).

### Fine-tuning Scope

- **Transformer LM:** fine-tuned.
- **T5 text encoder:** frozen.
- **EnCodec compression model:** frozen.
- **Method:** full fine-tune. LoRA rejected — no published MusicGen LoRA vs full-FT quality
  comparison, and merging LoRA weights back into audiocraft's `get_pretrained()` format
  is unsolved (HF→audiocraft converter does not exist upstream).

### Required Training Config

| Setting                      | Value      | Purpose                                                   |
| ---------------------------- | ---------- | --------------------------------------------------------- |
| `autocast`                   | true       | Mixed precision required to fit 80 GB                     |
| `autocast_dtype`             | bf16       | fp16 grad scaler unstable at 3.3B                         |
| `checkpointing`              | torch      | Gradient checkpointing required to fit 80 GB              |
| `dataset.batch_size`         | 1–2        | Per-step limit from activation memory                     |
| `optim.grad_accum`           | 16–32      | Effective batch matches Meta's per-GPU                    |
| `dataset.segment_duration`   | 30         | Pretrained context max; do not change                     |
| `optim.ema.use`              | false      | Avoids bug #550 (EMA + T5 fine-tune); saves VRAM          |
| `optim.updates_per_epoch`    | 100–200    | ~5min checkpoint cadence (vs 25min at default)            |
| `checkpoint.save_last`       | true       |                                                           |
| `checkpoint.keep_last`       | 2          | ~80 GB rolling storage                                    |

Deferred to training session: LR, total epochs, weight_decay, scheduler.

### Checkpoint Strategy

- Dora signature-based resume: identical `dora run` command picks up from latest
  `checkpoint.th`. No `--resume` flag exists; do not pass `--clear`.
- Persistent Modal Volume mounted at `AUDIOCRAFT_DORA_DIR`.
- On Modal container exit, `modal.Volume.commit()` flushes the latest checkpoint.

Resume restores: model, optimizer, lr_scheduler, scaler, best_state, epoch counter, history.
Does NOT restore: dataloader cursor, mid-step grad-accum, RNG. Each resume restarts the
current epoch from update 0 — losing ≤ updates_per_epoch worth of compute per preemption.

### Cost

A100-80GB spot on Modal: ~$3–4/hr. Estimated training time per run: 8–20 hours.
Expect 3–5 experimental runs before a promotable checkpoint: **~$100–400 total**.

---

## Checkpoint Promotion

After a training run:

1. **Convert** (audiocraft has the official script):
   ```python
   from audiocraft.utils import export
   export.export_lm(xp.folder / 'checkpoint.th', '/out/state_dict.bin')
   export.export_pretrained_compression_model(
       'facebook/encodec_32khz', '/out/compression_state_dict.bin')
   ```
2. **Upload** both files to private HF Hub repo with tag `vgm-melody-v{N}`.
3. **Automated eval:** generate from 30–40 fixed eval prompts (held out from training)
   with deterministic seeds across all candidate checkpoints.
4. **Human review:** blind A/B/C listening across top candidates; rank per prompt.
5. **Promotion:** chosen tag set as `MODEL_TAG` in inference service; redeploy.

### HF Hub Repo Layout

```
state_dict.bin                  (~6 GB fp16)
compression_state_dict.bin      (~1 KB pointer to facebook/encodec_32khz)
README.md                       (optional)
```

T5 encoder is NOT bundled — `T5Conditioner` fetches it separately at load time.

### Conversion Gotchas

- `continue_from` does NOT inherit config — must explicitly set `model/lm/model_scale=large`
  and `conditioner=text2music`. Mismatch → strict `load_state_dict` shape error.
- The exported `compression_state_dict.bin` is a pointer dict, not the EnCodec weights —
  the loader pulls EnCodec separately at load time.

---

## Retrieval Index Migration

Gated rollout. The synthetic corpus stays in place until an A/B test confirms real-audio
melody conditioning provides quality benefit (theory predicts modest gain at best;
empirical validation required).

### A/B Test Protocol

Runs after the first fine-tune is promoted, before any full index migration.

- Index 50 `chroma_stable=true` tracks (15 battle, 15 town, 10 boss, 10 emotional).
- 20 fixed prompts × 3 arms with identical seeds:
  - **A:** text-only `generate()`
  - **B:** `generate_with_chroma()` with top-1 retrieved real track
  - **C:** `generate_with_chroma()` with random real track (control)
- Score: CLAP-to-prompt similarity + blind human listening preference.

**Decision rule:** build full index iff B beats both A and C on both metrics.

### If A/B Passes

- Index all `chroma_stable=true` tracks at full length.
- Same Qdrant schema as current bootstrap corpus (CLAP embedding + metadata).
- Content-hash IDs ensure idempotent re-indexing.
- No inference code change — `MODEL_TAG` swap + `use_melody_conditioning=true` default.

### If A/B Fails

Melody conditioning stays disabled. Retrieval infrastructure retained for future use cases
(style search, sample browsing, training-time augmentation).

---

## Eval Strategy

### Per-Run Checkpoint Selection

- **30–40 fixed eval prompts** spanning all scene categories; held out from training.
- Generate from each epoch checkpoint deterministically (same seeds, same gen params).
- **Primary signal:** blind human listening, ranked per prompt, aggregated.
- **Secondary signal:** CLAP-to-prompt score — used only as a tiebreaker and regression detector.

### Ongoing Eval (post-promotion)

- Re-run `scripts/run_calibration.py` + `scripts/analyze_thresholds.py` with new model.
- Update `eval/thresholds.json` and `DEFAULT_SIMILARITY_THRESHOLD` in `config.py`.
- Compare prompt-coherence distribution vs base model — confirm style shift didn't
  degrade text following.

### Class Imbalance

Imbalance across `scene_type` is accepted as a feature — the corpus distribution IS the
style we are modeling. Manifest writes `weight: 1.0` for every entry; per-track weight
support is latent for future use if needed.

---

## PoC Phasing

Validate the full path on minimal scope before committing real compute. See M6 tasks in
PROGRESS.md for the session-level breakdown.

### Phase A — Data Prep PoC (local)

- Set up `training/.venv/` with audiocraft v1.3.0 + CPU-only torch.
- Hand-label 5–10 tracks covering main scene types → `training/source/labels.json`.
- Build and run full data prep pipeline (normalize → metadata → chroma → description →
  sidecar → manifest).
- **Pass:** manifest loads under `audiocraft.data.audio_dataset`; every entry has required
  fields; `MusicDataset.__getitem__` iterates without error.

### Phase B — Conversion Round-trip (Modal, ~15 min)

- Run `export.export_lm` on vanilla `musicgen-melody-large` weights.
- Upload `state_dict.bin` + `compression_state_dict.bin` to private HF repo
  (`KairnAI/flowing-trails-musicgen`), tag `base-melody-large-roundtrip`.
- Inference service loads via `MODEL_TAG` + `snapshot_download()`; generates a sample.
- **Pass:** output indistinguishable from current vanilla generation (same seed).

### Phase C — Training PoC (Modal A100-80GB, ~15 min)

Uses small model first to validate training code path before burning compute on large.

```
dora run solver=musicgen/musicgen_base_32khz \
  model/lm/model_scale=small \
  continue_from=//pretrained/facebook/musicgen-small \
  conditioner=text2music \
  dataset.batch_size=8 dataset.segment_duration=10 \
  optim.epochs=2 optim.updates_per_epoch=20 \
  optim.ema.use=false \
  checkpoint.save_last=true checkpoint.save_every=1 checkpoint.keep_last=3 \
  generate.every=1 generate.num_samples=2
```

- Mid-run SIGKILL → relaunch identical command → confirm "Restoring weights and history".
- Export checkpoint, push to HF, load in inference.
- **Pass:** training runs end-to-end; resume works; exported checkpoint loads in inference.

### Phase D — Full Run

- Switch to `musicgen-melody-large` + full data + locked training config.
- Multiple experimental runs; production hyperparams determined empirically.

---

## Key Design Decisions

- **Full FT, not LoRA.** No published MusicGen LoRA vs full-FT quality comparison;
  HF→audiocraft checkpoint merge is unsolved.
- **A100-80GB with mandatory memory levers.** bf16 + gradient checkpointing + batch 1–2 +
  grad_accum 16–32. Multi-GPU buys speed only, not quality at this dataset size.
- **No pre-chunking.** audiocraft's dataloader random-crops 30s on-the-fly; better
  generalization than fixed chunks, simpler pipeline, less disk.
- **Template captions, no LLM.** audiocraft's condition-merging augmentation produces
  per-epoch variance from a single per-track description.
- **Stylistic bias is intentional.** No class balancing.
- **Human is primary eval judge.** CLAP score is a tiebreaker only.
- **Retrieval index migration is A/B-gated.** Theory predicts modest gain at best.
- **Checkpoint promotion path validated on base weights before any training.**
