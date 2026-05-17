# FlowingTrails — Training and Custom Retrieval Design

## What This Document Covers
The fine-tuning pipeline, data preparation strategy, checkpoint management, and retrieval index
migration for the custom VGM model. Platform choices and key design decisions are locked; specific
hyperparameters and implementation details are deferred to the training session.

---

## Vision
Fine-tune `facebook/musicgen-melody-large` (3.3B) on a personal JRPG collection (~1000 tracks) to
produce a model with strong stylistic affinity for that specific collection. The model will have an
intentional stylistic bias — not a generic game music model, but one with a defined personality
shaped by a particular collection. This is a feature, not a limitation.

The same collection feeds both fine-tuning and the Qdrant retrieval index, making the full
pipeline domain-specific end-to-end.

---

## Platform Decisions

| Concern          | Platform                   | Why                                                                                                     |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------------------------- |
| Training compute | Modal A100-80GB (spot)     | Full fine-tune of 3.3B needs ~50-60GB VRAM; spot pricing reduces cost; Modal already used for inference |
| Training storage | Modal Volume               | Co-located with compute, large enough for chunked WAVs + checkpoints                                    |
| Data preparation | Local (CPU only)           | No GPU needed; simpler to debug; free to run                                                            |
| Model registry   | Hugging Face Hub (private) | Same as inference — tag-driven promotion, clean handoff point                                           |

---

## Training Data

### Source Collection
~1000 JRPG tracks, raw mp3. Covers the full mood spectrum: battle, boss, town, exploration,
dungeon, emotional/story beats, ambient, credits, menu. Vocals excluded manually before
any processing — no Demucs needed for clean instrumentals. Choir (wordless, chanted) is
retained; it functions as orchestral texture and does not cause generation artifacts.

### Human Labels
Eddy provides one label file per track before the data preparation pipeline runs. Fields:

| Field                  | Format              | Notes                                                                            |
| ---------------------- | ------------------- | -------------------------------------------------------------------------------- |
| `scene_type`           | string              | battle, boss, town, exploration, dungeon, emotional, ambient, credits, menu      |
| `energy`               | low / medium / high | Subjective gut read                                                              |
| `mood_tags`            | list (2–4)          | tense, triumphant, melancholic, mysterious, peaceful, whimsical, epic, etc.      |
| `dominant_instruments` | list (1–4 families) | piano, strings, brass, woodwinds, choir, synth, percussion, guitar               |
| `composer`             | string or null      | Where known — composers have distinguishable styles within JRPG                  |
| `notes`                | string or null      | Free-form: unusual structure, tempo changes, anything that would surprise Claude |

Fields intentionally excluded: `has_vocals` (excluded before labeling), `is_loop_friendly`
(irrelevant after chunking), `source_game` (titles within the same series are not meaningful
distinguishers; any useful origin context goes in `notes`).

### Machine-Generated Metadata
Applied automatically during data preparation:

| Field              | Tool                  |
| ------------------ | --------------------- |
| `bpm`              | librosa beat tracker  |
| `key`              | librosa key detection |
| `duration_seconds` | librosa / ffprobe     |

No automated instrument detection — available tools are unreliable for VGM and would add noise
to labels that Eddy can produce more accurately by ear.

### Training Caption
Claude generates the final text description from human labels + machine metadata. Example output:
*"Triumphant JRPG battle theme by Uematsu, fast tempo (~140 BPM), brass and strings with driving
percussion, high energy, tense undertone."* The caption is the training signal for the language
model — quality here directly affects fine-tuning quality.

---

## Data Preparation Pipeline

Runs locally. Input: mp3 files + label file. Output: directory of 30s WAV chunks + manifest JSON.
That directory is uploaded to Modal Volume and consumed directly by the training job.

```
mp3 files
label file (human labels, one row per track)
  │
  ├─ Audio analysis (librosa): BPM, key, duration
  ├─ Chunking: 30s segments, 5s overlap → WAV files at 32kHz
  │    Each chunk inherits its parent track's full metadata
  ├─ Claude API: generate training caption from all metadata
  └─ Manifest JSON: { "file": "chunk_001.wav", "description": "..." }
        (audiocraft training format)

Upload manifest + WAVs → Modal Volume
```

**Segment length rationale:** 30s is the standard for audiocraft fine-tuning. Longer hurts
training stability; shorter loses musical context. 5s overlap ensures musically coherent
boundary regions are not systematically discarded.

**Expected scale:** ~1000 tracks × ~5 segments = ~5000 training samples.

---

## Training Pipeline

### Setup
Training job runs on Modal, reads chunked WAVs + manifest from Modal Volume.

Fine-tuning scope:
- **Transformer (language model): fine-tuned.** This is where style is learned.
- **T5 text encoder: frozen.** Would need far more data to improve; risk of degradation.
- **EnCodec compression model: frozen.** Unfreezing risks audio codec quality degradation.

Fine-tuning method: **full fine-tune** (no LoRA). audiocraft does not have native LoRA support;
full fine-tuning is what the framework is designed for and produces a standard checkpoint that
loads identically to the base model.

### Checkpoint Strategy
Two simultaneous checkpoint concerns: **training resilience** and **checkpoint selection**.

**During training (resilience):**
- Full checkpoint (model weights + optimizer state) saved every 500 steps.
- Only the latest 2 full checkpoints retained — older ones overwritten. ~40GB rolling.
- On preemption, Modal restarts the job; it detects the latest checkpoint and resumes.

**Per epoch (selection candidates):**
- Model-only snapshot (no optimizer state, ~7GB fp16) saved at the end of each epoch.
- All epoch snapshots retained — these are the eval candidates.

**Expected output of a training run:** 2 epoch model snapshots + 2 rolling full checkpoints.
Total storage: ~55GB on Modal Volume.

### Hyperparameters
Not locked at design time — determined during training session. Starting points: low learning
rate (1e-5 range), 2 epochs, batch size 4–8 with gradient accumulation. Starting conservatively
avoids catastrophic forgetting; epochs can be extended if style shift is insufficient.

### Cost
A100-80GB spot on Modal: ~$3–4/hr. Estimated training time per run: 8–20 hours.
Budget per run: ~$30–80. Expect 3–5 experimental runs before a promotable checkpoint:
**~$100–400 total training compute.** Data prep is local — free.

---

## Checkpoint Promotion

After a training run completes:

1. **Automated eval:** run CLAP scoring across epoch snapshots using a fixed set of 15–20
   novel prompts (never used in training). Measure cosine similarity between prompt embedding
   and generated audio embedding. This is the same metric used in the inference pipeline.
2. **Human review:** Eddy listens to outputs from the top-scoring checkpoint(s). CLAP similarity
   is a proxy for coherence — the ear is the ground truth for whether it sounds good.
3. **Promotion:** winning checkpoint converted to `get_pretrained()`-compatible layout
   (state_dict.bin + compression_state_dict.bin + config) and pushed to HF Hub with tag
   `vgm-melody-v{N}`. Inference service picks it up via `MODEL_TAG` env var — no code change.

**Checkpoint format note:** audiocraft's trainer saves in its own format. A conversion step
is required before HF push to produce the layout expected by `MusicGen.get_pretrained()`.
This conversion is part of the training pipeline, not a manual step.

---

## Retrieval Index Migration

When the fine-tuned model is promoted, the Qdrant index is rebuilt from the personal collection,
replacing the synthetic bootstrap corpus.

**What gets indexed:** all ~1000 original tracks, **full-length** (not training chunks). Retrieval
conditioning works better from a full musical arc than from a 30s segment.

**Overlap with training data:** complete overlap — all 1000 tracks are both trained on and indexed.
This is intentional. Melody conditioning extracts chroma features (pitch-over-time only; timbre,
dynamics, and rhythm discarded). The model does not reproduce conditioning tracks; it generates
new audio shaped by their melodic contour. Conditioning on a trained track is not memorization —
it reinforces stylistic alignment, which is desirable.

**Index format:** same CLAP embedding + metadata schema as the current corpus. Re-indexing is
idempotent (content-hash point IDs). The service requires no code changes — swapping the index
contents is transparent to the orchestrator.

---

## Eval Strategy

No held-out tracks. Evaluation is text-prompt coherence via CLAP similarity — the score compares
a text embedding to a generated audio embedding. No reference track is involved.

**Ongoing eval:** same approach as current calibration — novel prompts → generate → CLAP score.
Compare score distributions between base model and fine-tuned model to verify style shift didn't
degrade prompt coherence.

**Style fidelity check:** CLAP similarity between generated audio and the corpus centroid (mean
embedding across all indexed tracks). A rising centroid similarity with maintained prompt
coherence indicates successful stylistic fine-tuning.

**Threshold recalibration:** `DEFAULT_SIMILARITY_THRESHOLD` in `config.py` was calibrated on
the base model. Re-run `scripts/run_calibration.py` + `scripts/analyze_thresholds.py` after
promotion and update `eval/thresholds.json` before updating CI.

---

## Key Design Decisions

**A100-80GB required for training, A10G sufficient for inference.**
Full fine-tuning at 3.3B with Adam optimizer states needs ~50–60GB VRAM. The inference service
stays on A10G — forward passes only need ~14GB. These are separate Modal app configurations.

**Full fine-tune over LoRA.**
audiocraft does not have native PEFT/LoRA support. Full fine-tuning is what the framework
is built for, produces a standard checkpoint format, and avoids custom adapter loading code.

**Data prep is local, not on Modal.**
The preparation pipeline is CPU-only. Running it locally avoids Modal overhead, is simpler to
debug interactively, and is free. The output artifact (WAVs + manifest) is uploaded once and
consumed by the training job.

**Stylistic bias is intentional.**
The model is trained on a single-collection JRPG corpus and is expected to be biased toward
that style. This is the goal — not a generic model, but one with a defined personality.
Diversity within the collection (battle, town, ambient, emotional) prevents mode collapse
without diluting the style signature.

**Checkpoint selection requires human review.**
CLAP similarity can shortlist candidates but cannot judge aesthetic quality. Eddy listens
to and approves the checkpoint before it is promoted. This gate is non-negotiable.

**PoC first, full data second.**
Before processing the full collection, the training pipeline will be verified end-to-end on
the base model (push vanilla weights → fine-tune on 1 sample → load in inference → confirm
generation works). Full data prep begins only after the pipeline is confirmed working.
