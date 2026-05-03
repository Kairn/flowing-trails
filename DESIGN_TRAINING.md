# FlowingTrails — Training and Custom Retrieval Design

## Status
**This design is a rough sketch.** The training story is deferred until the inference service is
complete and running. A full design pass is required before implementation begins. What's documented
here captures the known constraints and intentions so the design session has a solid starting point.

---

## Vision
Fine-tune `facebook/musicgen-melody` on a personal VGM collection (~800–1000 tracks) to produce
a model with stronger stylistic affinity for classic game music. Pair this with a custom Qdrant
index built from the same collection, replacing the bootstrap corpus.

This makes the full pipeline domain-specific end-to-end: the retrieval corpus, the conditioning
references, and the generative model all share the same style distribution.

---

## What We Know

### Registry Integration (stable, matches service design)
Fine-tuned checkpoints push to HF Hub repo `flowing-trails-musicgen` with tags `vgm-melody-v{N}`.
Inference service pulls by tag via `MODEL_TAG` env var — swapping from base to fine-tuned is a
single env var change and redeploy. No code changes required. Eval thresholds in
`/eval/thresholds.json` are recalibrated per model tag before updating CI.

### Training Data Contract (high level)
`audiocraft` fine-tuning expects a directory of 32kHz audio files and a JSON manifest mapping
each file to a text description. Text descriptions should be auto-generated via Claude from
available metadata (title, source, composer, BPM, detected instruments) — consistent with the
project's theme and higher quality than zero-shot captioning.

Same metadata ingestion pipeline feeds both training manifest generation and Qdrant indexing.

### Training and Retrieval Overlap (design intent)
The retrieval corpus and fine-tuning dataset should largely overlap. The model learns stylistic
affinity for the tracks in the index; retrieval returns those tracks as conditioning references.
Disjoint sets would mean the model generates audio guided by style cues it has no trained affinity
for.

Data split intent: ~800 tracks for training + Qdrant indexing; ~200 held out for eval prompt
inspiration (never indexed, never in training).

### Licensing
Commercially licensed soundtracks should not appear in distributed model weights. For a private
HF Hub repo this is a portfolio-internal concern, but the training data composition should be
documented clearly.

---

## Open Design Questions (requires full design pass)

- Training infrastructure: Modal GPU job vs local run for fine-tuning `audiocraft`
- Hyperparameter strategy: which params to tune, starting from `facebook/musicgen-melody` weights
- Training data pipeline: how to ingest, clean, and segment personal VGM tracks for `audiocraft`
- Metadata generation: Claude prompt design for producing high-quality text descriptions from
  audio metadata
- How to handle tracks with sparse metadata (titles only, no BPM or composer info)
- Checkpoint evaluation strategy: how to pick which checkpoint to promote vs discard
- Custom Qdrant index migration: switching from bootstrap corpus to personal collection without
  breaking the running service

---

## What to Build in /training (scaffold, deferred)
The `/training` directory will contain:
- `README.md`: data contract, manifest format, pipeline design (written at M5 as design artifact)
- `configs/`: hyperparameter configs (scaffold, not fully run)
- Training pipeline code: once designed

This directory exists as a portfolio signal — it shows where training feeds the registry even
before the training pipeline is fully implemented.
