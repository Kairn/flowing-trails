# FlowingTrails — Service Design

## What This Document Covers
The inference and agentic pipeline: the orchestrator, MusicGen service, retrieval, CLAP embedding,
scoring loop, observability, and MCP exposure. This design is stable — implementation details may
evolve but the component contracts and platform choices are fixed.

---

## Platform Decisions

| Concern | Platform | Why |
|---|---|---|
| GPU compute (inference) | Modal | Serverless GPU, scales to zero, no cluster management |
| Model registry | Hugging Face Hub (private) | Versioned artifacts, tag-driven promotion, clean registry separation from serving |
| Reference audio corpus | Modal Volume `flowing-trails-corpus` | Co-located with inference, mounted by both indexing job and orchestrator |
| Text LLM | Anthropic Claude API | Managed, no deployment overhead; different system prompts for different roles |
| Vector database | Qdrant Cloud (free tier) | CLAP vectors + metadata only (no audio bytes), ~2MB for 1000 tracks |
| Observability | Grafana Cloud (OTLP direct) | Hosted Tempo + Prometheus + Grafana, no collector needed |
| CI | GitHub Actions | Eval pipeline, lint, test |

---

## Components

### Model Registry
HF Hub private repo `flowing-trails-musicgen`. Tags: `musicgen-vgm-v{N}` for fine-tuned,
`base-melody-large` for the base model reference. Inference pulls by tag via
`MODEL_TAG` env var at startup — swapping model versions is a redeploy with no code change.
Registry is the only handoff point between training and serving.

### MusicGen Inference Service
Modal endpoint, `POST /generate`. Accepts: prompt string, duration_seconds, optional melody_audio
bytes, optional seed. Returns: audio bytes + metadata (model version, decoder, latency).

Model: `facebook/musicgen-melody-large` (3.3B). Decoder: Multi-Band Diffusion (MBD) by default,
configurable flag for A/B benchmarking. VRAM: ~14GB total — fits on A10G (24GB) with headroom.

Also exposed as an MCP tool for external use (separate wrapper, not used internally).

### CLAP Embedding
`laion/clap-htsat-unfused` loaded in-process at orchestrator startup on CPU. Provides
`embed_text(str) → vector` and `embed_audio(bytes) → vector`. Not a separate service —
lightweight enough to run in-process. Used for retrieval queries and post-generation scoring.

Critical: CLAP expects 48kHz audio; MusicGen outputs 32kHz. Resample is applied explicitly before
any CLAP audio embedding call.

### Retrieval Service
Qdrant Cloud-backed. Offline indexing job runs CLAP audio embeddings over the corpus and upserts
with metadata (category, mood tags, energy, instrumentation, bpm hint, prompt, `corpus_file_path`). Content-hash-based
point IDs ensure idempotent re-indexing.

At query time: text query → CLAP embed → Qdrant top-N search → reference tracks. Top-1 result
includes `corpus_file_path` for melody conditioning; others return metadata only. Qdrant stores
vectors and metadata — no audio bytes.

Melody conditioning is disabled by default (`use_melody_conditioning=false`). The synthetic
corpus degrades quality when used as conditioning input — retrieval infra is retained for
future use with real reference tracks.

Also exposed as an MCP tool for external use.

### Claude API Client
Single Anthropic client, two distinct roles with different system prompts:
- **Query Parser**: raw user input → structured `MusicSpec` JSON, calibrated to MusicGen's actual
  conditioning strengths (style descriptors, energy, mood, instrumentation — not precise theory params)
- **Spec Refiner**: `MusicSpec` + similarity score + attempt history → revised `MusicSpec` JSON

Same API key, different prompts, no separate deployment.

### Orchestrator
Modal HTTP endpoint, `POST /compose`. Plain Python — no agent framework. Owns the agent loop,
tool dispatch, OTel span creation, trace context propagation across Modal boundaries, and response
assembly. Can also be run locally calling deployed Modal endpoints via HTTP — same code both ways.

### Observability
OTel instrumentation in every service from M0. All spans use GenAI semantic conventions where
applicable. Trace context propagated across Modal function boundaries via explicit `traceparent`
parameter injection (Modal functions are isolated processes — automatic propagation does not work).

OTel utility library shared by all services: traceparent inject/extract helpers, standard span
setup. OTLP exports directly to Grafana Cloud via two env vars — no collector, no code-level
exporter config.

---

## Agent Loop

```
POST /compose  {description, tempo_bpm?, instruments?, duration_seconds?, key?,
                use_melody_conditioning?, cfg_coeff?, top_k?, temperature?}
  │
  ├─ [span: query_parse]
  │    Claude API → MusicSpec JSON
  │
  ├─ (if use_melody_conditioning) [span: retrieval]
  │    CLAP.embed_text(spec.clap_text()) → Qdrant top-N → reference_tracks[]
  │    top-1: load audio bytes from Modal Volume via corpus_file_path
  │
  ├─ CLAP.embed_text(spec.clap_text()) → query_vector
  │
  ├─ loop (max 2 attempts):
  │    │
  │    ├─ [span: music_generate, attempt=N]
  │    │    Build text prompt from MusicSpec
  │    │    If melody conditioning: pass top-1 audio to generate_with_chroma
  │    │    MusicGen (MBD decoder) → audio_bytes
  │    │    Score: resample 32kHz→48kHz, CLAP audio embed, cosine_similarity
  │    │    if score ≥ threshold OR attempt == max: break
  │    │
  │    └─ [span: spec_refine, attempt=N]
  │         Claude API(MusicSpec, score, attempt_history) → revised MusicSpec
  │         Re-embed revised spec as new query_vector
  │
  └─ Response: {audio_bytes, final_spec, similarity_score, attempts, trace_id}
```

---

## Key Design Decisions

**MBD decoder as default.** Multi-Band Diffusion produces higher perceptual quality than the
EnCodec decoder at the cost of ~2× latency. We benchmark both — MBD is the production default,
EnCodec is the comparison baseline in the benchmark table.

**CLAP in-process, not a service.** The model is CPU-only and lightweight. Running it in-process
avoids a network hop on every scoring call in the retry loop. Worth revisiting only if the
orchestrator becomes multi-process.

**Hard retry cap at 2.** MusicGen's text conditioning has inherent limits — parameter revision
beyond 2 attempts yields diminishing returns. The retry loop demonstrates the agentic pattern;
it is not a guarantee of improvement.

**Similarity threshold empirically calibrated.** CLAP cross-modal similarity for melody-large
(text-only) centers around 0.45 (mean across 24 calibration prompts, range 0.37–0.58).
Accept threshold set to p25 (0.40) — triggers retries on the weakest ~25% of generations.
Threshold is set per model tag in `/eval/thresholds.json` — fine-tuned models need separate
calibration.

**MCP is external-facing only.** The orchestrator calls services directly via Python/HTTP.
MCP wrappers are standalone layers around the same deployed endpoints, demonstrable independently
from Claude Desktop or Claude Code. This keeps MCP in the portfolio without adding protocol
complexity to the core call path.

**Traceparent injection at Modal boundaries.** Modal functions are isolated processes — the OTel
SDK does not propagate context automatically. Every cross-Modal call passes `traceparent` as an
explicit parameter and reconstructs the span context on the receiving end.

---

## Infrastructure Story

Modal handles runtime GPU scheduling. The K8s story is documented as design artifacts in `/deploy`:

- `/deploy/k8s/` — production manifests: GPU resource requests, node affinity to GPU node pool,
  HPA, liveness/readiness probes, PodDisruptionBudget
- `/deploy/DESIGN.md` — VRAM budget breakdown, why A10G over T4, concurrency rationale, Modal vs
  K8s tradeoff, what a multi-replica production deployment would look like

Narrative: Modal was chosen for development velocity and cost efficiency; the K8s manifests show
what the production deployment would look like and the reasoning behind each resource decision.
