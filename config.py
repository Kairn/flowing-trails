# Central reference for all shared constants across FlowingTrails services.
# Import with: from config import VOLUME_NAME, APP_NAME, etc.
#
# Modal usage: add `.add_local_python_source("config")` to every Modal image
# that needs these constants — this copies the file into the container so
# the import works identically in local scripts and deployed functions.

# ── Modal ─────────────────────────────────────────────────────────────────────

APP_NAME = "flowing-trails"
VOLUME_NAME = "flowing-trails-corpus"
VOLUME_MOUNT_PATH = "/corpus"
MODAL_SECRET_NAME = "flowing-trails-secrets"
GPU_CONFIG = "a10g"
MUSICGEN_APP_NAME = "flowing-trails-musicgen"

# ── Hugging Face ──────────────────────────────────────────────────────────────

HF_MUSICGEN_REPO = "flowing-trails-musicgen"

# Base model references — used by inference service before fine-tuned model exists
MUSICGEN_BASE_MODEL = "facebook/musicgen-melody"
MUSICGEN_BASE_MODEL_LARGE = "facebook/musicgen-melody-large"

# Tag convention: fine-tuned checkpoints use "musicgen-vgm-v{N}"
# Inference service pulls by MODEL_TAG env var — swap model = redeploy only
MUSICGEN_BASE_TAG = "base-melody"
MUSICGEN_BASE_LARGE_TAG = "base-melody-large"

# ── CLAP ──────────────────────────────────────────────────────────────────────

CLAP_MODEL = "laion/clap-htsat-unfused"

# MusicGen outputs at 32kHz; CLAP expects 48kHz — resample before scoring
MUSICGEN_SAMPLE_RATE = 32000
CLAP_SAMPLE_RATE = 48000

# ── Qdrant ────────────────────────────────────────────────────────────────────

QDRANT_COLLECTION_NAME = "flowing-trails-corpus"
QDRANT_VECTOR_SIZE = 512  # CLAP htsat-unfused embedding dimension
QDRANT_TOP_K = 3  # retrieval results returned to orchestrator

# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS_MANIFEST_PATH = f"{VOLUME_MOUNT_PATH}/corpus_manifest.json"
CORPUS_EMBEDDINGS_PATH = f"{VOLUME_MOUNT_PATH}/corpus_embeddings.json"
CORPUS_AUDIO_SAMPLE_RATE = 32000

# ── Orchestrator ──────────────────────────────────────────────────────────────

MAX_GENERATION_ATTEMPTS = 2

# Placeholder — calibrate empirically at M3 against 20+ prompts
# Target range per CLAP cross-modal literature; update in eval/thresholds.json per MODEL_TAG
DEFAULT_SIMILARITY_THRESHOLD = 0.30

# ── Claude API ────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"

# ── Observability ─────────────────────────────────────────────────────────────

OTEL_SERVICE_NAME = "flowing-trails"
