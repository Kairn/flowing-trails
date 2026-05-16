"""Orchestrator — POST /compose endpoint.

Pipeline: parse user brief → retrieve melody → generate audio → score → retry if below threshold.
"""

from __future__ import annotations

import base64
import json
import time
from typing import TYPE_CHECKING

import modal
from pydantic import BaseModel, Field

from config import (
    APP_NAME,
    CORPUS_AUDIO_SAMPLE_RATE,
    DEFAULT_SIMILARITY_THRESHOLD,
    MAX_GENERATION_ATTEMPTS,
    MODAL_SECRET_NAME,
    MUSICGEN_APP_NAME,
    QDRANT_COLLECTION_NAME,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
)

if TYPE_CHECKING:
    from models import MusicSpec

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "anthropic>=0.40",
        "fastapi[standard]",
        "pydantic>=2.0",
        "structlog",
        "python-dotenv",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "numpy<2",
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "torchvision>=0.19.0",
        "laion-clap",
        "qdrant-client",
    )
    .add_local_python_source("config")
    .add_local_python_source("models")
    .add_local_python_source("otel_utils")
    .add_local_python_source("prompts")
    .add_local_python_source("claude_client")
    .add_local_python_source("clap_utils")
    .add_local_python_source("qdrant_utils")
    .add_local_python_source("retrieval")
    .add_local_python_source("scoring")
)

corpus_volume = modal.Volume.from_name(VOLUME_NAME)


class ComposeRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    tempo_bpm: int | None = None
    instruments: list[str] | None = None
    duration_seconds: float | None = Field(default=None, ge=5.0, le=30.0)
    key: str | None = None
    use_melody_conditioning: bool = False
    cfg_coeff: float | None = Field(default=None, ge=0.0, le=20.0)
    top_k: int | None = Field(default=None, ge=0, le=1000)
    temperature: float | None = Field(default=None, gt=0.0, le=5.0)
    model: str | None = Field(
        default=None,
        max_length=100,
        description="Claude model override for query parsing and spec refinement.",
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    volumes={VOLUME_MOUNT_PATH: corpus_volume},
    timeout=300,
)
@modal.fastapi_endpoint(method="POST")
def compose(request: ComposeRequest) -> dict:
    from otel_utils import (
        flush_telemetry,
        get_logger,
        get_tracer,
        setup_logging,
        setup_tracing,
    )

    setup_tracing()
    setup_logging()
    tracer = get_tracer()
    log = get_logger("orchestrator")

    try:
        with tracer.start_as_current_span("compose") as root_span:
            t0 = time.monotonic()
            trace_id = format(root_span.get_span_context().trace_id, "032x")
            log.info(
                "compose_start",
                trace_id=trace_id,
                description=request.description[:80],
            )

            model_kwargs = {"model": request.model} if request.model else {}

            with tracer.start_as_current_span("query_parse"):
                spec = parse_query(request, log, **model_kwargs)

            if request.use_melody_conditioning:
                with tracer.start_as_current_span("retrieval") as retrieval_span:
                    melody_wav, melody_sr = retrieve_melody(spec, retrieval_span, log)
            else:
                melody_wav, melody_sr = None, None
                log.info("melody_conditioning_disabled")

            query_vector = _embed_query(spec, log)

            gen_params = {
                "cfg_coeff": request.cfg_coeff,
                "top_k": request.top_k,
                "temperature": request.temperature,
            }

            audio_bytes, score, attempts = _generate_with_scoring(
                spec,
                melody_wav,
                melody_sr,
                query_vector,
                gen_params,
                tracer,
                log,
                **model_kwargs,
            )

            root_span.set_attribute("compose.attempts", attempts)
            root_span.set_attribute("compose.final_score", score)
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            log.info(
                "compose_complete",
                trace_id=trace_id,
                attempts=attempts,
                final_score=round(score, 4),
                duration_ms=duration_ms,
            )

            audio_b64 = (
                base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None
            )

            return {
                "spec": spec.model_dump(),
                "audio_b64": audio_b64,
                "audio_format": "wav_base64",
                "trace_id": trace_id,
                "score": round(score, 4),
                "attempts": attempts,
            }
    finally:
        flush_telemetry()


def _embed_query(spec: MusicSpec, log):
    """Compute CLAP text embedding of the spec for scoring."""
    from clap_utils import embed_text

    query_text = spec.clap_text()
    log.info("embed_query", query_text=query_text[:80])
    return embed_text(query_text)


def _generate_with_scoring(
    spec: MusicSpec,
    melody_wav: bytes | None,
    melody_sr: int | None,
    query_vector,
    gen_params: dict,
    tracer,
    log,
    **claude_kwargs,
) -> tuple[bytes | None, float, int]:
    """Generate audio in a retry loop, scoring each attempt against the query.

    On below-threshold scores, refines the spec via Claude before retrying.
    Returns (best_audio_bytes, best_score, total_attempts).
    """
    from scoring import score_generation

    best_audio: bytes | None = None
    best_score = -1.0
    history: list[dict] = []
    current_spec = spec

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        with tracer.start_as_current_span("music_generate") as gen_span:
            gen_span.set_attribute("generate.attempt", attempt)
            audio_bytes = generate_music(
                current_spec, melody_wav, melody_sr, gen_params, log
            )

            if audio_bytes is None:
                log.warning("generate_returned_none", attempt=attempt)
                return best_audio, best_score, attempt

            sim = score_generation(audio_bytes, query_vector)
            gen_span.set_attribute("generate.score", round(sim, 4))
        log.info(
            "score_attempt",
            attempt=attempt,
            score=round(sim, 4),
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
        )

        history.append({"spec": current_spec.model_dump(), "score": round(sim, 4)})

        if sim > best_score:
            best_score = sim
            best_audio = audio_bytes

        if sim >= DEFAULT_SIMILARITY_THRESHOLD:
            log.info("score_accepted", attempt=attempt, score=round(sim, 4))
            return best_audio, best_score, attempt

        if attempt < MAX_GENERATION_ATTEMPTS:
            current_spec = refine_spec(
                current_spec, sim, history, tracer, log, **claude_kwargs
            )
            query_vector = _embed_query(current_spec, log)

    log.info(
        "score_exhausted",
        attempts=MAX_GENERATION_ATTEMPTS,
        best_score=round(best_score, 4),
    )
    return best_audio, best_score, MAX_GENERATION_ATTEMPTS


def refine_spec(
    spec: MusicSpec,
    score: float,
    history: list[dict],
    tracer,
    log,
    **claude_kwargs,
) -> MusicSpec:
    """Refine a MusicSpec via Claude based on CLAP score feedback."""
    from claude_client import call_claude_json
    from models import MusicSpec as MusicSpecCls
    from prompts import SPEC_REFINER_SYSTEM

    with tracer.start_as_current_span("spec_refine") as span:
        span.set_attribute("refine.prior_score", round(score, 4))
        span.set_attribute("refine.attempt", len(history))

        if len(history) >= 2:
            prev_score = history[-2]["score"]
            span.set_attribute("refine.score_delta", round(score - prev_score, 4))

        user_message = json.dumps({"history": history})

        refined_data, usage = call_claude_json(
            system=SPEC_REFINER_SYSTEM,
            user_message=user_message,
            log=log,
            **claude_kwargs,
        )

        refined_data["style_hint"] = spec.style_hint
        refined_data["duration_seconds"] = spec.duration_seconds

        refined_spec = MusicSpecCls(**refined_data)
        log.info(
            "spec_refined",
            prior_score=round(score, 4),
            new_description=refined_spec.description[:80],
        )
        return refined_spec


def parse_query(request: ComposeRequest, log, **claude_kwargs) -> MusicSpec:
    """Parse raw user brief into a MusicSpec via Claude."""
    from claude_client import call_claude_json
    from models import MusicSpec
    from prompts import QUERY_PARSER_SYSTEM

    log.info("parse_query", description=request.description[:80])

    spec_data, usage = call_claude_json(
        system=QUERY_PARSER_SYSTEM,
        user_message=request.description,
        log=log,
        **claude_kwargs,
    )

    if request.tempo_bpm is not None:
        spec_data["tempo_bpm"] = request.tempo_bpm
    if request.instruments is not None:
        spec_data["instruments"] = request.instruments
    if request.duration_seconds is not None:
        spec_data["duration_seconds"] = request.duration_seconds
    if request.key is not None:
        spec_data["key"] = request.key

    spec = MusicSpec(**spec_data)
    log.info(
        "parse_query_done",
        genre=spec.genre,
        mood_tags=spec.mood_tags,
        energy=spec.energy,
    )
    return spec


def retrieve_melody(spec, span, log) -> tuple[bytes | None, int | None]:
    """Embed spec text via CLAP, search Qdrant, load top-1 audio as melody."""
    from retrieval.search import search

    query_text = spec.clap_text()
    log.info("retrieve_melody", query_text=query_text[:80])

    results = search(query_text)

    span.set_attribute("db.system", "qdrant")
    span.set_attribute("db.operation.name", "query")
    span.set_attribute("db.collection.name", QDRANT_COLLECTION_NAME)
    span.set_attribute("retrieval.result_count", len(results))

    if not results:
        log.info("retrieve_melody_no_results")
        return None, None

    top = results[0]
    span.set_attribute("retrieval.top_score", top.score)
    span.set_attribute("retrieval.top_category", top.category or "")
    log.info(
        "retrieve_melody_top",
        score=top.score,
        category=top.category,
        corpus_file_path=top.corpus_file_path,
    )

    if not top.corpus_file_path:
        return None, None

    melody_path = f"{VOLUME_MOUNT_PATH}/{top.corpus_file_path}"
    try:
        with open(melody_path, "rb") as f:
            melody_bytes = f.read()
        span.set_attribute("retrieval.melody_loaded", True)
        log.info("retrieve_melody_loaded", path=melody_path)
        return melody_bytes, CORPUS_AUDIO_SAMPLE_RATE
    except FileNotFoundError:
        span.set_attribute("retrieval.melody_loaded", False)
        log.warning("retrieve_melody_file_missing", path=melody_path)
        return None, None


def generate_music(
    spec, melody_wav: bytes | None, melody_sr: int | None, gen_params: dict, log
) -> bytes | None:
    """Generate audio from MusicSpec via the deployed MusicGen service."""
    from opentelemetry import trace

    from otel_utils import inject_context

    prompt = spec.to_prompt()
    log.info(
        "generate_music", prompt=prompt[:80], melody_conditioned=melody_wav is not None
    )

    active_params = {k: v for k, v in gen_params.items() if v is not None}

    cls = modal.Cls.from_name(MUSICGEN_APP_NAME, "MusicGenService")
    result = cls().generate.remote(
        prompt=prompt,
        duration_seconds=spec.duration_seconds,
        melody_wav=melody_wav,
        melody_sample_rate=melody_sr,
        trace_context=inject_context(),
        **active_params,
    )

    span = trace.get_current_span()
    span.set_attribute("gen_ai.system", "audiocraft")
    span.set_attribute("gen_ai.operation.name", "generate")
    span.set_attribute("gen_ai.request.model", result["model"])
    span.set_attribute("gen_ai.request.audio.duration_seconds", spec.duration_seconds)
    span.set_attribute("gen_ai.request.melody_conditioned", melody_wav is not None)
    span.set_attribute("gen_ai.response.decoder", result["decoder"])
    span.set_attribute("gen_ai.response.latency_ms", result["latency_ms"])

    log.info(
        "generate_music_done",
        model=result["model"],
        decoder=result["decoder"],
        latency_ms=result["latency_ms"],
    )
    return result["audio_bytes"]
