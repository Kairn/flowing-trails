"""Orchestrator — POST /compose endpoint.

Single-pass pipeline: parse user brief → generate audio → return.
Retrieval (M2) and scoring loop (M3) are wired in later milestones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import modal
from pydantic import BaseModel, Field

from config import APP_NAME, MODAL_SECRET_NAME, MUSICGEN_APP_NAME

if TYPE_CHECKING:
    from models import MusicSpec

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic>=0.40",
        "pydantic>=2.0",
        "structlog",
        "python-dotenv",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
    )
    .add_local_python_source("config")
    .add_local_python_source("models")
    .add_local_python_source("otel_utils")
    .add_local_python_source("prompts")
    .add_local_python_source("claude_client")
)


class ComposeRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    tempo_bpm: int | None = None
    instruments: list[str] | None = None
    duration_seconds: float | None = Field(default=None, ge=5.0, le=30.0)
    key: str | None = None


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=120,
)
@modal.web_endpoint(method="POST")
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
            trace_id = format(root_span.get_span_context().trace_id, "032x")
            log.info(
                "compose_start",
                trace_id=trace_id,
                description=request.description[:80],
            )

            with tracer.start_as_current_span("query_parse"):
                spec = parse_query(request, log)

            with tracer.start_as_current_span("music_generate"):
                audio_bytes = generate_music(spec, log)

            log.info("compose_complete", trace_id=trace_id)

            return {
                "spec": spec.model_dump(),
                "audio_bytes": audio_bytes,
                "trace_id": trace_id,
            }
    finally:
        flush_telemetry()


def parse_query(request: ComposeRequest, log) -> MusicSpec:
    """Parse raw user brief into a MusicSpec via Claude."""
    from claude_client import call_claude_json
    from models import MusicSpec
    from prompts import QUERY_PARSER_SYSTEM

    log.info("parse_query", description=request.description[:80])

    spec_data, usage = call_claude_json(
        system=QUERY_PARSER_SYSTEM,
        user_message=request.description,
        log=log,
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


def generate_music(spec, log) -> bytes | None:
    """Generate audio from MusicSpec via the deployed MusicGen service."""
    from opentelemetry import trace

    prompt = spec.to_prompt()
    log.info("generate_music", prompt=prompt[:80])

    cls = modal.Cls.lookup(MUSICGEN_APP_NAME, "MusicGenService")
    result = cls().generate.remote(
        prompt=prompt,
        duration_seconds=spec.duration_seconds,
    )

    span = trace.get_current_span()
    span.set_attribute("gen_ai.system", "audiocraft")
    span.set_attribute("gen_ai.operation.name", "generate")
    span.set_attribute("gen_ai.request.model", result["model"])
    span.set_attribute("gen_ai.request.audio.duration_seconds", spec.duration_seconds)
    span.set_attribute("gen_ai.response.decoder", result["decoder"])
    span.set_attribute("gen_ai.response.latency_ms", result["latency_ms"])

    log.info(
        "generate_music_done",
        model=result["model"],
        decoder=result["decoder"],
        latency_ms=result["latency_ms"],
    )
    return result["audio_bytes"]
