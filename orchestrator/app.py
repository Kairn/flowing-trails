"""Orchestrator — POST /compose endpoint.

Single-pass pipeline: parse user brief → generate audio → return.
Retrieval (M2) and scoring loop (M3) are wired in later milestones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import modal
from pydantic import BaseModel, Field

from config import APP_NAME, MODAL_SECRET_NAME

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
    from opentelemetry import trace

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

            with tracer.start_as_current_span("query_parse") as parse_span:
                spec = parse_query(request, log)
                parse_span.set_attribute("music_spec.genre", spec.genre or "")
                parse_span.set_attribute(
                    "music_spec.duration_seconds", spec.duration_seconds
                )

            with tracer.start_as_current_span("music_generate") as gen_span:
                audio_bytes = generate_music(spec, log)
                gen_span.set_attribute(
                    "music_spec.duration_seconds", spec.duration_seconds
                )

            log.info("compose_complete", trace_id=trace_id)

            return {
                "spec": spec.model_dump(),
                "audio_bytes": audio_bytes,
                "trace_id": trace_id,
            }
    finally:
        flush_telemetry()


def parse_query(request: ComposeRequest, log) -> MusicSpec:
    """Parse raw user brief into a MusicSpec via Claude. Stub until M1-T3."""
    from models import MusicSpec

    log.info("parse_query_stub", description=request.description[:80])

    return MusicSpec(
        description=request.description,
        tempo_bpm=request.tempo_bpm,
        instruments=request.instruments or [],
        duration_seconds=request.duration_seconds or 10.0,
        key=request.key,
    )


def generate_music(spec, log) -> bytes | None:
    """Generate audio from MusicSpec via MusicGen service. Stub until M1-T4."""
    log.info("generate_music_stub", prompt=spec.to_prompt()[:80])
    return None
