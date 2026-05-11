"""MCP server exposing the FlowingTrails MusicGen service as a tool.

Standalone wrapper — not used by the orchestrator internally.
Calls the deployed Modal GPU endpoint for generation.
Run with: python -m mcp_servers.musicgen_server
Or configure as an MCP server in Claude Desktop / Claude Code.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import modal
from mcp.server.fastmcp import FastMCP

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import MUSICGEN_APP_NAME
from otel_utils import get_tracer, setup_tracing

setup_tracing()
_tracer = get_tracer("mcp.musicgen")

mcp = FastMCP("flowing-trails-musicgen")


@mcp.tool()
def generate_music(
    prompt: str,
    duration_seconds: float = 10.0,
    seed: int | None = None,
) -> dict:
    """Generate VGM-style audio from a text prompt using MusicGen.

    Calls the deployed MusicGen Modal service with Multi-Band Diffusion
    decoding. Returns metadata and base64-encoded WAV audio.

    Args:
        prompt: Text description of the desired music
                (e.g. "epic orchestral boss battle theme with choir")
        duration_seconds: Length of audio to generate in seconds (default 10.0)
        seed: Optional random seed for reproducible generation
    """
    from otel_utils import inject_context

    with _tracer.start_as_current_span(
        "mcp.generate_music",
        attributes={
            "mcp.tool": "generate_music",
            "gen_ai.system": "audiocraft",
            "gen_ai.operation.name": "generate",
            "gen_ai.request.audio.duration_seconds": duration_seconds,
        },
    ) as span:
        cls = modal.Cls.from_name(MUSICGEN_APP_NAME, "MusicGenService")
        result = cls().generate.remote(
            prompt=prompt,
            duration_seconds=duration_seconds,
            seed=seed,
            trace_context=inject_context(),
        )

        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("ascii")

        span.set_attribute("gen_ai.request.model", result["model"])
        span.set_attribute("gen_ai.response.decoder", result["decoder"])
        span.set_attribute("gen_ai.response.latency_ms", result["latency_ms"])

        return {
            "audio_base64": audio_b64,
            "sample_rate": result["sample_rate"],
            "model": result["model"],
            "decoder": result["decoder"],
            "duration_seconds": result["duration_seconds"],
            "latency_ms": result["latency_ms"],
        }


if __name__ == "__main__":
    mcp.run()
