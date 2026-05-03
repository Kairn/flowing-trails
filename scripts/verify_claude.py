"""Verify Claude API connection and JSON parsing.

Sends a test prompt asking for structured JSON, parses the response,
and wraps the call in an OTel span so it appears in Grafana Tempo.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from opentelemetry.trace import StatusCode

from config import CLAUDE_MODEL
from otel_utils import (
    flush_telemetry,
    get_logger,
    get_tracer,
    setup_logging,
    setup_tracing,
)

setup_tracing()
setup_logging()
log = get_logger("verify-claude")
tracer = get_tracer("verify-claude")

TEST_PROMPT = (
    "You are a music metadata generator. Return ONLY valid JSON, no markdown fences.\n"
    'Respond with a JSON object containing: "title" (string), "mood" (string), '
    '"bpm" (integer), "instruments" (list of strings). '
    "Generate metadata for a short fantasy RPG town theme."
)


def main() -> None:
    client = anthropic.Anthropic()

    with tracer.start_as_current_span("verify-claude-api") as span:
        span.set_attribute("test.purpose", "m0-t8-claude-verify")
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.request.model", CLAUDE_MODEL)

        log.info("Sending test prompt", model=CLAUDE_MODEL)
        t0 = time.monotonic()

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

        latency_ms = (time.monotonic() - t0) * 1000
        raw_text = response.content[0].text

        span.set_attribute("gen_ai.response.model", response.model)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
        span.set_attribute("test.latency_ms", round(latency_ms, 1))

        parsed = json.loads(raw_text)

        required_keys = {"title", "mood", "bpm", "instruments"}
        missing = required_keys - parsed.keys()
        if missing:
            raise ValueError(f"Missing keys in response: {missing}")

        span.set_status(StatusCode.OK)
        trace_id = format(span.get_span_context().trace_id, "032x")

    flush_telemetry()
    log.info(
        "Claude API verified",
        trace_id=trace_id,
        model=response.model,
        latency_ms=round(latency_ms, 1),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        parsed_title=parsed["title"],
    )


if __name__ == "__main__":
    main()
