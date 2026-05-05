"""Thin wrapper around the Anthropic SDK for structured JSON calls."""

from __future__ import annotations

import json
from typing import Any

import anthropic
from opentelemetry import trace

from config import CLAUDE_MODEL


def _set_genai_span_attrs(
    model: str,
    response_model: str,
    max_tokens: int,
    usage: anthropic.types.Usage,
) -> None:
    """Set OTel GenAI semantic convention attributes on the active span."""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.system", "anthropic")
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.response.model", response_model)
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)


def call_claude_json(
    system: str,
    user_message: str,
    *,
    model: str = CLAUDE_MODEL,
    max_tokens: int = 512,
    log=None,
) -> tuple[dict[str, Any], anthropic.types.Usage]:
    """Send a prompt to Claude and return parsed JSON plus usage metadata."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text
    data = json.loads(raw)

    _set_genai_span_attrs(model, response.model, max_tokens, response.usage)

    if log:
        log.info(
            "claude_call",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    return data, response.usage
