"""Thin wrapper around the Anthropic SDK for structured JSON calls."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from config import CLAUDE_MODEL


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

    if log:
        log.info(
            "claude_call",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    return data, response.usage
