"""Unit tests for claude_client wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _mock_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.text = json.dumps(data)
    usage = MagicMock(input_tokens=42, output_tokens=18)
    resp = MagicMock()
    resp.content = [block]
    resp.model = "claude-sonnet-4-6"
    resp.usage = usage
    return resp


@patch("anthropic.Anthropic")
def test_call_claude_json_returns_parsed_dict(mock_cls):
    payload = {"foo": "bar", "count": 3}
    mock_cls.return_value.messages.create.return_value = _mock_response(payload)

    from claude_client import call_claude_json

    data, usage = call_claude_json(system="test system", user_message="hello")

    assert data == payload
    assert usage.input_tokens == 42
    assert usage.output_tokens == 18

    call_args = mock_cls.return_value.messages.create.call_args
    assert call_args.kwargs["system"] == "test system"
    assert call_args.kwargs["messages"][0]["content"] == "hello"


@patch("anthropic.Anthropic")
def test_call_claude_json_logs_when_logger_provided(mock_cls):
    mock_cls.return_value.messages.create.return_value = _mock_response({"x": 1})
    log = MagicMock()

    from claude_client import call_claude_json

    call_claude_json(system="s", user_message="m", log=log)

    log.info.assert_called_once()
    call_kwargs = log.info.call_args.kwargs
    assert "input_tokens" in call_kwargs
    assert "output_tokens" in call_kwargs
