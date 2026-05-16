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
    resp.stop_reason = "end_turn"
    resp.id = "msg_01XFDUDYJgAACzvnptvVoYEL"
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


@patch("anthropic.Anthropic")
@patch("claude_client.trace.get_current_span")
def test_call_claude_json_sets_genai_span_attrs(mock_span_fn, mock_cls):
    mock_cls.return_value.messages.create.return_value = _mock_response({"ok": True})
    mock_span = MagicMock()
    mock_span_fn.return_value = mock_span

    from claude_client import call_claude_json

    call_claude_json(system="s", user_message="m")

    calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
    assert calls["gen_ai.system"] == "anthropic"
    assert calls["gen_ai.operation.name"] == "chat"
    assert calls["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert calls["gen_ai.response.model"] == "claude-sonnet-4-6"
    assert calls["gen_ai.request.max_tokens"] == 512
    assert calls["gen_ai.usage.input_tokens"] == 42
    assert calls["gen_ai.usage.output_tokens"] == 18
    assert calls["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert calls["gen_ai.response.id"] == "msg_01XFDUDYJgAACzvnptvVoYEL"
