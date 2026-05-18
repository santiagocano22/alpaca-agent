"""Tests for src/llm/strategy_parser.py.

The LLMClient is mocked so no real API calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.llm.client import LLMClient, LLMError
from src.llm.strategy_parser import ParseError, parse_strategy
from src.strategy.schema import (
    ComparisonOp,
    EodPolicy,
    Horizon,
    Session,
    Timeframe,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(response: str) -> LLMClient:
    """Return a mock LLMClient whose complete() returns `response`."""
    client = LLMClient.__new__(LLMClient)
    client.complete = AsyncMock(return_value=response)
    return client


def _make_client_error(exc: Exception) -> LLMClient:
    """Return a mock LLMClient whose complete() raises `exc`."""
    client = LLMClient.__new__(LLMClient)
    client.complete = AsyncMock(side_effect=exc)
    return client


def _valid_strategy_json(**overrides) -> str:
    """Return a valid Strategy JSON string, with optional field overrides."""
    data = {
        "name": "RSI Test",
        "universe": ["QQQ"],
        "timeframe": "15Min",
        "session": "regular",
        "horizon": "intraday",
        "eod_policy": "close_all",
        "entry_rules": {
            "logic": "AND",
            "conditions": [
                {"left": {"type": "rsi", "params": {"period": 14}}, "op": "<", "right": 30}
            ],
        },
        "exit_rules": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": None,
            "trailing_stop_pct": None,
            "inverse_signal": None,
        },
        "position_sizing": {
            "max_position_pct": 10.0,
            "max_total_exposure_pct": 50.0,
            "max_concurrent_positions": 4,
        },
    }
    data.update(overrides)
    return json.dumps(data)


# ── Happy-path tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_valid_strategy_returns_strategy():
    """Valid JSON response → Strategy object with correct fields."""
    client = _make_client(_valid_strategy_json())
    strategy = await parse_strategy("buy QQQ when RSI 14 < 30", client=client)

    assert strategy.name == "RSI Test"
    assert strategy.universe == ["QQQ"]
    assert strategy.timeframe == Timeframe.M15
    assert strategy.session == Session.REGULAR
    assert strategy.horizon == Horizon.INTRADAY
    assert strategy.eod_policy == EodPolicy.CLOSE_ALL
    assert strategy.exit_rules.stop_loss_pct == 2.0
    assert strategy.position_sizing.max_position_pct == 10.0


@pytest.mark.asyncio
async def test_parse_strips_markdown_code_fence():
    """Response wrapped in ```json ... ``` must be parsed correctly."""
    wrapped = "```json\n" + _valid_strategy_json() + "\n```"
    client = _make_client(wrapped)
    strategy = await parse_strategy("some strategy", client=client)
    assert strategy.name == "RSI Test"


@pytest.mark.asyncio
async def test_parse_strips_plain_code_fence():
    """Response wrapped in ``` ... ``` (no language tag) must be parsed."""
    wrapped = "```\n" + _valid_strategy_json() + "\n```"
    client = _make_client(wrapped)
    strategy = await parse_strategy("some strategy", client=client)
    assert strategy.name == "RSI Test"


@pytest.mark.asyncio
async def test_parse_eod_policy_intraday_inferred():
    """LLM can set eod_policy; when it outputs 'close_all' for intraday, we accept it."""
    data = json.loads(_valid_strategy_json())
    data["eod_policy"] = "close_all"
    data["horizon"] = "intraday"
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert strategy.eod_policy == EodPolicy.CLOSE_ALL


@pytest.mark.asyncio
async def test_parse_swing_eod_policy_hold():
    """Swing horizon with hold eod_policy is valid."""
    data = json.loads(_valid_strategy_json())
    data["horizon"] = "swing"
    data["eod_policy"] = "hold"
    data["timeframe"] = "1H"
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert strategy.eod_policy == EodPolicy.HOLD
    assert strategy.horizon == Horizon.SWING


@pytest.mark.asyncio
async def test_parse_multi_condition_and():
    """Multi-condition AND entry_rules are parsed correctly."""
    data = json.loads(_valid_strategy_json())
    data["entry_rules"] = {
        "logic": "AND",
        "conditions": [
            {"left": {"type": "rsi", "params": {"period": 14}}, "op": "<", "right": 30},
            {"left": {"type": "price", "params": {}}, "op": ">", "right": 100.0},
        ],
    }
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert len(strategy.entry_rules.conditions) == 2


@pytest.mark.asyncio
async def test_parse_with_take_profit():
    """take_profit_pct is passed through when present."""
    data = json.loads(_valid_strategy_json())
    data["exit_rules"]["take_profit_pct"] = 5.0
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert strategy.exit_rules.take_profit_pct == 5.0


@pytest.mark.asyncio
async def test_parse_with_trailing_stop():
    """trailing_stop_pct is passed through when present."""
    data = json.loads(_valid_strategy_json())
    data["exit_rules"]["trailing_stop_pct"] = 1.5
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert strategy.exit_rules.trailing_stop_pct == 1.5


@pytest.mark.asyncio
async def test_parse_with_inverse_signal():
    """inverse_signal RuleGroup in exit_rules is handled."""
    data = json.loads(_valid_strategy_json())
    data["exit_rules"]["inverse_signal"] = {
        "logic": "AND",
        "conditions": [
            {"left": {"type": "rsi", "params": {"period": 14}}, "op": ">", "right": 70}
        ],
    }
    client = _make_client(json.dumps(data))
    strategy = await parse_strategy("test", client=client)
    assert strategy.exit_rules.inverse_signal is not None


# ── Error-path tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_non_json_raises_parse_error():
    """LLM returns prose instead of JSON → ParseError."""
    client = _make_client("Lo siento, no puedo parsear esa estrategia.")
    with pytest.raises(ParseError) as exc_info:
        await parse_strategy("garbage input", client=client)
    assert "JSON" in exc_info.value.user_message or "interpretar" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_parse_llm_error_json_raises_parse_error():
    """LLM returns {"error": "..."} → ParseError with message."""
    client = _make_client('{"error": "No entiendo la estrategia"}')
    with pytest.raises(ParseError) as exc_info:
        await parse_strategy("ambiguous input", client=client)
    assert "No entiendo la estrategia" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_parse_invalid_schema_raises_parse_error():
    """Valid JSON but fails Pydantic validation → ParseError with hint."""
    data = json.loads(_valid_strategy_json())
    data["timeframe"] = "invalid_tf"  # Not a valid Timeframe enum
    client = _make_client(json.dumps(data))
    with pytest.raises(ParseError) as exc_info:
        await parse_strategy("test", client=client)
    assert exc_info.value.user_message  # must contain a useful message


@pytest.mark.asyncio
async def test_parse_pydantic_coherence_violation_raises_parse_error():
    """PositionSizing coherence violation → ParseError."""
    data = json.loads(_valid_strategy_json())
    # max_position_pct(30) * max_concurrent(4) = 120 > max_total_exposure(50)
    data["position_sizing"] = {
        "max_position_pct": 30.0,
        "max_total_exposure_pct": 50.0,
        "max_concurrent_positions": 4,
    }
    client = _make_client(json.dumps(data))
    with pytest.raises(ParseError):
        await parse_strategy("test", client=client)


@pytest.mark.asyncio
async def test_parse_llm_network_error_propagates():
    """LLMError from the client propagates as-is (not wrapped in ParseError)."""
    client = _make_client_error(LLMError("network failure"))
    with pytest.raises(LLMError):
        await parse_strategy("test", client=client)


@pytest.mark.asyncio
async def test_parse_missing_stop_loss_raises_parse_error():
    """exit_rules without stop_loss_pct (required field) → ParseError."""
    data = json.loads(_valid_strategy_json())
    del data["exit_rules"]["stop_loss_pct"]
    client = _make_client(json.dumps(data))
    with pytest.raises(ParseError):
        await parse_strategy("test", client=client)
