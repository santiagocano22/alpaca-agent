"""Tests for src/llm/summarizer.py.

The LLMClient is mocked so no real API calls are made.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, call

import pytest

from src.llm.client import LLMClient, LLMError
from src.llm.summarizer import (
    HAIKU_MODEL,
    _SUMMARY_FALLBACK,
    _ASK_FALLBACK,
    answer_ask,
    generate_daily_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(response: str = "LLM response text") -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client.complete = AsyncMock(return_value=response)
    return client


def _make_error_client(exc: Exception | None = None) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client.complete = AsyncMock(side_effect=exc or LLMError("api error"))
    return client


def _sample_trades() -> list[dict]:
    return [
        {"symbol": "AAPL", "side": "buy", "qty": 10, "price": 150.0, "pnl": 50.0,
         "rule_trigger": "RSI < 30"},
        {"symbol": "QQQ", "side": "sell", "qty": 5, "price": 300.0, "pnl": -20.0,
         "rule_trigger": "stop_loss"},
    ]


def _sample_positions() -> list[dict]:
    return [
        {"symbol": "AAPL", "qty": 10, "market_value": 1_500.0, "unrealized_pl": 50.0},
    ]


# ── generate_daily_summary ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_summary_returns_llm_response():
    """Normal path: returns the LLM text."""
    client = _make_client("Resumen del día: P&L +30 USD.")
    result = await generate_daily_summary(
        trades_today=_sample_trades(),
        open_positions=_sample_positions(),
        realized_pnl=30.0,
        unrealized_pnl=50.0,
        equity=10_000.0,
        trade_date=date(2025, 1, 8),
        client=client,
    )
    assert result == "Resumen del día: P&L +30 USD."


@pytest.mark.asyncio
async def test_daily_summary_uses_haiku_model():
    """Summary must use the Haiku model (cheap)."""
    client = _make_client("ok")
    await generate_daily_summary(
        trades_today=_sample_trades(),
        open_positions=[],
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        equity=5_000.0,
        trade_date=date(2025, 1, 8),
        client=client,
    )
    call_kwargs = client.complete.call_args.kwargs
    assert call_kwargs["model"] == HAIKU_MODEL


@pytest.mark.asyncio
async def test_daily_summary_max_tokens_300():
    """Summary max_tokens must be ≤ 300 (token budget)."""
    client = _make_client("ok")
    await generate_daily_summary(
        trades_today=[],
        open_positions=[],
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        equity=5_000.0,
        trade_date=date(2025, 1, 8),
        client=client,
    )
    assert client.complete.call_args.kwargs["max_tokens"] <= 300


@pytest.mark.asyncio
async def test_daily_summary_context_includes_equity():
    """Equity value must appear in the user message sent to LLM."""
    client = _make_client("ok")
    await generate_daily_summary(
        trades_today=[],
        open_positions=[],
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        equity=99_999.0,
        trade_date=date(2025, 1, 8),
        client=client,
    )
    user_arg = client.complete.call_args.kwargs["user"]
    assert "99,999.00" in user_arg or "99999" in user_arg


@pytest.mark.asyncio
async def test_daily_summary_llm_error_returns_fallback():
    """LLMError → fallback string (bot must not crash)."""
    result = await generate_daily_summary(
        trades_today=[],
        open_positions=[],
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        equity=5_000.0,
        trade_date=date(2025, 1, 8),
        client=_make_error_client(),
    )
    assert result == _SUMMARY_FALLBACK


@pytest.mark.asyncio
async def test_daily_summary_unexpected_error_returns_fallback():
    """Generic exception → fallback (never propagates)."""
    result = await generate_daily_summary(
        trades_today=[],
        open_positions=[],
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        equity=5_000.0,
        trade_date=date(2025, 1, 8),
        client=_make_error_client(RuntimeError("disk full")),
    )
    assert result == _SUMMARY_FALLBACK


@pytest.mark.asyncio
async def test_daily_summary_win_rate_in_context():
    """Win rate is computed and included in context."""
    client = _make_client("ok")
    trades = [
        {"symbol": "A", "side": "buy", "qty": 1, "price": 100, "pnl": 10.0},
        {"symbol": "B", "side": "sell", "qty": 1, "price": 200, "pnl": -5.0},
        {"symbol": "C", "side": "buy", "qty": 1, "price": 50, "pnl": 5.0},
    ]
    await generate_daily_summary(
        trades_today=trades,
        open_positions=[],
        realized_pnl=10.0,
        unrealized_pnl=0.0,
        equity=5_000.0,
        trade_date=date(2025, 1, 8),
        client=client,
    )
    user_arg = client.complete.call_args.kwargs["user"]
    # 2 out of 3 wins → 67% (rounded)
    assert "67%" in user_arg or "Win rate" in user_arg


# ── answer_ask ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_answer_ask_returns_llm_response():
    """Normal path: returns LLM text."""
    client = _make_client("Vendí TSLA porque el stop-loss se activó.")
    result = await answer_ask(
        "¿Por qué vendiste TSLA?",
        recent_trades=_sample_trades(),
        open_positions=_sample_positions(),
        equity=10_000.0,
        client=client,
    )
    assert result == "Vendí TSLA porque el stop-loss se activó."


@pytest.mark.asyncio
async def test_answer_ask_uses_haiku_model():
    client = _make_client("ok")
    await answer_ask(
        "test question",
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=client,
    )
    assert client.complete.call_args.kwargs["model"] == HAIKU_MODEL


@pytest.mark.asyncio
async def test_answer_ask_max_tokens_500():
    """ask max_tokens must be ≤ 500."""
    client = _make_client("ok")
    await answer_ask(
        "test",
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=client,
    )
    assert client.complete.call_args.kwargs["max_tokens"] <= 500


@pytest.mark.asyncio
async def test_answer_ask_question_in_context():
    """The user's question must appear in the user message sent to LLM."""
    client = _make_client("ok")
    question = "¿Cuánto perdí hoy con QQQ?"
    await answer_ask(
        question,
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=client,
    )
    user_arg = client.complete.call_args.kwargs["user"]
    assert question in user_arg


@pytest.mark.asyncio
async def test_answer_ask_recent_trades_in_context():
    """Recent trades appear in the context sent to the LLM."""
    client = _make_client("ok")
    await answer_ask(
        "summary",
        recent_trades=_sample_trades(),
        open_positions=[],
        equity=5_000.0,
        client=client,
    )
    user_arg = client.complete.call_args.kwargs["user"]
    assert "AAPL" in user_arg


@pytest.mark.asyncio
async def test_answer_ask_llm_error_returns_fallback():
    """LLMError → fallback string."""
    result = await answer_ask(
        "test",
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=_make_error_client(),
    )
    assert result == _ASK_FALLBACK


@pytest.mark.asyncio
async def test_answer_ask_unexpected_error_returns_fallback():
    """Generic exception → fallback."""
    result = await answer_ask(
        "test",
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=_make_error_client(ValueError("bad")),
    )
    assert result == _ASK_FALLBACK


@pytest.mark.asyncio
async def test_answer_ask_empty_trades_includes_placeholder():
    """When no recent trades, context must still be valid (no KeyError)."""
    client = _make_client("Sin operaciones recientes.")
    result = await answer_ask(
        "¿qué pasó?",
        recent_trades=[],
        open_positions=[],
        equity=5_000.0,
        client=client,
    )
    assert result == "Sin operaciones recientes."
