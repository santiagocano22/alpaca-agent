"""Tests for src/telegram_bot/commands.py.

python-telegram-bot objects (Update, Context) are fully mocked — no real
Telegram server is contacted.

Pattern:
  - _make_update(chat_id, args) → mock Update
  - _make_context(state, deps, args) → mock Context with bot_data wired up
  - handler(update, context) is called directly (it's a regular async function)
  - assert context.bot.send_message was called with the expected text fragment
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telegram_bot.bot import BotDeps, BotState
from src.telegram_bot.commands import (
    handle_ask,
    handle_cancel,
    handle_closeall,
    handle_confirm,
    handle_help,
    handle_pause,
    handle_positions,
    handle_resume,
    handle_setlimit,
    handle_start,
    handle_status,
    handle_strategy,
)
from src.strategy.schema import (
    ComparisonOp,
    Condition,
    EodPolicy,
    ExitRules,
    Horizon,
    IndicatorRef,
    IndicatorType,
    PositionSizing,
    RuleGroup,
    Session as StrategySession,
    Strategy,
    Timeframe,
)

# ── Constants ─────────────────────────────────────────────────────────────────

AUTHORIZED_CHAT_ID = 987654321
UNAUTHORIZED_CHAT_ID = 111111111

# ── Strategy factory ──────────────────────────────────────────────────────────


def _make_strategy(name: str = "Test RSI") -> Strategy:
    return Strategy(
        name=name,
        universe=["AAPL"],
        timeframe=Timeframe.M15,
        session=StrategySession.REGULAR,
        horizon=Horizon.INTRADAY,
        eod_policy=EodPolicy.CLOSE_ALL,
        entry_rules=RuleGroup(
            logic="AND",
            conditions=[
                Condition(
                    left=IndicatorRef(type=IndicatorType.RSI, params={"period": 14}),
                    op=ComparisonOp.LT,
                    right=30.0,
                )
            ],
        ),
        exit_rules=ExitRules(stop_loss_pct=2.0),
        position_sizing=PositionSizing(
            max_position_pct=10.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=4,
        ),
    )


# ── Mock factories ────────────────────────────────────────────────────────────


def _make_update(*, chat_id: int = AUTHORIZED_CHAT_ID, text: str = "/cmd") -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.text = text
    return update


def _make_context(
    state: BotState,
    deps: BotDeps,
    args: list[str] | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.bot_data = {"state": state, "deps": deps}
    ctx.bot.send_message = AsyncMock()
    ctx.args = args or []
    return ctx


def _make_alpaca(
    *,
    positions=None,
    portfolio_value: float = 10_000.0,
    buying_power: float = 5_000.0,
) -> MagicMock:
    alpaca = MagicMock()
    account = MagicMock()
    account.portfolio_value = portfolio_value
    account.buying_power = buying_power
    alpaca.get_account = AsyncMock(return_value=account)
    alpaca.get_positions = AsyncMock(return_value=positions or [])
    alpaca.close_position = AsyncMock()
    return alpaca


def _make_position(symbol: str = "AAPL", unrealized_pl: float = 50.0) -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = 10.0
    pos.avg_entry_price = 150.0
    pos.market_value = 1_500.0
    pos.unrealized_pl = unrealized_pl
    return pos


def _make_session_factory(session=None):
    """Return a fake async session factory (async context manager)."""
    @asynccontextmanager
    async def _factory():
        mock_session = session or MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        yield mock_session
    return _factory


def _make_deps(
    *,
    authorized_chat_id: int = AUTHORIZED_CHAT_ID,
    alpaca=None,
    llm_client=None,
    session_factory=None,
) -> BotDeps:
    return BotDeps(
        alpaca=alpaca or _make_alpaca(),
        session_factory=session_factory or _make_session_factory(),
        llm_client=llm_client or MagicMock(),
        authorized_chat_id=authorized_chat_id,
        pending_strategy_ttl_seconds=600,
    )


def _sent_text(context: MagicMock) -> str:
    """Return the concatenated text of all send_message calls."""
    return " ".join(
        call.kwargs.get("text", "") or (call.args[1] if len(call.args) > 1 else "")
        for call in context.bot.send_message.call_args_list
    )


# ── Authorization tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unauthorized_chat_gets_neutral_response():
    state = BotState()
    deps = _make_deps()
    update = _make_update(chat_id=UNAUTHORIZED_CHAT_ID)
    context = _make_context(state, deps)

    await handle_start(update, context)

    context.bot.send_message.assert_called_once()
    call_kwargs = context.bot.send_message.call_args.kwargs
    assert call_kwargs["text"] == "👋"


@pytest.mark.asyncio
async def test_unauthorized_does_not_execute_command():
    """Unauthorized caller must not change state."""
    state = BotState()
    deps = _make_deps()
    update = _make_update(chat_id=UNAUTHORIZED_CHAT_ID)
    context = _make_context(state, deps, args=[])

    original_paused = state.bot_paused
    await handle_pause(update, context)
    assert state.bot_paused == original_paused  # unchanged


# ── /start ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_shows_welcome():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_start(update, context)

    text = _sent_text(context)
    assert "Bot" in text or "bot" in text
    assert "ACTIVO" in text or "PAUSADO" in text


@pytest.mark.asyncio
async def test_start_shows_strategy_name():
    state = BotState(active_strategy=_make_strategy("My Strat"))
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_start(update, context)

    assert "My Strat" in _sent_text(context)


@pytest.mark.asyncio
async def test_start_shows_paused_when_paused():
    state = BotState(bot_paused=True)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_start(update, context)

    assert "PAUSADO" in _sent_text(context)


# ── /status ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_shows_market_state():
    state = BotState(market_state="ACTIVE")
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_status(update, context)

    assert "ACTIVE" in _sent_text(context)


@pytest.mark.asyncio
async def test_status_shows_paused():
    state = BotState(bot_paused=True, market_state="IDLE")
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_status(update, context)

    assert "PAUSADO" in _sent_text(context)


@pytest.mark.asyncio
async def test_status_shows_equity():
    state = BotState()
    deps = _make_deps(alpaca=_make_alpaca(portfolio_value=12_345.67))
    update = _make_update()
    context = _make_context(state, deps)

    await handle_status(update, context)

    assert "12,345.67" in _sent_text(context) or "12345" in _sent_text(context)


# ── /positions ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_positions_no_positions():
    state = BotState()
    deps = _make_deps(alpaca=_make_alpaca(positions=[]))
    update = _make_update()
    context = _make_context(state, deps)

    await handle_positions(update, context)

    assert "No hay" in _sent_text(context) or "no hay" in _sent_text(context).lower()


@pytest.mark.asyncio
async def test_positions_shows_symbols():
    pos = _make_position("TSLA", unrealized_pl=100.0)
    state = BotState()
    deps = _make_deps(alpaca=_make_alpaca(positions=[pos]))
    update = _make_update()
    context = _make_context(state, deps)

    await handle_positions(update, context)

    assert "TSLA" in _sent_text(context)


# ── /strategy ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_strategy_no_args_no_strategy():
    state = BotState(active_strategy=None)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=[])

    await handle_strategy(update, context)

    text = _sent_text(context).lower()
    assert "no hay" in text or "ninguna" in text


@pytest.mark.asyncio
async def test_strategy_no_args_shows_active_strategy():
    state = BotState(active_strategy=_make_strategy("My RSI"))
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=[])

    await handle_strategy(update, context)

    assert "My RSI" in _sent_text(context)


@pytest.mark.asyncio
async def test_strategy_with_args_parses_and_stores_pending():
    from src.llm.client import LLMClient
    new_strategy = _make_strategy("Parsed RSI")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete = AsyncMock()  # not called directly; parse_strategy is patched

    state = BotState()
    deps = _make_deps(llm_client=mock_llm)
    update = _make_update()
    context = _make_context(state, deps, args=["buy", "AAPL"])

    with patch(
        "src.telegram_bot.commands.parse_strategy",
        new=AsyncMock(return_value=new_strategy),
    ):
        await handle_strategy(update, context)

    assert state.pending_strategy is new_strategy
    assert state.pending_strategy_expires_at is not None
    assert "Parsed RSI" in _sent_text(context)
    assert "/confirm" in _sent_text(context)


@pytest.mark.asyncio
async def test_strategy_parse_error_shows_user_message():
    from src.llm.strategy_parser import ParseError

    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["gibberish"])

    with patch(
        "src.telegram_bot.commands.parse_strategy",
        new=AsyncMock(side_effect=ParseError("Estrategia ambigua")),
    ):
        await handle_strategy(update, context)

    assert "Estrategia ambigua" in _sent_text(context)
    assert state.pending_strategy is None


# ── /confirm ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_activates_pending_strategy():
    pending = _make_strategy("New Strat")
    state = BotState(
        pending_strategy=pending,
        pending_strategy_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_confirm(update, context)

    assert state.active_strategy is pending
    assert state.pending_strategy is None
    assert "New Strat" in _sent_text(context)


@pytest.mark.asyncio
async def test_confirm_no_pending_error():
    state = BotState(pending_strategy=None)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_confirm(update, context)

    text = _sent_text(context).lower()
    assert "no hay" in text or "pendiente" in text


@pytest.mark.asyncio
async def test_confirm_expired_clears_pending():
    pending = _make_strategy("Expired Strat")
    state = BotState(
        pending_strategy=pending,
        pending_strategy_expires_at=datetime.now(UTC) - timedelta(minutes=1),  # expired
    )
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_confirm(update, context)

    assert state.pending_strategy is None
    assert state.active_strategy is None  # not activated
    assert "expir" in _sent_text(context).lower()


# ── /cancel ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_clears_pending():
    pending = _make_strategy("To Cancel")
    state = BotState(pending_strategy=pending)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_cancel(update, context)

    assert state.pending_strategy is None
    assert "To Cancel" in _sent_text(context)


@pytest.mark.asyncio
async def test_cancel_no_pending_informs_user():
    state = BotState(pending_strategy=None)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_cancel(update, context)

    # Should not crash, should send a message
    context.bot.send_message.assert_called_once()


# ── /pause and /resume ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_sets_paused_flag():
    state = BotState(bot_paused=False)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_pause(update, context)

    assert state.bot_paused is True
    assert "pausado" in _sent_text(context).lower()


@pytest.mark.asyncio
async def test_pause_already_paused_informs():
    state = BotState(bot_paused=True)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_pause(update, context)

    assert state.bot_paused is True  # unchanged
    context.bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_resume_clears_paused_flag():
    state = BotState(bot_paused=True)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_resume(update, context)

    assert state.bot_paused is False
    assert "reanudado" in _sent_text(context).lower()


@pytest.mark.asyncio
async def test_resume_already_active_informs():
    state = BotState(bot_paused=False)
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_resume(update, context)

    assert state.bot_paused is False
    context.bot.send_message.assert_called_once()


# ── /closeall ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_closeall_no_confirm_asks_confirmation():
    state = BotState(market_state="ACTIVE")
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=[])

    await handle_closeall(update, context)

    assert state.awaiting_closeall_confirm is True
    text = _sent_text(context).lower()
    assert "confirm" in text or "confirmar" in text


@pytest.mark.asyncio
async def test_closeall_confirm_active_market_closes_positions():
    pos = _make_position("AAPL")
    alpaca = _make_alpaca(positions=[pos])
    state = BotState(market_state="ACTIVE")
    deps = _make_deps(alpaca=alpaca)
    update = _make_update()
    context = _make_context(state, deps, args=["confirm"])

    await handle_closeall(update, context)

    alpaca.close_position.assert_called_once_with("AAPL")
    assert "AAPL" in _sent_text(context)


@pytest.mark.asyncio
async def test_closeall_confirm_market_closed_enqueues():
    state = BotState(market_state="IDLE")
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["confirm"])

    await handle_closeall(update, context)

    text = _sent_text(context).lower()
    assert "encolada" in text or "apertura" in text


@pytest.mark.asyncio
async def test_closeall_confirm_no_positions():
    alpaca = _make_alpaca(positions=[])
    state = BotState(market_state="ACTIVE")
    deps = _make_deps(alpaca=alpaca)
    update = _make_update()
    context = _make_context(state, deps, args=["confirm"])

    await handle_closeall(update, context)

    alpaca.close_position.assert_not_called()
    text = _sent_text(context).lower()
    assert "no hay" in text or "sin posicion" in text or "no" in text


# ── /ask ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_no_question_shows_usage():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=[])

    await handle_ask(update, context)

    text = _sent_text(context)
    assert "/ask" in text


@pytest.mark.asyncio
async def test_ask_returns_llm_response():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["¿cuánto", "perdí?"])

    with patch(
        "src.telegram_bot.commands.answer_ask",
        new=AsyncMock(return_value="Perdiste $100."),
    ):
        await handle_ask(update, context)

    assert "Perdiste $100." in _sent_text(context)


# ── /setlimit ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_setlimit_valid_param_saves_to_db(async_session):
    state = BotState()
    deps = _make_deps(session_factory=_make_session_factory(async_session))
    update = _make_update()
    context = _make_context(state, deps, args=["max_position_pct", "5.0"])

    await handle_setlimit(update, context)

    text = _sent_text(context)
    assert "max" in text and "position" in text  # escaped: max\_position\_pct
    assert "5.0" in text


@pytest.mark.asyncio
async def test_setlimit_invalid_param_shows_error():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["invalid_param", "5.0"])

    await handle_setlimit(update, context)

    text = _sent_text(context)
    assert "inválido" in text.lower() or "invalid" in text.lower()


@pytest.mark.asyncio
async def test_setlimit_non_numeric_value_shows_error():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["max_position_pct", "abc"])

    await handle_setlimit(update, context)

    assert "inválido" in _sent_text(context).lower()


@pytest.mark.asyncio
async def test_setlimit_wrong_arg_count_shows_usage():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["max_position_pct"])  # missing value

    await handle_setlimit(update, context)

    assert "/setlimit" in _sent_text(context)


@pytest.mark.asyncio
async def test_setlimit_max_concurrent_positions_out_of_range():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps, args=["max_concurrent_positions", "100"])

    await handle_setlimit(update, context)

    assert "1" in _sent_text(context) and "50" in _sent_text(context)


# ── /help ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_help_lists_commands():
    state = BotState()
    deps = _make_deps()
    update = _make_update()
    context = _make_context(state, deps)

    await handle_help(update, context)

    text = _sent_text(context)
    for cmd in ["/start", "/status", "/positions", "/strategy", "/pause", "/help"]:
        assert cmd in text
