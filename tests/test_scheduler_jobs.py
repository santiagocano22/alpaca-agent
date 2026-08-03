"""Tests for src/scheduler/jobs.py.

All external I/O (Alpaca, Telegram, LLM) is mocked.
APScheduler is not started — job functions are called directly.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.broker.schemas import (
    BarData,
    BarEvent,
    OrderIntent,
    OrderSide,
    OrderStatusResult,
    OrderType,
    PositionSnapshot,
    ValidationResult,
    make_client_order_id,
)
from src.scheduler.jobs import (
    _load_warmup_bars,
    _submit_persisted_order,
    close_job,
    daily_summary_job,
    healthcheck_job,
    holiday_job,
    on_bar_event,
    open_job,
    reconcile_market_state_job,
    reconcile_order_attempts_job,
    warmup_job,
)
from src.storage.models import OrderAttempt, Trade
from src.strategy.engine import StrategyEngine
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
from src.telegram_bot.bot import BotDeps, BotState
from sqlalchemy import select

# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2025, 1, 8, 14, 30, 0, tzinfo=UTC)


def _make_strategy(
    *,
    eod_policy: EodPolicy = EodPolicy.CLOSE_ALL,
    universe: list[str] | None = None,
    timeframe: Timeframe = Timeframe.M15,
) -> Strategy:
    return Strategy(
        name="Test",
        universe=universe or ["AAPL"],
        timeframe=timeframe,
        session=StrategySession.REGULAR,
        horizon=Horizon.INTRADAY,
        eod_policy=eod_policy,
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


def _make_alpaca(
    *,
    positions: list | None = None,
    portfolio_value: float = 10_000.0,
) -> MagicMock:
    m = MagicMock()
    account = MagicMock()
    account.portfolio_value = portfolio_value
    account.buying_power = 5_000.0
    m.get_account = AsyncMock(return_value=account)
    m.get_positions = AsyncMock(return_value=positions or [])
    m.close_position = AsyncMock()
    m.submit_order = AsyncMock()
    m.get_clock = AsyncMock(return_value=MagicMock())
    m.get_bars = AsyncMock(return_value=pd.DataFrame())
    # asset_cache for risk_manager
    asset_cache = MagicMock()
    asset = MagicMock()
    asset.tradeable = True
    asset.status = "active"
    asset_cache.get = AsyncMock(return_value=asset)
    m.asset_cache = asset_cache
    return m


def _make_position(symbol: str = "AAPL", qty: float = 10.0) -> MagicMock:
    pos = MagicMock(spec=PositionSnapshot)
    pos.symbol = symbol
    pos.qty = qty
    pos.side = "long"
    pos.market_value = 1_500.0
    pos.avg_entry_price = 150.0
    pos.unrealized_pl = 50.0
    pos.unrealized_plpc = 0.03
    pos.current_price = 155.0
    return pos


def _make_deps(*, alpaca=None, bars_cache: dict | None = None) -> BotDeps:
    @asynccontextmanager
    async def _factory():
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(first=MagicMock(return_value=None)))))
        yield mock_session

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Daily summary text")

    return BotDeps(
        alpaca=alpaca or _make_alpaca(),
        session_factory=_factory,
        llm_client=llm,
        authorized_chat_id=123456,
        bars_cache=bars_cache if bars_cache is not None else {},
    )


async def _no_notify(_: str) -> None:
    """Null notify function for tests that don't need to assert on messages."""
    pass


def _bar_event(
    symbol: str = "AAPL",
    close: float = 140.0,
    timestamp: datetime | None = None,
) -> BarEvent:
    ts = timestamp or _NOW
    return BarEvent(
        bar=BarData(
            symbol=symbol,
            timestamp=ts,
            open=close - 1,
            high=close + 1,
            low=close - 2,
            close=close,
            volume=10_000.0,
        ),
        received_at=ts,
    )


# ── warmup_job ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warmup_filters_extended_hours_for_regular_strategy():
    alpaca = _make_alpaca()
    alpaca.get_bars = AsyncMock(
        return_value=[
            BarData(
                symbol="AAPL",
                timestamp=datetime(2025, 1, 8, 14, 0, tzinfo=UTC),  # 09:00 ET
                open=100,
                high=101,
                low=99,
                close=100,
                volume=100,
            ),
            BarData(
                symbol="AAPL",
                timestamp=datetime(2025, 1, 8, 14, 30, tzinfo=UTC),  # 09:30 ET
                open=101,
                high=102,
                low=100,
                close=101,
                volume=100,
            ),
        ]
    )

    bars = await _load_warmup_bars(
        alpaca,
        "AAPL",
        "15Min",
        2,
        StrategySession.REGULAR,
    )

    assert len(bars) == 1
    assert bars.index[0] == datetime(2025, 1, 8, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_warmup_sets_state_to_warmup():
    state = BotState(market_state="IDLE")
    deps = _make_deps()
    messages = []

    async def notify(text):
        messages.append(text)

    await warmup_job(state, deps, notify, warmup_minutes=5)

    assert state.market_state == "WARMUP"


@pytest.mark.asyncio
async def test_warmup_sends_notification_with_strategy_name():
    state = BotState(market_state="IDLE", active_strategy=_make_strategy())
    deps = _make_deps()
    messages = []

    async def notify(text):
        messages.append(text)

    await warmup_job(state, deps, notify, warmup_minutes=5)

    assert len(messages) >= 1
    assert "Test" in messages[0] or "test" in messages[0].lower()


@pytest.mark.asyncio
async def test_warmup_calls_get_bars_for_universe():
    strategy = _make_strategy(universe=["AAPL", "QQQ"])
    alpaca = _make_alpaca()
    state = BotState(market_state="IDLE", active_strategy=strategy)
    deps = _make_deps(alpaca=alpaca)

    await warmup_job(state, deps, _no_notify, warmup_minutes=5)

    assert alpaca.get_bars.call_count == 2  # one per symbol


@pytest.mark.asyncio
async def test_warmup_skips_if_not_idle():
    state = BotState(market_state="ACTIVE")
    deps = _make_deps()
    messages = []

    async def notify(text):
        messages.append(text)

    await warmup_job(state, deps, notify, warmup_minutes=5)

    assert state.market_state == "ACTIVE"  # unchanged
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_warmup_no_strategy_skips_bar_fetch():
    alpaca = _make_alpaca()
    state = BotState(market_state="IDLE", active_strategy=None)
    deps = _make_deps(alpaca=alpaca)

    await warmup_job(state, deps, _no_notify, warmup_minutes=5)

    alpaca.get_bars.assert_not_called()


# ── open_job ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_job_sets_state_to_active():
    state = BotState(market_state="WARMUP")
    messages = []

    async def notify(text):
        messages.append(text)

    await open_job(state, notify)

    assert state.market_state == "ACTIVE"
    assert "🟢" in messages[0] or "abierto" in messages[0].lower()


# ── close_job ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_job_sets_state_to_idle():
    state = BotState(market_state="ACTIVE", active_strategy=_make_strategy(eod_policy=EodPolicy.HOLD))
    deps = _make_deps()
    messages = []

    async def notify(text):
        messages.append(text)

    await close_job(state, deps, notify)

    assert state.market_state == "IDLE"
    assert "🔴" in messages[0] or "cerrado" in messages[0].lower()


@pytest.mark.asyncio
async def test_close_job_eod_close_all_closes_positions():
    pos = _make_position("AAPL")
    alpaca = _make_alpaca(positions=[pos])
    state = BotState(market_state="ACTIVE", active_strategy=_make_strategy(eod_policy=EodPolicy.CLOSE_ALL))
    deps = _make_deps(alpaca=alpaca)

    await close_job(state, deps, _no_notify)

    alpaca.close_position.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_close_job_eod_hold_does_not_close():
    pos = _make_position("AAPL")
    alpaca = _make_alpaca(positions=[pos])
    state = BotState(
        market_state="ACTIVE",
        active_strategy=_make_strategy(eod_policy=EodPolicy.HOLD),
    )
    deps = _make_deps(alpaca=alpaca)

    await close_job(state, deps, _no_notify)

    alpaca.close_position.assert_not_called()


# ── holiday_job ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_holiday_sends_notification():
    messages = []

    async def notify(text):
        messages.append(text)

    await holiday_job(notify, trade_date=date(2025, 1, 1), reason="Año Nuevo")

    assert len(messages) == 1
    assert "2025-01-01" in messages[0]
    assert "Año Nuevo" in messages[0]


# ── daily_summary_job ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_summary_sends_message():
    state = BotState(daily_realized_pnl=100.0, daily_trade_count=5, daily_win_count=3)
    deps = _make_deps()
    messages = []

    async def notify(text):
        messages.append(text)

    await daily_summary_job(state, deps, notify)

    assert len(messages) == 1
    assert messages[0]  # non-empty


@pytest.mark.asyncio
async def test_daily_summary_resets_daily_tallies():
    state = BotState(daily_realized_pnl=500.0, daily_trade_count=10, daily_win_count=7)
    deps = _make_deps()

    await daily_summary_job(state, deps, _no_notify)

    assert state.daily_realized_pnl == 0.0
    assert state.daily_trade_count == 0
    assert state.daily_win_count == 0


# ── healthcheck_job ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healthcheck_reachable_no_alert():
    alpaca = _make_alpaca()
    deps = _make_deps(alpaca=alpaca)
    messages = []

    async def notify(text):
        messages.append(text)

    await healthcheck_job(deps, notify)

    assert len(messages) == 0  # no alert when Alpaca is reachable


@pytest.mark.asyncio
async def test_healthcheck_timeout_sends_alert():
    alpaca = _make_alpaca()
    alpaca.get_clock = AsyncMock(side_effect=asyncio.TimeoutError())
    deps = _make_deps(alpaca=alpaca)
    messages = []

    async def notify(text):
        messages.append(text)

    await healthcheck_job(deps, notify)

    assert len(messages) == 1
    assert "timeout" in messages[0].lower() or "ALERTA" in messages[0]


@pytest.mark.asyncio
async def test_healthcheck_error_sends_alert():
    alpaca = _make_alpaca()
    alpaca.get_clock = AsyncMock(side_effect=RuntimeError("connection refused"))
    deps = _make_deps(alpaca=alpaca)
    messages = []

    async def notify(text):
        messages.append(text)

    await healthcheck_job(deps, notify)

    assert len(messages) == 1


# ── on_bar_event ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_bar_event_market_not_active_skips():
    state = BotState(market_state="IDLE", active_strategy=_make_strategy())
    deps = _make_deps()
    engine = StrategyEngine(_make_strategy())
    messages = []

    await on_bar_event(
        bar_event=_bar_event(),
        state=state,
        deps=deps,
        notify=messages.append,
        engine=engine,
    )

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_on_bar_event_paused_skips():
    state = BotState(market_state="ACTIVE", bot_paused=True, active_strategy=_make_strategy())
    deps = _make_deps()
    engine = StrategyEngine(_make_strategy())
    messages = []

    await on_bar_event(
        bar_event=_bar_event(),
        state=state,
        deps=deps,
        notify=messages.append,
        engine=engine,
    )

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_on_bar_event_no_strategy_skips():
    state = BotState(market_state="ACTIVE", bot_paused=False, active_strategy=None)
    deps = _make_deps()
    engine = StrategyEngine(_make_strategy())  # not used
    messages = []

    await on_bar_event(
        bar_event=_bar_event(),
        state=state,
        deps=deps,
        notify=messages.append,
        engine=engine,
    )

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_on_bar_event_entry_fires_when_rsi_low(mocker):
    """When RSI < 30 condition is met (40 strictly decreasing bars → RSI ≈ 0), entry fires."""
    strategy = _make_strategy(timeframe=Timeframe.M1)
    engine = StrategyEngine(strategy)

    # Build a bars_cache with sufficient bars for RSI to be below 30
    # Use strictly decreasing prices so RSI → 0
    n = engine.required_lookback_bars() + 5
    timestamps = pd.date_range("2025-01-08 09:30", periods=n, freq="1min", tz="UTC")
    prices = [100.0 - i * 0.5 for i in range(n)]
    bars_df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [10_000.0] * n,
        },
        index=timestamps,
    )
    bars_cache = {"AAPL": bars_df}

    alpaca = _make_alpaca(positions=[])  # no existing positions
    alpaca.submit_order = AsyncMock(
        return_value=MagicMock(alpaca_order_id="order-1", status="submitted")
    )
    mocker.patch(
        "src.scheduler.jobs.validate_order",
        new=AsyncMock(return_value=ValidationResult.approve("AAPL", 1000.0)),
    )
    state = BotState(market_state="ACTIVE", bot_paused=False, active_strategy=strategy)
    deps = _make_deps(alpaca=alpaca, bars_cache=bars_cache)
    messages = []

    async def notify(text):
        messages.append(text)

    # The bar event adds a new (still decreasing) bar
    await on_bar_event(
        bar_event=_bar_event("AAPL", close=prices[-1] - 0.5),
        state=state,
        deps=deps,
        notify=notify,
        engine=engine,
    )

    alpaca.submit_order.assert_awaited_once()
    assert any("BUY AAPL" in message for message in messages)


@pytest.mark.asyncio
async def test_on_bar_event_no_signal_no_order():
    """When no entry or exit signal, submit_order is never called."""
    strategy = _make_strategy()
    engine = StrategyEngine(strategy)

    # Use flat prices — RSI is NaN/100, not < 30 → no entry
    n = engine.required_lookback_bars() + 5
    timestamps = pd.date_range("2025-01-08 09:30", periods=n, freq="1min", tz="UTC")
    prices = [150.0] * n
    bars_df = pd.DataFrame(
        {
            "open": prices, "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices], "close": prices,
            "volume": [1000.0] * n,
        },
        index=timestamps,
    )
    bars_cache = {"AAPL": bars_df}

    alpaca = _make_alpaca(positions=[])
    state = BotState(market_state="ACTIVE", bot_paused=False, active_strategy=strategy)
    deps = _make_deps(alpaca=alpaca, bars_cache=bars_cache)

    await on_bar_event(
        bar_event=_bar_event("AAPL", close=150.0),
        state=state,
        deps=deps,
        notify=_no_notify,
        engine=engine,
    )

    alpaca.submit_order.assert_not_called()


# ── Recurring market-state reconciliation ────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_mid_session_recovers_to_active(sample_calendar):
    alpaca = _make_alpaca()
    alpaca.get_bars = AsyncMock(return_value=[])
    state = BotState(
        market_state="IDLE",
        active_strategy=_make_strategy(eod_policy=EodPolicy.HOLD),
    )
    deps = _make_deps(alpaca=alpaca)
    deps.calendar = sample_calendar
    messages = []

    async def notify(message):
        messages.append(message)

    await reconcile_market_state_job(
        state,
        deps,
        notify,
        now=datetime(2025, 1, 8, 15, 0, tzinfo=UTC),  # 10:00 ET
    )

    assert state.market_state == "ACTIVE"
    assert any("Mercado abierto" in message for message in messages)
    assert not any("abre en" in message for message in messages)


@pytest.mark.asyncio
async def test_reconcile_operates_across_multiple_sessions(sample_calendar):
    alpaca = _make_alpaca()
    alpaca.get_bars = AsyncMock(return_value=[])
    state = BotState(
        market_state="IDLE",
        active_strategy=_make_strategy(eod_policy=EodPolicy.HOLD),
    )
    deps = _make_deps(alpaca=alpaca)
    deps.calendar = sample_calendar
    messages = []

    async def notify(message):
        messages.append(message)

    moments = [
        datetime(2025, 1, 8, 14, 25, tzinfo=UTC),  # warmup
        datetime(2025, 1, 8, 14, 30, tzinfo=UTC),  # day 1 open
        datetime(2025, 1, 8, 21, 0, tzinfo=UTC),   # day 1 close
        datetime(2025, 1, 9, 14, 30, tzinfo=UTC),  # day 2 open
    ]
    for moment in moments:
        await reconcile_market_state_job(state, deps, notify, now=moment)

    assert state.market_state == "ACTIVE"
    assert sum("Mercado abierto" in message for message in messages) == 2
    assert sum("Mercado cerrado" in message for message in messages) == 1


@pytest.mark.asyncio
async def test_reconcile_eod_liquidation_is_idempotent(sample_calendar):
    position = _make_position("AAPL")
    alpaca = _make_alpaca(positions=[position])
    alpaca.get_bars = AsyncMock(return_value=[])
    state = BotState(
        market_state="ACTIVE",
        active_strategy=_make_strategy(eod_policy=EodPolicy.CLOSE_ALL),
    )
    deps = _make_deps(alpaca=alpaca)
    deps.calendar = sample_calendar
    now = datetime(2025, 1, 8, 20, 56, tzinfo=UTC)  # 15:56 ET

    await reconcile_market_state_job(state, deps, _no_notify, now=now)
    await reconcile_market_state_job(state, deps, _no_notify, now=now)

    alpaca.close_position.assert_called_once_with("AAPL")
    assert state.last_eod_liquidation_date == date(2025, 1, 8)


@pytest.mark.asyncio
async def test_on_bar_event_waits_for_configured_timeframe(mocker):
    strategy = _make_strategy(timeframe=Timeframe.M5)
    state = BotState(market_state="ACTIVE", active_strategy=strategy)
    deps = _make_deps()
    engine = StrategyEngine(strategy)
    evaluate = mocker.patch("src.scheduler.jobs._evaluate_symbol", new_callable=AsyncMock)

    start = datetime(2025, 1, 8, 14, 30, tzinfo=UTC)
    for minute in range(4):
        await on_bar_event(
            _bar_event(timestamp=start + timedelta(minutes=minute)),
            state,
            deps,
            _no_notify,
            engine,
        )
    evaluate.assert_not_awaited()

    await on_bar_event(
        _bar_event(timestamp=start + timedelta(minutes=4)),
        state,
        deps,
        _no_notify,
        engine,
    )
    evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_order_attempt_is_committed_with_submission_result(async_session):
    alpaca = _make_alpaca()
    alpaca.submit_order = AsyncMock(
        return_value=MagicMock(alpaca_order_id="alpaca-123", status="accepted")
    )
    deps = _make_deps(alpaca=alpaca)
    state = BotState()
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.MARKET,
        client_order_id=make_client_order_id(1, "AAPL"),
    )

    await _submit_persisted_order(
        intent=intent,
        db_session=async_session,
        deps=deps,
        state=state,
        notify=_no_notify,
    )

    attempt = (
        await async_session.execute(
            select(OrderAttempt).where(OrderAttempt.client_order_id == intent.client_order_id)
        )
    ).scalar_one()
    assert attempt.alpaca_order_id == "alpaca-123"
    assert attempt.status == "accepted"


@pytest.mark.asyncio
async def test_order_reconciler_recovers_attempt_by_client_id(async_session):
    attempt = OrderAttempt(
        client_order_id="client-recover-1",
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        status="pending",
    )
    async_session.add(attempt)
    await async_session.commit()

    alpaca = _make_alpaca()
    alpaca.get_order_by_client_id = AsyncMock(
        return_value=OrderStatusResult(
            alpaca_order_id="alpaca-recovered",
            client_order_id="client-recover-1",
            status="filled",
            submitted_at=datetime.now(UTC),
            filled_at=datetime.now(UTC),
            filled_qty=1,
            filled_avg_price=150,
            side=OrderSide.BUY,
            symbol="AAPL",
            qty=1,
        )
    )
    deps = _make_deps(
        alpaca=alpaca,
    )

    @asynccontextmanager
    async def session_factory():
        yield async_session

    deps.session_factory = session_factory

    await reconcile_order_attempts_job(BotState(), deps, _no_notify)

    await async_session.refresh(attempt)
    assert attempt.alpaca_order_id == "alpaca-recovered"
    assert attempt.status == "filled"
    trade = (
        await async_session.execute(
            select(Trade).where(Trade.client_order_id == attempt.client_order_id)
        )
    ).scalar_one()
    assert trade.filled_price == 150
    assert trade.qty == 1
