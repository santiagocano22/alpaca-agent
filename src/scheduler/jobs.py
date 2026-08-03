"""Scheduler jobs for market state transitions and periodic tasks.

Every job is a plain async function taking explicit arguments, so they can
be unit-tested without running APScheduler or requiring a live connection.

Market state machine:
  IDLE → WARMUP   (warmup_minutes_before_open before session open)
  WARMUP → ACTIVE (at session open)
  ACTIVE → IDLE   (at session close)

Telegram notifications (text only, no LLM):
  WARMUP : "⏰ Mercado abre en N min. Preparando estrategia: <name>."
  ACTIVE : "🟢 Mercado abierto. Bot operativo."
  IDLE   : "🔴 Mercado cerrado. Generando resumen del día…"
  HOLIDAY: "📅 Hoy <date> es día no hábil. Bot en standby."

Bar evaluation callback:
  on_bar_event() is called by the StreamManager for each completed OHLCV bar.
  It evaluates entry / exit rules via the StrategyEngine, validates via
  risk_manager, submits to Alpaca, and persists the result to DB.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.broker.risk_manager import validate_order
from src.broker.schemas import (
    BarEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    RiskOverrides,
    make_client_order_id,
)
from src.storage.models import AlertLog, OrderAttempt, PendingAction, RiskOverride, Trade
from src.strategy.engine import StrategyEngine
from src.strategy.schema import EodPolicy, Session as StrategySession
from src.telegram_bot.bot import BotDeps, BotState
from src.utils.market_hours import (
    ET,
    CalendarCache,
    MarketDay,
    MarketState,
    get_market_state,
    is_extended_hours_open,
    minutes_to_close,
    today_is_holiday,
)

# ── Type aliases ──────────────────────────────────────────────────────────────

NotifyFn = Callable[[str], Awaitable[None]]
"""Async callable that sends a Telegram message to the authorized chat."""

# ── Warmup helpers ────────────────────────────────────────────────────────────

_WARMUP_BARS_SAFETY_MULTIPLIER = 1.5
"""Fetch 50% more bars than required to account for weekends/holidays in lookback."""

_NON_TERMINAL_ORDER_STATUSES = {
    "pending",
    "submitted",
    "accepted",
    "new",
    "pending_new",
    "partially_filled",
}


async def _load_warmup_bars(
    alpaca,
    symbol: str,
    timeframe_str: str,
    required_bars: int,
    session: StrategySession = StrategySession.REGULAR,
) -> pd.DataFrame:
    """Fetch historical bars for indicator warmup.

    Returns a DataFrame (may be empty on error; caller handles InsufficientHistoryError).
    """
    from datetime import UTC, datetime, timedelta

    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    fetch_count = int(required_bars * _WARMUP_BARS_SAFETY_MULTIPLIER) + 1  # noqa: F841

    # Map strategy timeframe string → calendar days to look back
    _DAYS_BACK = {"1D": 600, "1H": 30, "15Min": 10, "5Min": 4, "1Min": 2}
    days_back = _DAYS_BACK.get(timeframe_str, 600)

    # Map strategy timeframe string → AlpacaTimeFrame
    _TF_MAP = {
        "1D": TimeFrame.Day,
        "1H": TimeFrame.Hour,
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Min": TimeFrame.Minute,
    }
    tf = _TF_MAP.get(timeframe_str, TimeFrame.Day)

    end = datetime.now(UTC)
    start = end - timedelta(days=days_back)

    try:
        bar_list = await alpaca.get_bars(symbol, start=start, end=end, timeframe=tf)
        if not bar_list:
            logger.warning("Warmup: no bars returned for {}", symbol)
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bar_list
            ],
            index=pd.DatetimeIndex([b.timestamp for b in bar_list]),
        )
        if timeframe_str != "1D" and session != StrategySession.CRYPTO_24_7:
            index_et = df.index.tz_convert(ET)
            minute_of_day = index_et.hour * 60 + index_et.minute
            if session == StrategySession.EXTENDED:
                session_start, session_end = 4 * 60, 20 * 60
            else:
                session_start, session_end = 9 * 60 + 30, 16 * 60
            df = df[(minute_of_day >= session_start) & (minute_of_day < session_end)]
        logger.info("Warmup: fetched {} bars for {} (needed {})", len(df), symbol, required_bars)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Warmup: failed to fetch bars for {}: {}", symbol, exc)
        return pd.DataFrame()


# ── Market state job functions ─────────────────────────────────────────────────


async def preload_strategy_bars(state: BotState, deps: BotDeps) -> tuple[int, int]:
    """Load timeframe-consistent history for the currently active strategy."""
    if state.active_strategy is None:
        return 0, 0
    engine = StrategyEngine(state.active_strategy)
    needed = engine.required_lookback_bars()
    timeframe = state.active_strategy.timeframe.value
    symbols = state.active_strategy.universe
    deps.bar_aggregator.reset(set(symbols))
    loaded = 0
    for symbol in symbols:
        frame = await _load_warmup_bars(
            deps.alpaca,
            symbol,
            timeframe,
            needed,
            state.active_strategy.session,
        )
        if not frame.empty:
            deps.bars_cache[symbol] = frame
            loaded += 1
    return loaded, len(symbols)


async def warmup_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    warmup_minutes: int = 5,
    notify_user: bool = True,
) -> None:
    """Transition IDLE → WARMUP and preload indicator bars.

    Fetches historical OHLCV bars for every symbol in the active strategy's
    universe so that indicators are ready at the first live bar.  Logs a
    warning (does not abort) if there is insufficient history.

    Args:
        state:           Shared BotState (mutated: market_state → WARMUP).
        deps:            Shared BotDeps.
        notify:          Async callable to send a Telegram message.
        warmup_minutes:  Minutes until market open (for the notification text).
    """
    if state.market_state != "IDLE":
        logger.debug("warmup_job: market_state={}, skipping", state.market_state)
        return

    state.market_state = "WARMUP"
    strategy_name = state.active_strategy.name if state.active_strategy else "ninguna"
    if notify_user:
        msg = f"⏰ Mercado abre en {warmup_minutes} min. Preparando estrategia: {strategy_name}."
        await notify(msg)

    if state.active_strategy is None:
        logger.info("Warmup: no active strategy, skipping bar preload")
        return

    loaded, total = await preload_strategy_bars(state, deps)
    if loaded < total:
        state.last_error = f"Warmup incompleto: {loaded}/{total} símbolos con histórico"


async def refresh_calendar_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    now: datetime | None = None,
    lookahead_days: int = 45,
) -> bool:
    """Refresh the shared Alpaca calendar in place, with retryable failure state."""
    reference = now or datetime.now(UTC)
    try:
        raw_days = await deps.alpaca.get_market_calendar(
            start=reference.date(),
            end=(reference + timedelta(days=lookahead_days)).date(),
        )
        refreshed = {
            day.date: MarketDay(date=day.date, open=day.open_et, close=day.close_et)
            for day in raw_days
        }
        deps.calendar.clear()
        deps.calendar.update(refreshed)
        state.last_calendar_refresh = reference
        logger.info("Calendar refreshed: {} trading days", len(refreshed))
        return True
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Calendar refresh failed: {exc}"
        logger.error("Calendar refresh failed: {}", exc)
        await notify(f"⚠️ ALERTA: no se pudo actualizar el calendario de mercado: {exc}")
        return False


async def open_job(
    state: BotState,
    notify: NotifyFn,
    deps: BotDeps | None = None,
) -> None:
    """Transition WARMUP → ACTIVE at session open."""
    state.market_state = "ACTIVE"
    logger.info("Market opened — transitioning to ACTIVE")
    await notify("🟢 Mercado abierto. Bot operativo.")
    if deps is None:
        return

    async with deps.session_factory() as db:
        pending = (
            await db.execute(
                select(PendingAction).where(
                    PendingAction.status == "pending",
                    PendingAction.type == "close_all",
                )
            )
        ).scalars().all()
        if not pending:
            return
        try:
            positions = await deps.alpaca.get_positions()
            for position in positions:
                await deps.alpaca.close_position(position.symbol)
            for action in pending:
                action.status = "executed"
                action.executed_at = datetime.now(UTC)
            await db.commit()
            await notify(
                f"✅ Acción pendiente close_all ejecutada para {len(positions)} posición(es)."
            )
        except Exception as exc:  # noqa: BLE001
            await db.rollback()
            state.last_error = f"Pending close_all failed: {exc}"
            await notify(f"⚠️ Falló la acción pendiente close_all: {exc}")


async def close_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    now: datetime | None = None,
) -> None:
    """Transition ACTIVE → IDLE at session close.

    If the active strategy has ``eod_policy=close_all``, all open positions
    are closed with market orders before transitioning.
    """
    reference = now or datetime.now(UTC)
    current_trade_date = reference.astimezone(ET).date()
    if (
        state.active_strategy
        and state.active_strategy.eod_policy == EodPolicy.CLOSE_ALL
        and state.last_eod_liquidation_date != current_trade_date
    ):
        logger.info("EOD policy=close_all: closing all positions")
        try:
            positions = await deps.alpaca.get_positions()
            for pos in positions:
                await deps.alpaca.close_position(pos.symbol)
                logger.info("EOD: closed position {}", pos.symbol)
            state.last_eod_liquidation_date = current_trade_date
        except Exception as exc:  # noqa: BLE001
            logger.error("EOD close_all error: {}", exc)

    # For daily strategies: fetch today's completed bar and evaluate entry/exit signals
    if (
        state.active_strategy is not None
        and state.active_strategy.timeframe.value == "1D"
        and not state.bot_paused
    ):
        from alpaca.data.timeframe import TimeFrame

        engine = StrategyEngine(state.active_strategy)
        today_end = reference
        today_start = today_end - timedelta(hours=24)

        for symbol in state.active_strategy.universe:
            try:
                bar_list = await deps.alpaca.get_bars(
                    symbol, start=today_start, end=today_end, timeframe=TimeFrame.Day
                )
                if not bar_list:
                    continue
                last_bar = bar_list[-1]
                new_row = pd.DataFrame(
                    [
                        {
                            "open": last_bar.open,
                            "high": last_bar.high,
                            "low": last_bar.low,
                            "close": last_bar.close,
                            "volume": last_bar.volume,
                        }
                    ],
                    index=pd.DatetimeIndex([last_bar.timestamp]),
                )
                existing = deps.bars_cache.get(symbol, pd.DataFrame())
                deps.bars_cache[symbol] = pd.concat([existing, new_row]).tail(
                    engine.required_lookback_bars() * 2
                )
                bars = deps.bars_cache[symbol]
                await _evaluate_symbol(
                    symbol=symbol,
                    bars=bars,
                    state=state,
                    deps=deps,
                    notify=notify,
                    engine=engine,
                    overrides=None,
                    calendar=None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("close_job: error evaluating {} signals: {}", symbol, exc)

    state.market_state = "IDLE"
    logger.info("Market closed — transitioning to IDLE")
    await notify("🔴 Mercado cerrado. Generando resumen del día…")


async def eod_liquidation_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    trade_date: date,
) -> None:
    """Apply close-all policy once, before the official market close."""
    if state.last_eod_liquidation_date == trade_date:
        return
    if state.active_strategy is None or state.active_strategy.eod_policy != EodPolicy.CLOSE_ALL:
        state.last_eod_liquidation_date = trade_date
        return
    try:
        positions = await deps.alpaca.get_positions()
        for position in positions:
            await deps.alpaca.close_position(position.symbol)
        state.last_eod_liquidation_date = trade_date
        if positions:
            await notify(
                f"🧹 Cierre EOD enviado para {len(positions)} posición(es) antes del cierre."
            )
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"EOD liquidation failed: {exc}"
        logger.error("EOD liquidation failed: {}", exc)
        await notify(f"⚠️ ALERTA: falló el cierre EOD de posiciones: {exc}")


async def reconcile_market_state_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    now: datetime | None = None,
    warmup_minutes: int = 5,
    eod_close_minutes_before: int = 5,
    summary_delay_minutes: int = 5,
) -> None:
    """Idempotently reconcile runtime state with the current Alpaca session.

    Unlike one-shot date jobs, this runs throughout the process lifetime and
    recovers correctly after restarts, missed jobs, calendar refreshes, and
    multi-day operation.
    """
    reference = now or datetime.now(UTC)
    if not deps.calendar:
        await refresh_calendar_job(state, deps, notify, now=reference)
        if not deps.calendar:
            return

    reference_et = reference.astimezone(ET)
    trade_date = reference_et.date()
    strategy_session = (
        state.active_strategy.session
        if state.active_strategy is not None
        else StrategySession.REGULAR
    )
    if strategy_session == StrategySession.EXTENDED:
        if is_extended_hours_open(reference, deps.calendar):
            desired = MarketState.ACTIVE
        else:
            desired = MarketState.IDLE
            for offset in range(45):
                candidate_date = trade_date + timedelta(days=offset)
                candidate = deps.calendar.get(candidate_date)
                if candidate is None:
                    continue
                session_open = datetime.combine(
                    candidate_date, candidate.session_open, tzinfo=ET
                )
                delta = (session_open - reference_et).total_seconds() / 60
                if 0 < delta <= warmup_minutes:
                    desired = MarketState.WARMUP
                if delta > 0:
                    break
    else:
        desired = get_market_state(reference, deps.calendar, warmup_minutes)

    if (
        desired == MarketState.IDLE
        and today_is_holiday(reference, deps.calendar)
        and state.last_holiday_notice_date != trade_date
        and state.last_calendar_refresh is not None
    ):
        await holiday_job(notify, trade_date=trade_date)
        state.last_holiday_notice_date = trade_date

    if desired == MarketState.WARMUP:
        if state.market_state == "IDLE":
            await warmup_job(
                state,
                deps,
                notify,
                warmup_minutes=warmup_minutes,
            )
        return

    if desired == MarketState.ACTIVE:
        if state.market_state == "IDLE":
            # Mid-session restart: preload history without claiming the market
            # is about to open, then transition immediately to ACTIVE.
            await warmup_job(
                state,
                deps,
                notify,
                warmup_minutes=warmup_minutes,
                notify_user=False,
            )
        if state.market_state != "ACTIVE":
            await open_job(state, notify, deps)

        if strategy_session == StrategySession.EXTENDED:
            market_day = deps.calendar.get(trade_date)
            session_close = (
                datetime.combine(trade_date, market_day.session_close, tzinfo=ET)
                if market_day is not None
                else None
            )
            remaining = (
                (session_close - reference_et).total_seconds() / 60
                if session_close is not None
                else None
            )
        else:
            remaining = minutes_to_close(reference, deps.calendar)
        if (
            remaining is not None
            and remaining <= eod_close_minutes_before
            and state.last_eod_liquidation_date != trade_date
        ):
            await eod_liquidation_job(
                state,
                deps,
                notify,
                trade_date=trade_date,
            )
        return

    # Desired IDLE. Only emit a close transition if this process observed an
    # active/warmup state; a nighttime restart must not invent a close event.
    if state.market_state in {"ACTIVE", "WARMUP"}:
        await close_job(state, deps, notify, now=reference)

    market_day = deps.calendar.get(trade_date)
    if market_day is not None and state.last_summary_date != trade_date:
        close_time = (
            market_day.session_close
            if strategy_session == StrategySession.EXTENDED
            else market_day.close
        )
        close_at = datetime.combine(trade_date, close_time, tzinfo=ET)
        if reference_et >= close_at + timedelta(minutes=summary_delay_minutes):
            await daily_summary_job(state, deps, notify, now=reference)
            state.last_summary_date = trade_date


async def holiday_job(
    notify: NotifyFn,
    *,
    trade_date: date,
    reason: str = "mercado cerrado",
) -> None:
    """Send holiday notification (no state transition — bot stays IDLE)."""
    msg = f"📅 Hoy {trade_date} es día no hábil ({reason}). Bot en standby."
    await notify(msg)
    logger.info("Holiday notification sent for date={}", trade_date)


async def daily_summary_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    now: datetime | None = None,
) -> None:
    """Generate and send the daily P&L summary via Haiku (runs 5 min after close).

    Uses aggregated data from BotState (updated by trade callbacks during the day)
    rather than re-querying the full trade log, to stay within the Haiku token budget.
    """
    from src.llm.summarizer import generate_daily_summary

    try:
        account = await deps.alpaca.get_account()
        positions = await deps.alpaca.get_positions()
        equity = account.portfolio_value
        pos_dicts = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
            }
            for p in positions
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("daily_summary: failed to fetch account/positions: {}", exc)
        equity = 0.0
        pos_dicts = []

    today = (now or datetime.now(UTC)).astimezone(ET).date()
    day_start = datetime.combine(today, datetime.min.time(), tzinfo=ET).astimezone(UTC)
    day_end = day_start + timedelta(days=1)
    try:
        async with deps.session_factory() as db:
            trade_rows = (
                await db.execute(
                    select(Trade)
                    .where(Trade.filled_at >= day_start, Trade.filled_at < day_end)
                    .order_by(Trade.filled_at)
                )
            ).scalars().all()
        trades_today = [
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "qty": trade.qty,
                "filled_price": trade.filled_price,
                "pnl": trade.pnl,
            }
            for trade in trade_rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("daily_summary: failed to load fills: {}", exc)
        trades_today = []

    summary = await generate_daily_summary(
        trades_today=trades_today,
        open_positions=pos_dicts,
        realized_pnl=state.daily_realized_pnl,
        unrealized_pnl=sum(p.get("unrealized_pl", 0) for p in pos_dicts),
        equity=equity,
        trade_date=today,
        client=deps.llm_client,
    )
    await notify(summary)

    # Reset daily tallies for next trading day.
    state.daily_realized_pnl = 0.0
    state.daily_trade_count = 0
    state.daily_win_count = 0


async def healthcheck_job(
    deps: BotDeps,
    notify: NotifyFn,
    state: BotState | None = None,
) -> None:
    """Ping Alpaca (get_clock) and notify if unreachable.

    This job is non-fatal: Alpaca connectivity issues are alarming but the bot
    should keep running and retrying.
    """
    try:
        await asyncio.wait_for(deps.alpaca.get_clock(), timeout=5.0)
        logger.debug("Healthcheck: Alpaca reachable")
    except asyncio.TimeoutError:
        logger.error("Healthcheck: Alpaca clock timeout")
        await notify("⚠️ ALERTA: Alpaca no respondió al healthcheck (timeout 5s).")
        if state is not None:
            state.last_error = "Alpaca healthcheck: timeout 5s"
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Healthcheck: Alpaca error: {}", exc)
        await notify(f"⚠️ ALERTA: Error de conexión con Alpaca: {exc}")
        if state is not None:
            state.last_error = f"Alpaca healthcheck: {exc}"
        return

    if state is None or deps.stream_manager is None:
        return
    stream = deps.stream_manager
    unhealthy: list[str] = []
    if not stream.is_data_stream_running:
        unhealthy.append("stream de datos detenido")
    if not stream.is_trading_stream_running:
        unhealthy.append("stream de órdenes detenido")
    if state.market_state == "ACTIVE" and state.subscribed_symbols:
        now = datetime.now(UTC)
        stale = [
            symbol
            for symbol in state.subscribed_symbols
            if symbol not in state.last_bar_at
            or (now - state.last_bar_at[symbol]).total_seconds() > 300
        ]
        if stale:
            unhealthy.append(f"sin barras recientes: {', '.join(sorted(stale))}")
    if unhealthy:
        message = "; ".join(unhealthy)
        state.last_error = message
        await notify(f"⚠️ ALERTA operativa: {message}")


async def reconcile_order_attempts_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
) -> None:
    """Recover order state after crashes between DB commit and API response."""
    try:
        async with deps.session_factory() as db:
            attempts = (
                await db.execute(
                    select(OrderAttempt).where(
                        OrderAttempt.status.in_(_NON_TERMINAL_ORDER_STATUSES)
                    )
                )
            ).scalars().all()
            for attempt in attempts:
                if attempt.alpaca_order_id:
                    order = await deps.alpaca.get_order(attempt.alpaca_order_id)
                else:
                    order = await deps.alpaca.get_order_by_client_id(
                        attempt.client_order_id
                    )
                if order is None:
                    continue
                attempt.alpaca_order_id = order.alpaca_order_id
                attempt.status = order.status
                attempt.updated_at = datetime.now(UTC)
                if order.status == "filled":
                    existing_trade = (
                        await db.execute(
                            select(Trade).where(
                                Trade.client_order_id == attempt.client_order_id
                            )
                        )
                    ).scalars().first()
                    if existing_trade is None:
                        db.add(
                            Trade(
                                client_order_id=attempt.client_order_id,
                                order_attempt_id=attempt.id,
                                symbol=order.symbol,
                                side=order.side.value,
                                qty=order.filled_qty or order.qty,
                                filled_price=order.filled_avg_price,
                                order_type=attempt.order_type,
                                status="filled",
                                rule_trigger=attempt.rule_trigger,
                                strategy_version_id=attempt.strategy_version_id,
                                filled_at=order.filled_at or datetime.now(UTC),
                            )
                        )
                        state.daily_trade_count += 1
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        message = f"Order reconciliation failed: {exc}"
        logger.error(message)
        if state.last_error != message:
            await notify(f"⚠️ {message}")
        state.last_error = message


# ── Bar event evaluation ───────────────────────────────────────────────────────


async def _load_active_risk_overrides(
    db_session: AsyncSession,
    *,
    now: datetime | None = None,
) -> RiskOverrides:
    """Load the newest non-expired override for every supported parameter."""
    reference = now or datetime.now(UTC)
    rows = (
        await db_session.execute(
            select(RiskOverride)
            .where(RiskOverride.is_active.is_(True))
            .order_by(RiskOverride.set_at.desc())
        )
    ).scalars().all()
    values: dict[str, float | int] = {}
    valid_fields = set(RiskOverrides.model_fields)
    for row in rows:
        if row.param_name not in valid_fields or row.param_name in values:
            continue
        expires_at = row.expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= reference:
                continue
        value: float | int = row.param_value
        if row.param_name == "max_concurrent_positions":
            value = int(value)
        values[row.param_name] = value
    return RiskOverrides.model_validate(values)


async def _submit_persisted_order(
    *,
    intent: OrderIntent,
    db_session: AsyncSession,
    deps: BotDeps,
    state: BotState,
    notify: NotifyFn,
) -> None:
    """Commit the idempotency record before Alpaca, then persist its response."""
    attempt = _make_order_attempt(intent)
    db_session.add(attempt)
    await db_session.commit()
    try:
        submission = await deps.alpaca.submit_order(intent)
    except Exception as exc:
        attempt.status = "failed_to_submit"
        attempt.error_message = str(exc)[:500]
        attempt.updated_at = datetime.now(UTC)
        await db_session.commit()
        message = f"Orden {intent.side.value} {intent.symbol} rechazada al enviar: {exc}"
        state.last_error = message
        await notify(f"⚠️ {message}")
        raise

    attempt.alpaca_order_id = submission.alpaca_order_id
    attempt.status = submission.status
    attempt.updated_at = datetime.now(UTC)
    await db_session.commit()


async def _record_risk_rejection(
    *,
    state: BotState,
    notify: NotifyFn,
    code: str,
    reason: str,
) -> None:
    message = f"{code}: {reason}"
    if state.last_risk_rejection != message:
        await notify(f"🛡 Orden bloqueada por riesgo: {message}")
    state.last_risk_rejection = message


async def _evaluate_symbol(
    symbol: str,
    bars: pd.DataFrame,
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    engine: StrategyEngine,
    overrides: RiskOverrides | None = None,
    calendar: CalendarCache | None = None,
) -> None:
    """Evaluate entry/exit rules for one symbol and submit orders if warranted.

    Shared by ``on_bar_event`` (stream bars) and ``close_job`` (daily REST bars).
    The caller is responsible for updating ``deps.bars_cache`` before calling this.

    Args:
        symbol:    Ticker to evaluate.
        bars:      Full bars DataFrame for this symbol (updated by caller).
        state:     BotState — must be ACTIVE and not paused (caller enforces).
        deps:      BotDeps with alpaca and session_factory.
        notify:    Async callable to send Telegram trade notification.
        engine:    StrategyEngine loaded with the active strategy.
        overrides: Active risk overrides (may be None).
        calendar:  CalendarCache for risk_manager check 3.
    """
    try:
        state.last_evaluation_at[symbol] = datetime.now(UTC)
        state.condition_snapshots[symbol] = "; ".join(engine.explain_entry(bars))
        # ── Exit check ────────────────────────────────────────────────────────
        # Check exit first (reduce existing position before opening new one).
        positions = await deps.alpaca.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        if position is not None:
            latest_high = float(bars["high"].iloc[-1])
            previous_high = state.position_highs.get(symbol, position.avg_entry_price)
            position_high = max(previous_high, latest_high, position.avg_entry_price)
            state.position_highs[symbol] = position_high
            try:
                from src.storage.runtime_state import persist_position_highs

                async with deps.session_factory() as runtime_db:
                    await persist_position_highs(runtime_db, state.position_highs)
            except Exception as exc:  # noqa: BLE001
                state.last_error = f"Failed to persist trailing peaks: {exc}"
                logger.error("Failed to persist trailing peaks: {}", exc)
            exit_signal = engine.evaluate_exit(
                symbol,
                bars,
                position,
                high_since_entry=position_high,
            )
            if exit_signal is not None:
                state.last_signal = f"SELL {symbol}: {exit_signal.reason}"
                intent = OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=abs(position.qty),
                    order_type=OrderType.MARKET,
                    client_order_id=make_client_order_id(None, symbol),
                    is_closing=True,
                    rule_trigger=exit_signal.reason,
                )
                async with deps.session_factory() as db_session:
                    effective_overrides = overrides or await _load_active_risk_overrides(
                        db_session
                    )
                    result = await validate_order(
                        intent=intent,
                        strategy=state.active_strategy,
                        overrides=effective_overrides,
                        alpaca=deps.alpaca,
                        asset_cache=deps.alpaca.asset_cache,
                        db_session=db_session,
                        bot_paused=state.bot_paused,
                        calendar=calendar,
                    )
                    if result.approved:
                        await _submit_persisted_order(
                            intent=intent,
                            db_session=db_session,
                            deps=deps,
                            state=state,
                            notify=notify,
                        )
                        msg = (
                            f"🔴 *SELL {symbol}* qty={intent.qty:.4g} "
                            f"({exit_signal.exit_type})\n"
                            f"_{exit_signal.reason}_"
                        )
                        await notify(msg)
                    else:
                        await db_session.commit()
                        await _record_risk_rejection(
                            state=state,
                            notify=notify,
                            code=result.code.value,
                            reason=result.reason,
                        )
                return  # Don't also enter when we just exited
        else:
            if state.position_highs.pop(symbol, None) is not None:
                try:
                    from src.storage.runtime_state import persist_position_highs

                    async with deps.session_factory() as runtime_db:
                        await persist_position_highs(runtime_db, state.position_highs)
                except Exception as exc:  # noqa: BLE001
                    state.last_error = f"Failed to persist trailing peaks: {exc}"
                    logger.error("Failed to persist trailing peaks: {}", exc)

        # ── Entry check ───────────────────────────────────────────────────────
        entry_signal = engine.evaluate_entry(symbol, bars)
        if entry_signal is None:
            return
        state.last_signal = f"BUY {symbol}: {entry_signal.reason}"

        # Simple position sizing: use max_position_pct * portfolio equity
        account = await deps.alpaca.get_account()
        equity = account.portfolio_value
        target_value = equity * state.active_strategy.position_sizing.max_position_pct / 100.0
        # Use close price for quantity estimation
        last_price = float(bars["close"].iloc[-1])
        qty = target_value / last_price if last_price > 0 else 0
        asset = await deps.alpaca.asset_cache.get(symbol)
        if not asset.fractionable:
            qty = float(int(qty))
        else:
            qty = round(qty, 9)
        if qty <= 0:
            message = f"No se puede dimensionar {symbol}: capital objetivo menor a una acción"
            logger.warning(message)
            state.last_risk_rejection = message
            await notify(f"🛡 {message}")
            return

        intent = OrderIntent(
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.MARKET,
            client_order_id=make_client_order_id(None, symbol),
            is_closing=False,
            rule_trigger=entry_signal.reason,
            estimated_price=last_price,
        )
        async with deps.session_factory() as db_session:
            effective_overrides = overrides or await _load_active_risk_overrides(db_session)
            result = await validate_order(
                intent=intent,
                strategy=state.active_strategy,
                overrides=effective_overrides,
                alpaca=deps.alpaca,
                asset_cache=deps.alpaca.asset_cache,
                db_session=db_session,
                bot_paused=state.bot_paused,
                calendar=calendar,
            )
            if result.approved:
                await _submit_persisted_order(
                    intent=intent,
                    db_session=db_session,
                    deps=deps,
                    state=state,
                    notify=notify,
                )
                msg = (
                    f"🟢 *BUY {symbol}* qty={qty:.4g} @ ~${last_price:.2f}\n"
                    f"_{entry_signal.reason}_"
                )
                await notify(msg)
            else:
                await db_session.commit()
                logger.info(
                    "Entry blocked by risk check {}: {}", result.code.value, result.reason
                )
                await _record_risk_rejection(
                    state=state,
                    notify=notify,
                    code=result.code.value,
                    reason=result.reason,
                )

    except Exception as exc:  # noqa: BLE001
        logger.error("_evaluate_symbol error for {}: {}", symbol, exc)
        message = f"Evaluación de {symbol}: {exc}"
        if state.last_error != message:
            await notify(f"⚠️ Error en {message}")
        state.last_error = message


async def on_bar_event(
    bar_event: BarEvent,
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    engine: StrategyEngine,
    overrides: RiskOverrides | None = None,
    calendar: CalendarCache | None = None,
) -> None:
    """Evaluate entry/exit rules on a new OHLCV bar and submit orders if warranted.

    Called by the StreamManager callback for every completed bar (minute bars
    from the WebSocket stream).  Daily strategies are evaluated in ``close_job``
    via REST instead, so this function skips them.

    Args:
        bar_event:  The received BarEvent (symbol + OHLCV data).
        state:      BotState — must be ACTIVE and not paused.
        deps:       BotDeps with alpaca, session_factory, and bars_cache.
        notify:     Async callable to send Telegram trade notification.
        engine:     StrategyEngine loaded with the active strategy.
        overrides:  Active risk overrides (may be None).
        calendar:   CalendarCache for risk_manager check 3.
    """
    if state.market_state != "ACTIVE":
        return
    if state.bot_paused:
        return
    if state.active_strategy is None:
        return

    # Daily strategies are evaluated in close_job via REST, not via stream bars
    if state.active_strategy.timeframe.value == "1D":
        return

    symbol = bar_event.bar.symbol
    if symbol not in state.active_strategy.universe:
        return
    state.last_bar_at[symbol] = bar_event.received_at

    event_date = bar_event.bar.timestamp.astimezone(ET).date()
    market_day = (calendar or deps.calendar).get(event_date)
    session_close = None
    if market_day is not None:
        session_close = (
            market_day.session_close
            if state.active_strategy.session == StrategySession.EXTENDED
            else market_day.close
        )
    completed_bars = deps.bar_aggregator.push(
        bar_event.bar,
        state.active_strategy.timeframe,
        state.active_strategy.session,
        session_close=session_close,
    )
    for completed in completed_bars:
        new_row = pd.DataFrame(
            [
                {
                    "open": completed.open,
                    "high": completed.high,
                    "low": completed.low,
                    "close": completed.close,
                    "volume": completed.volume,
                }
            ],
            index=pd.DatetimeIndex([completed.timestamp]),
        )
        existing = deps.bars_cache.get(symbol, pd.DataFrame())
        if completed.timestamp in existing.index:
            existing = existing.drop(index=completed.timestamp)
        deps.bars_cache[symbol] = pd.concat([existing, new_row]).sort_index().tail(
            engine.required_lookback_bars() * 2
        )
        await _evaluate_symbol(
            symbol=symbol,
            bars=deps.bars_cache[symbol],
            state=state,
            deps=deps,
            notify=notify,
            engine=engine,
            overrides=overrides,
            calendar=calendar,
        )


def _make_order_attempt(intent: OrderIntent) -> OrderAttempt:
    return OrderAttempt(
        client_order_id=intent.client_order_id,
        symbol=intent.symbol,
        side=intent.side.value,
        qty=intent.qty,
        order_type=intent.order_type.value,
        limit_price=intent.limit_price,
        stop_price=intent.stop_price,
        status="pending",
        rule_trigger=intent.rule_trigger,
        strategy_version_id=intent.strategy_version_id,
        submitted_at=datetime.now(UTC),
    )


# ── APScheduler factory ───────────────────────────────────────────────────────


def create_scheduler(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    calendar: CalendarCache,
    *,
    warmup_minutes: int = 5,
    eod_close_minutes_before: int = 5,
    summary_delay_minutes: int = 5,
    healthcheck_interval_minutes: int = 15,
) -> Any:
    """Build and return an AsyncIOScheduler with all jobs registered.

    Market state is reconciled every 30 seconds, so session transitions survive
    missed ticks, calendar refreshes, multi-day uptime, and mid-session restarts.

    Args:
        state:                       BotState instance.
        deps:                        BotDeps instance.
        notify:                      Async callable to send Telegram messages.
        calendar:                    CalendarCache for today's session times.
        warmup_minutes:              How many minutes before open to run warmup_job.
        eod_close_minutes_before:     Lead time for a close-all EOD policy.
        summary_delay_minutes:       How many minutes after close to run daily_summary_job.
        healthcheck_interval_minutes: Interval for the healthcheck job.

    Returns:
        Configured AsyncIOScheduler (not yet started).
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    deps.calendar = calendar
    scheduler = AsyncIOScheduler()

    # Healthcheck — runs every N minutes regardless of market state.
    scheduler.add_job(
        healthcheck_job,
        "interval",
        minutes=healthcheck_interval_minutes,
        args=[deps, notify, state],
        id="healthcheck",
        replace_existing=True,
        max_instances=1,
    )

    # Reconcile continuously instead of relying on one-shot jobs that disappear
    # after the first session.
    scheduler.add_job(
        reconcile_market_state_job,
        "interval",
        seconds=30,
        args=[state, deps, notify],
        kwargs={
            "warmup_minutes": warmup_minutes,
            "eod_close_minutes_before": eod_close_minutes_before,
            "summary_delay_minutes": summary_delay_minutes,
        },
        id="market_state_reconciler",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        refresh_calendar_job,
        "interval",
        hours=6,
        args=[state, deps, notify],
        id="calendar_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        reconcile_order_attempts_job,
        "interval",
        minutes=5,
        args=[state, deps, notify],
        id="order_reconciler",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    logger.info("Recurring market reconciliation and calendar refresh scheduled")
    return scheduler
