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
from src.storage.models import AlertLog, OrderAttempt
from src.strategy.engine import StrategyEngine
from src.strategy.schema import EodPolicy
from src.telegram_bot.bot import BotDeps, BotState
from src.utils.market_hours import CalendarCache

# ── Type aliases ──────────────────────────────────────────────────────────────

NotifyFn = Callable[[str], Awaitable[None]]
"""Async callable that sends a Telegram message to the authorized chat."""

# ── Warmup helpers ────────────────────────────────────────────────────────────

_WARMUP_BARS_SAFETY_MULTIPLIER = 1.5
"""Fetch 50% more bars than required to account for weekends/holidays in lookback."""


async def _load_warmup_bars(
    alpaca,
    symbol: str,
    timeframe_str: str,
    required_bars: int,
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
        logger.info("Warmup: fetched {} bars for {} (needed {})", len(df), symbol, required_bars)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Warmup: failed to fetch bars for {}: {}", symbol, exc)
        return pd.DataFrame()


# ── Market state job functions ─────────────────────────────────────────────────


async def warmup_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    *,
    warmup_minutes: int = 5,
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
    msg = f"⏰ Mercado abre en {warmup_minutes} min. Preparando estrategia: {strategy_name}."
    await notify(msg)

    if state.active_strategy is None:
        logger.info("Warmup: no active strategy, skipping bar preload")
        return

    # Preload bars for all universe symbols.
    engine = StrategyEngine(state.active_strategy)
    needed = engine.required_lookback_bars()
    tf = state.active_strategy.timeframe.value

    for symbol in state.active_strategy.universe:
        df = await _load_warmup_bars(deps.alpaca, symbol, tf, needed)
        if not df.empty:
            deps.bars_cache[symbol] = df


async def open_job(
    state: BotState,
    notify: NotifyFn,
) -> None:
    """Transition WARMUP → ACTIVE at session open."""
    state.market_state = "ACTIVE"
    logger.info("Market opened — transitioning to ACTIVE")
    await notify("🟢 Mercado abierto. Bot operativo.")


async def close_job(
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
) -> None:
    """Transition ACTIVE → IDLE at session close.

    If the active strategy has ``eod_policy=close_all``, all open positions
    are closed with market orders before transitioning.
    """
    if state.active_strategy and state.active_strategy.eod_policy == EodPolicy.CLOSE_ALL:
        logger.info("EOD policy=close_all: closing all positions")
        try:
            positions = await deps.alpaca.get_positions()
            for pos in positions:
                await deps.alpaca.close_position(pos.symbol)
                logger.info("EOD: closed position {}", pos.symbol)
        except Exception as exc:  # noqa: BLE001
            logger.error("EOD close_all error: {}", exc)

    # For daily strategies: fetch today's completed bar and evaluate entry/exit signals
    if (
        state.active_strategy is not None
        and state.active_strategy.timeframe.value == "1D"
        and not state.bot_paused
    ):
        from datetime import UTC, datetime, timedelta

        from alpaca.data.timeframe import TimeFrame

        from src.broker.schemas import RiskOverrides

        engine = StrategyEngine(state.active_strategy)
        overrides = RiskOverrides()
        today_end = datetime.now(UTC)
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
                    overrides=overrides,
                    calendar=None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("close_job: error evaluating {} signals: {}", symbol, exc)

    state.market_state = "IDLE"
    logger.info("Market closed — transitioning to IDLE")
    await notify("🔴 Mercado cerrado. Generando resumen del día…")


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

    summary = await generate_daily_summary(
        trades_today=[],      # Detailed trades not tracked here; tally via state
        open_positions=pos_dicts,
        realized_pnl=state.daily_realized_pnl,
        unrealized_pnl=sum(p.get("unrealized_pl", 0) for p in pos_dicts),
        equity=equity,
        trade_date=datetime.now(UTC).date(),
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
    except Exception as exc:  # noqa: BLE001
        logger.error("Healthcheck: Alpaca error: {}", exc)
        await notify(f"⚠️ ALERTA: Error de conexión con Alpaca: {exc}")


# ── Bar event evaluation ───────────────────────────────────────────────────────


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
        # ── Exit check ────────────────────────────────────────────────────────
        # Check exit first (reduce existing position before opening new one).
        positions = await deps.alpaca.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        if position is not None:
            exit_signal = engine.evaluate_exit(symbol, bars, position)
            if exit_signal is not None:
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
                    result = await validate_order(
                        intent=intent,
                        strategy=state.active_strategy,
                        overrides=overrides,
                        alpaca=deps.alpaca,
                        asset_cache=deps.alpaca.asset_cache,
                        db_session=db_session,
                        bot_paused=state.bot_paused,
                        calendar=calendar,
                    )
                    if result.approved:
                        attempt = _make_order_attempt(intent)
                        db_session.add(attempt)
                        await db_session.flush()
                        await deps.alpaca.submit_order(intent)
                        msg = (
                            f"🔴 *SELL {symbol}* qty={intent.qty:.4g} "
                            f"({exit_signal.exit_type})\n"
                            f"_{exit_signal.reason}_"
                        )
                        await notify(msg)
                return  # Don't also enter when we just exited

        # ── Entry check ───────────────────────────────────────────────────────
        entry_signal = engine.evaluate_entry(symbol, bars)
        if entry_signal is None:
            return

        # Simple position sizing: use max_position_pct * portfolio equity
        account = await deps.alpaca.get_account()
        equity = account.portfolio_value
        target_value = equity * state.active_strategy.position_sizing.max_position_pct / 100.0
        # Use close price for quantity estimation
        last_price = float(bars["close"].iloc[-1])
        qty = target_value / last_price if last_price > 0 else 0
        if qty <= 0:
            logger.warning("_evaluate_symbol: computed qty=0 for {} — skipping", symbol)
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
            result = await validate_order(
                intent=intent,
                strategy=state.active_strategy,
                overrides=overrides,
                alpaca=deps.alpaca,
                asset_cache=deps.alpaca.asset_cache,
                db_session=db_session,
                bot_paused=state.bot_paused,
                calendar=calendar,
            )
            if result.approved:
                attempt = _make_order_attempt(intent)
                db_session.add(attempt)
                await db_session.flush()
                await deps.alpaca.submit_order(intent)
                msg = (
                    f"🟢 *BUY {symbol}* qty={qty:.4g} @ ~${last_price:.2f}\n"
                    f"_{entry_signal.reason}_"
                )
                await notify(msg)
            else:
                logger.info(
                    "Entry blocked by risk check {}: {}", result.code.value, result.reason
                )

    except Exception as exc:  # noqa: BLE001
        logger.error("_evaluate_symbol error for {}: {}", symbol, exc)


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

    # Update in-memory bars cache with the new bar
    new_row = pd.DataFrame(
        [
            {
                "open": bar_event.bar.open,
                "high": bar_event.bar.high,
                "low": bar_event.bar.low,
                "close": bar_event.bar.close,
                "volume": bar_event.bar.volume,
            }
        ],
        index=pd.DatetimeIndex([bar_event.bar.timestamp]),
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
    summary_delay_minutes: int = 5,
    healthcheck_interval_minutes: int = 15,
) -> Any:
    """Build and return an AsyncIOScheduler with all jobs registered.

    Jobs are scheduled for today's market session.  To handle the next day,
    ``schedule_daily_jobs()`` must be called at midnight or at bot startup.

    Args:
        state:                       BotState instance.
        deps:                        BotDeps instance.
        notify:                      Async callable to send Telegram messages.
        calendar:                    CalendarCache for today's session times.
        warmup_minutes:              How many minutes before open to run warmup_job.
        summary_delay_minutes:       How many minutes after close to run daily_summary_job.
        healthcheck_interval_minutes: Interval for the healthcheck job.

    Returns:
        Configured AsyncIOScheduler (not yet started).
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()

    # Healthcheck — runs every N minutes regardless of market state
    scheduler.add_job(
        healthcheck_job,
        "interval",
        minutes=healthcheck_interval_minutes,
        args=[deps, notify],
        id="healthcheck",
        replace_existing=True,
    )

    # Schedule today's market session jobs
    _schedule_session_jobs(
        scheduler,
        state=state,
        deps=deps,
        notify=notify,
        calendar=calendar,
        warmup_minutes=warmup_minutes,
        summary_delay_minutes=summary_delay_minutes,
    )

    return scheduler


def _schedule_session_jobs(
    scheduler,
    state: BotState,
    deps: BotDeps,
    notify: NotifyFn,
    calendar: CalendarCache,
    *,
    warmup_minutes: int,
    summary_delay_minutes: int,
) -> None:
    """Add today's warmup / open / close / summary jobs to the scheduler."""
    from src.utils.market_hours import next_open, next_close

    now = datetime.now(UTC)
    open_time = next_open(now, calendar)
    close_time = next_close(now, calendar)

    if open_time is None or close_time is None:
        logger.info("No trading session found in calendar; only healthcheck scheduled")
        return

    warmup_time = open_time - timedelta(minutes=warmup_minutes)
    summary_time = close_time + timedelta(minutes=summary_delay_minutes)

    if warmup_time > now:
        scheduler.add_job(
            warmup_job,
            "date",
            run_date=warmup_time,
            args=[state, deps, notify],
            kwargs={"warmup_minutes": warmup_minutes},
            id="warmup",
            replace_existing=True,
        )

    if open_time > now:
        scheduler.add_job(
            open_job,
            "date",
            run_date=open_time,
            args=[state, notify],
            id="open",
            replace_existing=True,
        )

    if close_time > now:
        scheduler.add_job(
            close_job,
            "date",
            run_date=close_time,
            args=[state, deps, notify],
            id="close",
            replace_existing=True,
        )

    if summary_time > now:
        scheduler.add_job(
            daily_summary_job,
            "date",
            run_date=summary_time,
            args=[state, deps, notify],
            id="daily_summary",
            replace_existing=True,
        )

    logger.info(
        "Session jobs scheduled: warmup={} open={} close={} summary={}",
        warmup_time,
        open_time,
        close_time,
        summary_time,
    )
