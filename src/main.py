"""Entry point: starts all services and runs the asyncio event loop.

Startup sequence:
  1. Load Settings from .env
  2. Safety guard: refuse to start if live_trading_confirmed=False and base_url is live
  3. Run Alembic migrations (alembic upgrade head) — ensures DB schema is current
  4. Open SQLAlchemy async engine + session factory
  5. Build AlpacaClient + AssetCache + AlpacaStreamManager
  6. Build LLMClient (Anthropic)
  7. Load active strategy from DB (if any) into BotState
  8. Build BotState + BotDeps
  9. Build TelegramApplication  (registers handlers; does not connect yet)
  10. Fetch market calendar for today +30 days
  11. Build APScheduler with recurring market/calendar/order reconciliation
  12. Register stream callbacks (minute bars → aggregation/evaluation, trades → DB)
  13. Start everything concurrently:
       - TelegramApplication polling
       - AlpacaStreamManager (trading + data streams)
       - APScheduler recurring jobs
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings
from src.utils.logger import setup_logger


async def _main() -> None:  # noqa: PLR0912, PLR0915
    settings = get_settings()
    setup_logger()

    logger.info("Starting trading bot (dry_run={}, live={})", settings.dry_run, settings.is_live)

    # ── Database ──────────────────────────────────────────────────────────────
    logger.info("Running Alembic migrations…")
    # Use the same Python interpreter that launched this process so alembic
    # is always resolved inside the active virtualenv (avoids FileNotFoundError
    # when the venv bin dir is not on PATH).
    alembic_bin = Path(sys.executable).parent / "alembic"
    result = subprocess.run(
        [str(alembic_bin), "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Alembic migration failed:\n{}", result.stderr)
        sys.exit(1)
    logger.info("Migrations OK")

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # ── Broker ────────────────────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream

    from src.broker.alpaca_client import AlpacaClient
    from src.broker.asset_cache import AssetCache
    from src.broker.rate_limiter import TokenBucketLimiter
    from src.broker.stream_manager import AlpacaStreamManager

    # Alpaca paper = True when the URL contains "paper"; live otherwise.
    is_paper = "paper-api" in settings.alpaca_base_url

    trading_client = TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_api_secret,
        paper=is_paper,
    )
    data_client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_api_secret,
    )
    # 200 calls/min Alpaca limit for paper trading (REST)
    rate_limiter = TokenBucketLimiter(rate=200 / 60, burst=20)

    alpaca = AlpacaClient(
        trading_client=trading_client,
        data_client=data_client,
        settings=settings,
        rate_limiter=rate_limiter,
    )
    # AssetCache wraps the trading_client directly (bypasses AlpacaClient retries
    # for read-only asset lookups with their own in-memory TTL cache).
    asset_cache = AssetCache(trading_client=trading_client)

    # Make asset_cache accessible from deps.alpaca for risk_manager
    alpaca.asset_cache = asset_cache  # type: ignore[attr-defined]

    # ── LLM ───────────────────────────────────────────────────────────────────
    from src.llm.client import LLMClient

    llm_client = LLMClient(api_key=settings.anthropic_api_key)

    # ── Load active strategy ──────────────────────────────────────────────────
    from sqlalchemy import select

    from src.storage.models import StrategyVersion
    from src.storage.runtime_state import load_runtime_state
    from src.strategy.schema import Strategy
    from src.telegram_bot.bot import BotDeps, BotState

    active_strategy: Strategy | None = None
    bot_paused = False
    position_highs: dict[str, float] = {}
    async with session_factory() as db:
        row = (
            await db.execute(
                select(StrategyVersion).where(StrategyVersion.is_active.is_(True))
            )
        ).scalars().first()
        if row:
            try:
                active_strategy = Strategy.model_validate(row.parsed_config)
                logger.info("Loaded active strategy: {!r}", active_strategy.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load strategy from DB: {}; starting without strategy", exc)

        runtime = await load_runtime_state(db)
        bot_paused = bool(runtime.bot_paused)
        position_highs = {
            symbol.upper(): float(high)
            for symbol, high in (runtime.position_highs or {}).items()
        }
        logger.info(
            "Loaded runtime state: paused={}, trailing_peaks={}",
            bot_paused,
            len(position_highs),
        )

    state = BotState(
        active_strategy=active_strategy,
        bot_paused=bot_paused,
        position_highs=position_highs,
    )
    deps = BotDeps(
        alpaca=alpaca,
        session_factory=session_factory,
        llm_client=llm_client,
        authorized_chat_id=settings.telegram_authorized_chat_id,
        pending_strategy_ttl_seconds=settings.pending_strategy_ttl_seconds,
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    from src.telegram_bot.bot import create_application

    tg_app = create_application(
        bot_token=settings.telegram_bot_token,
        state=state,
        deps=deps,
    )

    async def notify(text: str) -> None:
        try:
            await tg_app.bot.send_message(
                chat_id=settings.telegram_authorized_chat_id,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send Telegram notification: {}", exc)

    # ── Calendar ──────────────────────────────────────────────────────────────
    from datetime import UTC, datetime, timedelta

    from src.utils.market_hours import CalendarCache, MarketDay

    logger.info("Fetching market calendar…")
    calendar: CalendarCache = {}
    try:
        today = datetime.now(UTC).date()
        end = today + timedelta(days=30)
        raw_days = await alpaca.get_market_calendar(start=today, end=end)
        # Convert MarketCalendarDay (UTC-stored) → MarketDay (ET-stored) for CalendarCache
        calendar = {
            day.date: MarketDay(
                date=day.date,
                open=day.open_et,
                close=day.close_et,
            )
            for day in raw_days
        }
        logger.info("Calendar loaded: {} trading days", len(calendar))
        state.last_calendar_refresh = datetime.now(UTC)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch calendar: {}; scheduler will not fire session jobs", exc)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    from src.scheduler.jobs import create_scheduler

    scheduler = create_scheduler(
        state=state,
        deps=deps,
        notify=notify,
        calendar=calendar,
        warmup_minutes=settings.warmup_minutes_before_open,
        eod_close_minutes_before=settings.eod_close_minutes_before,
        healthcheck_interval_minutes=settings.healthcheck_interval_minutes,
    )
    scheduler.start()
    logger.info("APScheduler started")

    # ── Stream callbacks ──────────────────────────────────────────────────────
    from src.broker.schemas import BarEvent, TradeUpdateEvent
    from src.scheduler.jobs import on_bar_event
    from src.strategy.engine import StrategyEngine

    def _get_engine() -> StrategyEngine | None:
        if state.active_strategy is None:
            return None
        return StrategyEngine(state.active_strategy)

    async def bar_callback(event: BarEvent) -> None:
        eng = _get_engine()
        if eng is None:
            return
        await on_bar_event(
            bar_event=event,
            state=state,
            deps=deps,
            notify=notify,
            engine=eng,
            calendar=calendar,
        )

    async def trade_callback(event: TradeUpdateEvent) -> None:
        logger.info(
            "TradeUpdate: {} {} status={}",
            event.order.symbol,
            event.event,
            event.order.status,
        )
        async with session_factory() as db:
            from src.storage.models import OrderAttempt as OA
            from src.storage.models import Trade

            attempt = (
                await db.execute(
                    select(OA).where(OA.client_order_id == event.order.client_order_id)
                )
            ).scalars().first()
            if attempt is not None:
                attempt.alpaca_order_id = event.order.alpaca_order_id
                attempt.status = event.order.status
                attempt.updated_at = event.timestamp

            if event.event == "fill":
                existing_trade = (
                    await db.execute(
                        select(Trade).where(
                            Trade.client_order_id == event.order.client_order_id
                        )
                    )
                ).scalars().first()
                if existing_trade is None:
                    db.add(
                        Trade(
                            client_order_id=event.order.client_order_id,
                            order_attempt_id=attempt.id if attempt is not None else None,
                            symbol=event.order.symbol,
                            side=event.order.side.value,
                            qty=event.order.filled_qty or event.order.qty,
                            filled_price=event.order.filled_avg_price,
                            order_type=attempt.order_type if attempt is not None else "market",
                            status="filled",
                            rule_trigger=attempt.rule_trigger if attempt is not None else None,
                            strategy_version_id=(
                                attempt.strategy_version_id if attempt is not None else None
                            ),
                            filled_at=event.order.filled_at or event.timestamp,
                        )
                    )
                    state.daily_trade_count += 1
            await db.commit()

    # Build stream manager with the SDK stream objects
    universe = list(state.active_strategy.universe) if state.active_strategy else []
    trading_stream = TradingStream(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_api_secret,
        paper=is_paper,
    )
    from alpaca.data.enums import DataFeed

    data_stream = StockDataStream(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_api_secret,
        feed=DataFeed.IEX,
    )
    stream_mgr = AlpacaStreamManager(
        trading_stream=trading_stream,
        data_stream=data_stream,
        settings=settings,
    )
    deps.stream_manager = stream_mgr
    deps.calendar = calendar
    stream_mgr.subscribe_trade_updates(trade_callback)
    await stream_mgr.update_bar_subscriptions(bar_callback, set(universe))
    state.subscribed_symbols = stream_mgr.subscribed_symbols

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info("Starting Telegram polling and streams…")
    try:
        async with tg_app:
            await tg_app.initialize()
            await tg_app.start()
            await tg_app.updater.start_polling(drop_pending_updates=True)
            await stream_mgr.start()
            from src.scheduler.jobs import (
                reconcile_market_state_job,
                reconcile_order_attempts_job,
            )

            await reconcile_order_attempts_job(state, deps, notify)
            await reconcile_market_state_job(
                state,
                deps,
                notify,
                warmup_minutes=settings.warmup_minutes_before_open,
                eod_close_minutes_before=settings.eod_close_minutes_before,
            )
            logger.info("All services running. Press Ctrl+C to stop.")
            # Park here; Ctrl+C raises KeyboardInterrupt / CancelledError
            await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")
    finally:
        await stream_mgr.stop()
        scheduler.shutdown(wait=False)
        logger.info("Bot stopped cleanly")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
