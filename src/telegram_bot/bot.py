"""Telegram bot: Application factory and shared state/dependency containers.

Design:
- ``BotState`` holds all mutable in-process state (paused flag, active strategy,
  pending strategy awaiting confirmation, daily P&L tallies).  Updated in-place
  by command handlers and by the scheduler.
- ``BotDeps`` holds all injected dependencies (alpaca client, DB session factory,
  LLM client, authorized chat_id).  Read-only after construction.
- Handlers access both via ``context.bot_data["state"]`` and
  ``context.bot_data["deps"]``.  This makes them easily testable without wiring
  up a real Telegram server.
- ``create_application()`` builds the ``telegram.ext.Application`` with all
  command handlers registered and the bot_data pre-populated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Literal

from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.llm.client import LLMClient
from src.strategy.schema import Strategy
from src.telegram_bot import commands


# ── BotState ──────────────────────────────────────────────────────────────────


@dataclass
class BotState:
    """All mutable in-process bot state.

    Updated by command handlers (pause/resume, strategy activation) and by the
    scheduler jobs (market state transitions, daily P&L tallies).
    Thread-safety note: this object is only ever accessed from the asyncio event
    loop, so no locking is needed.
    """

    bot_paused: bool = False
    """True while /pause is active; no new orders will be sent."""

    market_state: Literal["IDLE", "WARMUP", "ACTIVE"] = "IDLE"
    """Current trading loop state, updated by the scheduler."""

    active_strategy: Strategy | None = None
    """Validated strategy currently used by the engine; None if no strategy loaded."""

    pending_strategy: Strategy | None = None
    """Strategy parsed by LLM, awaiting /confirm or /cancel."""

    pending_strategy_raw: str = ""
    """Original natural-language text of the pending strategy (for display)."""

    pending_strategy_expires_at: datetime | None = None
    """UTC expiry of the pending confirmation window (default: 10 min after parse)."""

    awaiting_closeall_confirm: bool = False
    """True after /closeall is issued; cleared after /closeall confirm or timeout."""

    # Daily aggregates — updated by the scheduler's daily-summary job and the
    # trade-update stream callback.
    daily_realized_pnl: float = 0.0
    daily_trade_count: int = 0
    daily_win_count: int = 0


# ── BotDeps ───────────────────────────────────────────────────────────────────


@dataclass
class BotDeps:
    """Injected dependencies for all Telegram command handlers.

    ``session_factory`` is an async context manager factory:
    ``async with deps.session_factory() as session: ...``
    """

    alpaca: Any
    """AlpacaClient instance."""

    session_factory: Callable
    """Async session factory: ``async_sessionmaker[AsyncSession]``."""

    llm_client: LLMClient
    """LLMClient for strategy parsing and /ask."""

    authorized_chat_id: int
    """The only Telegram chat_id allowed to control the bot."""

    pending_strategy_ttl_seconds: int = 600
    """Seconds a parsed strategy remains pending before it expires (default 10 min)."""

    bars_cache: dict = field(default_factory=dict)
    """Per-symbol rolling DataFrame of historical bars used by the strategy engine."""


# ── Application factory ───────────────────────────────────────────────────────


def create_application(
    bot_token: str,
    state: BotState,
    deps: BotDeps,
) -> Application:
    """Build and return the telegram Application with all handlers registered.

    Args:
        bot_token:  Telegram bot token from BotFather.
        state:      Shared BotState instance (injected into bot_data).
        deps:       Shared BotDeps instance (injected into bot_data).

    Returns:
        Configured Application ready to call ``run_polling()``.
    """
    app = Application.builder().token(bot_token).build()

    # Make state and deps available to every handler via context.bot_data.
    app.bot_data["state"] = state
    app.bot_data["deps"] = deps

    # Register command handlers.
    app.add_handler(CommandHandler("start", commands.handle_start))
    app.add_handler(CommandHandler("status", commands.handle_status))
    app.add_handler(CommandHandler("positions", commands.handle_positions))
    app.add_handler(CommandHandler("strategy", commands.handle_strategy))
    app.add_handler(CommandHandler("confirm", commands.handle_confirm))
    app.add_handler(CommandHandler("cancel", commands.handle_cancel))
    app.add_handler(CommandHandler("pause", commands.handle_pause))
    app.add_handler(CommandHandler("resume", commands.handle_resume))
    app.add_handler(CommandHandler("closeall", commands.handle_closeall))
    app.add_handler(CommandHandler("ask", commands.handle_ask))
    app.add_handler(CommandHandler("setlimit", commands.handle_setlimit))
    app.add_handler(CommandHandler("help", commands.handle_help))

    return app
