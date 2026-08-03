"""Telegram command handlers.

Every handler signature is ``async (update, context) -> None`` as required by
python-telegram-bot v21.

Authorization: every handler calls ``_check_auth()`` first.  Unauthorized
chat_ids receive a neutral one-emoji response (no information leakage) and
the attempt is logged.

State and dependencies are accessed via ``context.bot_data``:
    state: BotState = context.bot_data["state"]
    deps:  BotDeps  = context.bot_data["deps"]
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from src.broker.schemas import RiskOverrides
from src.llm.strategy_parser import ParseError, parse_strategy
from src.llm.summarizer import answer_ask
from src.storage.models import RiskOverride
from src.utils.market_hours import ET, is_market_open

# ── Helpers ───────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
📋 *Comandos disponibles:*

/start — Bienvenida y estado actual
/status — Estado del bot, mercado y P&L del día
/positions — Posiciones abiertas con P&L
/strategy — Ver/cambiar estrategia activa
/confirm — Confirmar nueva estrategia
/cancel — Cancelar nueva estrategia pendiente
/pause — Pausar ejecución de nuevas órdenes
/resume — Reanudar ejecución
/closeall — Cerrar todas las posiciones
/ask <pregunta> — Consulta libre (usa IA)
/setlimit <param> <valor> — Override de límite de riesgo
/diagnostics — Estado de calendario, streams, barras y última evaluación
/backtest [días] — Simular estrategia activa con históricos (30 días por defecto)
/help — Este mensaje

Parámetros válidos para /setlimit:
  max\\_position\\_pct, max\\_total\\_exposure\\_pct,
  stop\\_loss\\_pct, max\\_concurrent\\_positions\
"""

_VALID_SETLIMIT_PARAMS: frozenset[str] = frozenset(
    {"max_position_pct", "max_total_exposure_pct", "stop_loss_pct", "max_concurrent_positions"}
)


async def _check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Return True if the message comes from the authorized chat_id."""
    from src.telegram_bot.bot import BotDeps
    deps: BotDeps = context.bot_data["deps"]
    chat_id = update.effective_chat.id
    if chat_id != deps.authorized_chat_id:
        logger.warning("Unauthorized access attempt from chat_id={}", chat_id)
        await context.bot.send_message(chat_id=chat_id, text="👋")
        return False
    return True


async def _reply(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs) -> None:
    """Send a message to the authorized chat."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        **kwargs,
    )


def _md(value: object) -> str:
    """Escape special Markdown v1 characters in a dynamic value.

    Telegram's Markdown v1 parser treats ``_`` as italic delimiter and ``*``
    as bold delimiter anywhere in the string — including inside ``*...*``
    blocks.  Strategy names like ``etf_pullback_trend_filtered`` therefore
    break the parser.  This helper makes any dynamic value safe to embed.
    """
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("*", "\\*")
        .replace("`", "\\`")
        .replace("[", "\\[")
    )


# ── /start ────────────────────────────────────────────────────────────────────


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotState
    state: BotState = context.bot_data["state"]

    strategy_name = state.active_strategy.name if state.active_strategy else "ninguna"
    status_icon = "⏸" if state.bot_paused else "▶️"
    market_icon = {"IDLE": "🔴", "WARMUP": "⏰", "ACTIVE": "🟢"}.get(state.market_state, "❓")

    text = (
        f"👋 ¡Hola! Bot de trading Alpaca.\n\n"
        f"{status_icon} Bot: {'PAUSADO' if state.bot_paused else 'ACTIVO'}\n"
        f"{market_icon} Mercado: {state.market_state}\n"
        f"📊 Estrategia: {strategy_name}\n\n"
        f"Usa /help para ver todos los comandos."
    )
    await _reply(update, context, text)


# ── /status ───────────────────────────────────────────────────────────────────


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    # Fetch live account info (best-effort; show N/A on error)
    try:
        account = await deps.alpaca.get_account()
        equity_str = f"${account.portfolio_value:,.2f}"
        bp_str = f"${account.buying_power:,.2f}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Status: failed to fetch account: {}", exc)
        equity_str = "N/A"
        bp_str = "N/A"

    strategy_name = state.active_strategy.name if state.active_strategy else "ninguna"
    win_rate = (
        f"{100 * state.daily_win_count / state.daily_trade_count:.0f}%"
        if state.daily_trade_count > 0
        else "N/A"
    )

    status_icon = "⏸" if state.bot_paused else "▶️"
    market_icon = {"IDLE": "🔴", "WARMUP": "⏰", "ACTIVE": "🟢"}.get(state.market_state, "❓")

    text = (
        f"{status_icon} *Estado del bot:* {'PAUSADO' if state.bot_paused else 'ACTIVO'}\n"
        f"{market_icon} *Mercado:* {state.market_state}\n"
        f"📊 *Estrategia:* {_md(strategy_name)}\n\n"
        f"💰 *Equity:* {equity_str}\n"
        f"💵 *Buying power:* {bp_str}\n\n"
        f"📈 *P&L realizado hoy:* ${state.daily_realized_pnl:+,.2f}\n"
        f"🔢 *Operaciones hoy:* {state.daily_trade_count} (win rate: {win_rate})"
    )
    await _reply(update, context, text, parse_mode="Markdown")


# ── /positions ────────────────────────────────────────────────────────────────


async def handle_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps
    deps: BotDeps = context.bot_data["deps"]

    try:
        positions = await deps.alpaca.get_positions()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Positions: failed to fetch: {}", exc)
        await _reply(update, context, "❌ Error al obtener posiciones.")
        return

    if not positions:
        await _reply(update, context, "📭 No hay posiciones abiertas.")
        return

    lines = ["📊 *Posiciones abiertas:*\n"]
    for p in positions:
        sign = "🟢" if p.unrealized_pl >= 0 else "🔴"
        lines.append(
            f"{sign} *{p.symbol}* — qty={p.qty} @ ${p.avg_entry_price:.2f}\n"
            f"   Valor: ${p.market_value:,.2f} | P&L: ${p.unrealized_pl:+,.2f}"
        )
    await _reply(update, context, "\n".join(lines), parse_mode="Markdown")


# ── /strategy ─────────────────────────────────────────────────────────────────


async def handle_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    args = context.args or []

    if not args:
        # Show current strategy
        if state.active_strategy is None:
            await _reply(update, context, "📭 No hay estrategia activa.")
        else:
            s = state.active_strategy
            text = (
                f"📊 *Estrategia activa: {_md(s.name)}*\n\n"
                f"🕐 Timeframe: {s.timeframe.value}\n"
                f"🌐 Universe: {', '.join(s.universe)}\n"
                f"📅 Sesión: {s.session.value}\n"
                f"🎯 Horizonte: {s.horizon.value}\n"
                f"📉 Política EOD: {s.eod_policy.value}\n"
                f"🛑 Stop-loss: {s.exit_rules.stop_loss_pct}%\n"
                f"💼 Max posición: {s.position_sizing.max_position_pct}%\n"
                f"📊 Max exposición: {s.position_sizing.max_total_exposure_pct}%\n"
                f"🔢 Max posiciones: {s.position_sizing.max_concurrent_positions}"
            )
            await _reply(update, context, text, parse_mode="Markdown")
        return

    # Parse new strategy
    raw_text = " ".join(args)
    await _reply(update, context, "⏳ Procesando estrategia con IA…")

    try:
        new_strategy = await parse_strategy(raw_text, client=deps.llm_client)
    except ParseError as exc:
        await _reply(update, context, f"❌ {exc.user_message}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Strategy parse unexpected error: {}", exc)
        await _reply(update, context, "❌ Error inesperado al procesar la estrategia.")
        return

    # Store as pending with TTL
    state.pending_strategy = new_strategy
    state.pending_strategy_raw = raw_text
    state.pending_strategy_expires_at = datetime.now(UTC) + timedelta(
        seconds=deps.pending_strategy_ttl_seconds
    )

    # Format a preview for the user
    preview = json.dumps(new_strategy.model_dump(mode="json"), ensure_ascii=False, indent=2)
    # Telegram message limit is 4096 chars; truncate preview if needed
    if len(preview) > 3000:
        preview = preview[:3000] + "\n… (truncado)"

    text = (
        f"✅ Estrategia parseada: *{_md(new_strategy.name)}*\n\n"
        f"```json\n{preview}\n```\n\n"
        f"Usa /confirm para activar o /cancel para descartar.\n"
        f"⏳ Expira en {deps.pending_strategy_ttl_seconds // 60} min."
    )
    await _reply(update, context, text, parse_mode="Markdown")


# ── /confirm ──────────────────────────────────────────────────────────────────


async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotState
    state: BotState = context.bot_data["state"]

    if state.pending_strategy is None:
        await _reply(update, context, "❌ No hay estrategia pendiente de confirmación.")
        return

    # Check expiry
    now = datetime.now(UTC)
    if state.pending_strategy_expires_at and now > state.pending_strategy_expires_at:
        state.pending_strategy = None
        state.pending_strategy_expires_at = None
        await _reply(
            update, context,
            "⏰ La estrategia pendiente ha expirado. Vuelve a enviarla con /strategy."
        )
        return

    from src.telegram_bot.bot import BotDeps
    deps: BotDeps = context.bot_data["deps"]

    old_name = state.active_strategy.name if state.active_strategy else "ninguna"
    next_strategy = state.pending_strategy
    raw_input = state.pending_strategy_raw

    # Persist first. If this transaction fails, the running strategy and its
    # data subscriptions remain untouched and the pending proposal can be retried.
    from sqlalchemy import update as sa_update

    from src.storage.models import StrategyVersion

    try:
        async with deps.session_factory() as db:
            await db.execute(
                sa_update(StrategyVersion)
                .where(StrategyVersion.is_active.is_(True))
                .values(is_active=False, deactivated_at=now)
            )
            db.add(
                StrategyVersion(
                    name=next_strategy.name,
                    parsed_config=next_strategy.model_dump(mode="json"),
                    raw_input=raw_input,
                    is_active=True,
                    activated_at=now,
                )
            )
            await db.commit()
        logger.info("Strategy '{}' persisted to DB", next_strategy.name)
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"Failed to persist strategy: {exc}"
        logger.error("Failed to persist strategy to DB: {}", exc)
        await _reply(
            update,
            context,
            "❌ No se pudo guardar la estrategia; la estrategia activa no cambió. "
            "Puedes volver a intentar /confirm.",
        )
        return

    state.active_strategy = next_strategy
    state.pending_strategy = None
    state.pending_strategy_expires_at = None
    state.pending_strategy_raw = ""

    # A strategy change is also a market-data routing change. Never reuse bars
    # from another timeframe/universe, and update the live subscriptions now.
    deps.bars_cache.clear()
    deps.bar_aggregator.reset()
    subscription_notice = ""
    if deps.stream_manager is not None:
        try:
            added, removed = await deps.stream_manager.replace_bar_symbols(
                set(state.active_strategy.universe)
            )
            state.subscribed_symbols = deps.stream_manager.subscribed_symbols
            logger.info(
                "Strategy subscriptions updated: added={} removed={}",
                sorted(added),
                sorted(removed),
            )
        except Exception as exc:  # noqa: BLE001
            state.last_error = f"Failed to update subscriptions: {exc}"
            logger.error("Failed to update strategy subscriptions: {}", exc)
            subscription_notice = (
                "\n⚠️ No se pudieron actualizar las suscripciones; "
                "revisa /diagnostics."
            )

    if state.market_state in {"WARMUP", "ACTIVE"}:
        from src.scheduler.jobs import preload_strategy_bars

        loaded, total = await preload_strategy_bars(state, deps)
        if loaded < total:
            subscription_notice += f"\n⚠️ Warmup incompleto: {loaded}/{total} símbolos."

    await _reply(
        update, context,
        f"✅ Estrategia *{_md(state.active_strategy.name)}* activada.\n"
        f"(Anterior: {_md(old_name)})\n"
        f"Símbolos suscritos: {_md(', '.join(sorted(state.subscribed_symbols)) or 'ninguno')}"
        f"{subscription_notice}",
        parse_mode="Markdown",
    )


# ── /cancel ───────────────────────────────────────────────────────────────────


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotState
    state: BotState = context.bot_data["state"]

    if state.pending_strategy is None:
        await _reply(update, context, "ℹ️ No hay estrategia pendiente.")
        return

    name = state.pending_strategy.name
    state.pending_strategy = None
    state.pending_strategy_expires_at = None
    state.pending_strategy_raw = ""
    await _reply(update, context, f"🗑 Estrategia *{_md(name)}* cancelada.", parse_mode="Markdown")


# ── /pause ────────────────────────────────────────────────────────────────────


async def handle_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.storage.runtime_state import persist_bot_paused
    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    if state.bot_paused:
        await _reply(update, context, "⏸ El bot ya está pausado.")
        return

    # Failing closed is safer: pause in memory even if persistence is degraded.
    state.bot_paused = True
    try:
        async with deps.session_factory() as db:
            await persist_bot_paused(db, True)
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"No se pudo persistir /pause: {exc}"
        logger.error("Failed to persist paused state: {}", exc)
        await _reply(
            update,
            context,
            "⚠️ Bot pausado en memoria, pero no se pudo guardar el estado. "
            "No lo reinicies hasta revisar la base de datos.",
        )
        return
    logger.info("Bot paused by Telegram user chat_id={}", update.effective_chat.id)
    await _reply(update, context, "⏸ Bot pausado. No se enviarán nuevas órdenes.\nUsa /resume para reanudar.")


# ── /resume ───────────────────────────────────────────────────────────────────


async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.storage.runtime_state import persist_bot_paused
    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    if not state.bot_paused:
        await _reply(update, context, "▶️ El bot ya está activo.")
        return

    # Persist before reopening the order gate. A DB failure leaves the bot paused.
    try:
        async with deps.session_factory() as db:
            await persist_bot_paused(db, False)
    except Exception as exc:  # noqa: BLE001
        state.last_error = f"No se pudo persistir /resume: {exc}"
        logger.error("Failed to persist resumed state: {}", exc)
        await _reply(
            update,
            context,
            "❌ No se pudo guardar la reanudación; el bot permanece pausado.",
        )
        return
    state.bot_paused = False
    logger.info("Bot resumed by Telegram user chat_id={}", update.effective_chat.id)
    await _reply(update, context, "▶️ Bot reanudado. Volviendo a ejecutar órdenes.")


# ── /closeall ─────────────────────────────────────────────────────────────────


async def handle_closeall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    args = context.args or []

    if not args or args[0].lower() != "confirm":
        # First call: ask for confirmation
        state.awaiting_closeall_confirm = True
        await _reply(
            update, context,
            "⚠️ ¿Cerrar TODAS las posiciones?\n\n"
            "Escribe /closeall confirm para confirmar.\n"
            "Cualquier otro comando cancela esta acción."
        )
        return

    # Confirmed
    state.awaiting_closeall_confirm = False

    if state.market_state != "ACTIVE":
        # Market is closed — enqueue as PendingAction
        async with deps.session_factory() as session:
            from src.storage.models import PendingAction
            action = PendingAction(
                type="close_all",
                status="pending",
            )
            session.add(action)
            await session.commit()
        await _reply(
            update, context,
            "📋 Mercado cerrado. La orden de cierre está encolada para la próxima apertura."
        )
        return

    # Market is open — close immediately
    try:
        positions = await deps.alpaca.get_positions()
        if not positions:
            await _reply(update, context, "📭 No hay posiciones abiertas para cerrar.")
            return

        closed = []
        for pos in positions:
            await deps.alpaca.close_position(pos.symbol)
            closed.append(pos.symbol)

        await _reply(
            update, context,
            f"✅ Órdenes de cierre enviadas para: {', '.join(closed)}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("closeall error: {}", exc)
        await _reply(update, context, f"❌ Error al cerrar posiciones: {exc}")


# ── /ask ──────────────────────────────────────────────────────────────────────


async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps, BotState
    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]

    args = context.args or []
    if not args:
        await _reply(update, context, "❓ Uso: /ask <pregunta>")
        return

    question = " ".join(args)
    await _reply(update, context, "⏳ Consultando…")

    # Gather context for the LLM
    try:
        positions = await deps.alpaca.get_positions()
        account = await deps.alpaca.get_account()
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
        logger.warning("ask: failed to fetch positions/account: {}", exc)
        pos_dicts = []
        equity = 0.0

    answer = await answer_ask(
        question,
        recent_trades=[],   # Scheduler populates this; for now leave empty
        open_positions=pos_dicts,
        equity=equity,
        client=deps.llm_client,
    )
    await _reply(update, context, answer)


# ── /setlimit ─────────────────────────────────────────────────────────────────


async def handle_setlimit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps
    deps: BotDeps = context.bot_data["deps"]

    args = context.args or []
    if len(args) != 2:  # noqa: PLR2004
        await _reply(
            update, context,
            "❌ Uso: /setlimit <param> <valor>\n"
            f"Parámetros válidos: {', '.join(sorted(_VALID_SETLIMIT_PARAMS))}"
        )
        return

    param, raw_value = args[0].lower(), args[1]

    if param not in _VALID_SETLIMIT_PARAMS:
        await _reply(
            update, context,
            f"❌ Parámetro inválido: {param!r}\n"
            f"Válidos: {', '.join(sorted(_VALID_SETLIMIT_PARAMS))}"
        )
        return

    try:
        value = float(raw_value)
    except ValueError:
        await _reply(update, context, f"❌ Valor inválido: {raw_value!r} (debe ser un número)")
        return

    # Validate ranges
    if param == "max_concurrent_positions":
        if not (1 <= value <= 50):
            await _reply(update, context, f"❌ {param} debe estar entre 1 y 50")
            return
    else:
        if not (0 < value <= 100):
            await _reply(update, context, f"❌ {param} debe estar entre 0 y 100")
            return

    async with deps.session_factory() as session:
        override = RiskOverride(
            param_name=param,
            param_value=value,
            set_by=f"telegram:{update.effective_chat.id}",
            is_active=True,
        )
        session.add(override)
        await session.commit()

    await _reply(
        update, context,
        f"✅ Override guardado: *{_md(param)}* = {value}\n"
        f"_Activo hasta que se cambie o la estrategia se recargue._",
        parse_mode="Markdown",
    )


# ── /diagnostics ─────────────────────────────────────────────────────────────


async def handle_diagnostics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotDeps, BotState
    from src.utils.market_hours import next_close, next_open

    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]
    now = datetime.now(UTC)

    def fmt(value: datetime | None) -> str:
        return value.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S %Z") if value else "nunca"

    subscribed = sorted(state.subscribed_symbols)
    bar_lines = []
    for symbol in subscribed:
        received = state.last_bar_at.get(symbol)
        age = (now - received).total_seconds() if received else None
        age_text = f"{age:.0f}s" if age is not None else "sin barras"
        evaluated = state.last_evaluation_at.get(symbol)
        snapshot = state.condition_snapshots.get(symbol, "sin evaluación")
        bar_lines.append(
            f"  {symbol}: barra={age_text}, evaluación={fmt(evaluated)}\n"
            f"    condiciones: {_md(snapshot)}"
        )

    stream = deps.stream_manager
    stream_status = (
        f"datos={'up' if stream.is_data_stream_running else 'down'}, "
        f"órdenes={'up' if stream.is_trading_stream_running else 'down'}"
        if stream is not None
        else "no configurado"
    )
    nxt_open = next_open(now, deps.calendar) if deps.calendar else None
    nxt_close = next_close(now, deps.calendar) if deps.calendar else None
    strategy = state.active_strategy
    lines = [
        "🔎 *Diagnóstico operativo*",
        f"Estado: {state.market_state} | pausado={state.bot_paused}",
        f"Estrategia: {_md(strategy.name if strategy else 'ninguna')}",
        f"Timeframe: {strategy.timeframe.value if strategy else 'N/A'}",
        f"Streams: {stream_status}",
        f"Calendario actualizado: {fmt(state.last_calendar_refresh)}",
        f"Próxima apertura: {fmt(nxt_open)}",
        f"Próximo cierre: {fmt(nxt_close)}",
        f"Suscripciones: {_md(', '.join(subscribed) or 'ninguna')}",
    ]
    lines.extend(bar_lines or ["  Sin símbolos suscritos"])
    lines.extend(
        [
            f"Última señal: {_md(state.last_signal or 'ninguna')}",
            f"Último bloqueo: {_md(state.last_risk_rejection or 'ninguno')}",
            f"Último error: {_md(state.last_error or 'ninguno')}",
        ]
    )
    await _reply(update, context, "\n".join(lines), parse_mode="Markdown")


# ── /backtest ────────────────────────────────────────────────────────────────


async def handle_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run a historical simulation of the active strategy without placing orders."""
    if not await _check_auth(update, context):
        return

    import asyncio

    from src.backtest import BacktestConfig, format_backtest_report, run_backtest
    from src.telegram_bot.bot import BotDeps, BotState

    state: BotState = context.bot_data["state"]
    deps: BotDeps = context.bot_data["deps"]
    if state.active_strategy is None:
        await _reply(update, context, "❌ No hay estrategia activa para simular.")
        return
    if state.backtest_running:
        await _reply(update, context, "⏳ Ya hay una simulación en curso.")
        return

    args = context.args or []
    try:
        days = int(args[0]) if args else 30
    except ValueError:
        await _reply(update, context, "❌ Uso: /backtest [días], por ejemplo /backtest 30")
        return
    if not 5 <= days <= 730:
        await _reply(update, context, "❌ El periodo debe estar entre 5 y 730 días.")
        return

    end_date = datetime.now(ET).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    try:
        account = await deps.alpaca.get_account()
        initial_cash = account.portfolio_value
    except Exception as exc:  # noqa: BLE001
        logger.warning("Backtest: could not read account equity, using 100k: {}", exc)
        initial_cash = 100_000.0

    await _reply(
        update,
        context,
        f"🧪 Simulación iniciada para {state.active_strategy.name}: "
        f"{start_date} → {end_date}. No se enviarán órdenes.",
    )
    state.backtest_running = True

    async def execute() -> None:
        try:
            result = await run_backtest(
                state.active_strategy,
                deps.alpaca,
                BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash,
                ),
            )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=format_backtest_report(result),
            )
        except Exception as exc:  # noqa: BLE001
            state.last_error = f"Backtest failed: {exc}"
            logger.exception("Backtest failed")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ No se pudo completar la simulación: {exc}",
            )
        finally:
            state.backtest_running = False
            state.backtest_task = None

    state.backtest_task = asyncio.create_task(execute(), name="telegram_backtest")


# ── /help ─────────────────────────────────────────────────────────────────────


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return
    await _reply(update, context, _HELP_TEXT, parse_mode="Markdown")
