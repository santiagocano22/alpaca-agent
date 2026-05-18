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
from src.utils.market_hours import is_market_open

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
    return str(value).replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*").replace("`", "\\`").replace("[", "\\[")


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

    old_name = state.active_strategy.name if state.active_strategy else "ninguna"
    state.active_strategy = state.pending_strategy
    state.pending_strategy = None
    state.pending_strategy_expires_at = None
    state.pending_strategy_raw = ""

    await _reply(
        update, context,
        f"✅ Estrategia *{_md(state.active_strategy.name)}* activada.\n"
        f"(Anterior: {_md(old_name)})\n\n"
        f"⚠️ El engine recargará la suscripción de streams en el próximo ciclo.",
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

    from src.telegram_bot.bot import BotState
    state: BotState = context.bot_data["state"]

    if state.bot_paused:
        await _reply(update, context, "⏸ El bot ya está pausado.")
        return

    state.bot_paused = True
    logger.info("Bot paused by Telegram user chat_id={}", update.effective_chat.id)
    await _reply(update, context, "⏸ Bot pausado. No se enviarán nuevas órdenes.\nUsa /resume para reanudar.")


# ── /resume ───────────────────────────────────────────────────────────────────


async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return

    from src.telegram_bot.bot import BotState
    state: BotState = context.bot_data["state"]

    if not state.bot_paused:
        await _reply(update, context, "▶️ El bot ya está activo.")
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


# ── /help ─────────────────────────────────────────────────────────────────────


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_auth(update, context):
        return
    await _reply(update, context, _HELP_TEXT, parse_mode="Markdown")
