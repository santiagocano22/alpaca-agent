"""LLM-powered daily summary and /ask handler.

Both functions use claude-haiku-4-5 for cost efficiency.

Token budgets:
  - generate_daily_summary: ≤ 500 input tokens, max_tokens=300 for output.
  - answer_ask:             ≤ 800 input tokens, max_tokens=500 for output.

On any LLM error (network, API, malformed response) these functions return a
safe fallback string so the Telegram bot never crashes.  Errors are logged at
WARNING level.
"""
from __future__ import annotations

from datetime import date

from loguru import logger

from src.llm.client import LLMClient, LLMError

# ── Constants ─────────────────────────────────────────────────────────────────

HAIKU_MODEL = "claude-haiku-4-5"

_SUMMARY_SYSTEM = """\
Eres el asistente de un bot de trading. Recibes estadísticas compactas del día \
y produces un resumen conciso en español (máximo 250 palabras). \
Incluye: operaciones realizadas, P&L realizado, P&L no realizado, \
mejor y peor operación, win rate. Sé directo, usa formato Telegram (sin HTML). \
No inventes datos que no estén en el contexto.\
"""

_ASK_SYSTEM = """\
Eres el asistente de un bot de trading. Tienes acceso a las operaciones recientes \
y posiciones abiertas del usuario. Responde la pregunta del usuario de forma \
concisa y precisa en español (máximo 350 palabras). \
Si no puedes responder con los datos disponibles, dilo claramente. \
No inventes información.\
"""

_SUMMARY_FALLBACK = (
    "⚠️ No pude generar el resumen diario (error en el LLM). "
    "Consulta los logs para más detalles."
)
_ASK_FALLBACK = (
    "⚠️ No pude procesar tu pregunta en este momento (error en el LLM). "
    "Intenta de nuevo más tarde."
)


# ── Public API ────────────────────────────────────────────────────────────────


async def generate_daily_summary(
    *,
    trades_today: list[dict],
    open_positions: list[dict],
    realized_pnl: float,
    unrealized_pnl: float,
    equity: float,
    trade_date: date,
    client: LLMClient,
) -> str:
    """Generate a daily P&L summary using Haiku.

    Accepts pre-aggregated data (never the full trade log) to stay within
    the 500-input-token budget.

    Args:
        trades_today:    List of compact trade dicts (symbol, side, qty, price,
                         pnl, rule_trigger).  Caller should limit to ≤ 20 items.
        open_positions:  Compact position dicts (symbol, qty, market_value,
                         unrealized_pl).
        realized_pnl:    Total realized P&L for the day (USD).
        unrealized_pnl:  Total unrealized P&L across open positions (USD).
        equity:          Current portfolio equity (USD).
        trade_date:      The trading date being summarised.
        client:          LLMClient instance.

    Returns:
        A Telegram-ready summary string.  Returns a fallback message on error.
    """
    wins = sum(1 for t in trades_today if (t.get("pnl") or 0) > 0)
    total = len(trades_today)
    win_rate = f"{100 * wins / total:.0f}%" if total > 0 else "N/A"
    pnl_per_trade = sorted(
        [t for t in trades_today if t.get("pnl") is not None],
        key=lambda t: t.get("pnl", 0),
    )
    best = pnl_per_trade[-1] if pnl_per_trade else None
    worst = pnl_per_trade[0] if pnl_per_trade else None

    context = (
        f"Fecha: {trade_date}\n"
        f"Equity: ${equity:,.2f}\n"
        f"P&L realizado: ${realized_pnl:+,.2f}\n"
        f"P&L no realizado: ${unrealized_pnl:+,.2f}\n"
        f"Operaciones del día: {total} | Win rate: {win_rate}\n"
        f"Mejor operación: {_fmt_trade(best)}\n"
        f"Peor operación: {_fmt_trade(worst)}\n"
        f"Posiciones abiertas: {len(open_positions)}\n"
    )
    if open_positions:
        context += "Detalle posiciones:\n"
        for p in open_positions[:10]:  # cap at 10 to stay within token budget
            context += (
                f"  {p.get('symbol')} qty={p.get('qty')} "
                f"pnl={p.get('unrealized_pl', 0):+.2f}\n"
            )

    try:
        return await client.complete(
            model=HAIKU_MODEL,
            system=_SUMMARY_SYSTEM,
            user=context,
            max_tokens=300,
        )
    except LLMError as exc:
        logger.warning("generate_daily_summary LLM error: {}", exc)
        return _SUMMARY_FALLBACK
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_daily_summary unexpected error: {}", exc)
        return _SUMMARY_FALLBACK


async def answer_ask(
    question: str,
    *,
    recent_trades: list[dict],
    open_positions: list[dict],
    equity: float,
    client: LLMClient,
) -> str:
    """Answer a free-form /ask question using Haiku with trade context.

    Args:
        question:       The user's question (from /ask <question>).
        recent_trades:  Last ≤ 10 trades as compact dicts.
        open_positions: Current open positions as compact dicts.
        equity:         Current portfolio equity.
        client:         LLMClient instance.

    Returns:
        A Telegram-ready answer string.  Returns a fallback message on error.
    """
    trades_text = "\n".join(
        f"- {t.get('symbol')} {t.get('side')} {t.get('qty')} "
        f"@ {t.get('price', '?')} pnl={t.get('pnl', 'N/A')} motivo={t.get('rule_trigger', '')}"
        for t in recent_trades[:10]
    ) or "(sin operaciones recientes)"

    positions_text = "\n".join(
        f"- {p.get('symbol')} qty={p.get('qty')} value={p.get('market_value', '?')} "
        f"pnl={p.get('unrealized_pl', 0):+.2f}"
        for p in open_positions[:10]
    ) or "(sin posiciones abiertas)"

    context = (
        f"Equity actual: ${equity:,.2f}\n\n"
        f"Últimas operaciones:\n{trades_text}\n\n"
        f"Posiciones abiertas:\n{positions_text}\n\n"
        f"Pregunta del usuario: {question}"
    )

    try:
        return await client.complete(
            model=HAIKU_MODEL,
            system=_ASK_SYSTEM,
            user=context,
            max_tokens=500,
        )
    except LLMError as exc:
        logger.warning("answer_ask LLM error: {}", exc)
        return _ASK_FALLBACK
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer_ask unexpected error: {}", exc)
        return _ASK_FALLBACK


# ── Internal helpers ──────────────────────────────────────────────────────────


def _fmt_trade(trade: dict | None) -> str:
    if trade is None:
        return "N/A"
    return (
        f"{trade.get('symbol')} ({trade.get('side')}) "
        f"pnl={trade.get('pnl', 0):+.2f}"
    )
