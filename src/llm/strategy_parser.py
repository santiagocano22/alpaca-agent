"""LLM-powered natural-language strategy parser.

Calls claude-sonnet-4-5 ONCE per /strategy command to convert a free-text
trading strategy description into a validated Strategy pydantic object.

The LLM is instructed to:
  1. Output ONLY a JSON object — no markdown, no prose.
  2. Use the exact schema documented in the system prompt.
  3. Infer ``eod_policy`` from ``horizon`` when the user omits it
     (intraday → close_all, swing/position → hold).
  4. Output ``{"error": "<user-facing reason>"}`` if it cannot map the
     description to the schema.

ParseError carries a ``user_message`` attribute suitable for display in
Telegram (plain text, no internal details).
"""
from __future__ import annotations

import json
import re

from loguru import logger
from pydantic import ValidationError

from src.llm.client import LLMClient
from src.strategy.schema import Strategy

# ── Constants ─────────────────────────────────────────────────────────────────

SONNET_MODEL = "claude-sonnet-4-5"
_MAX_TOKENS = 1024   # strategy JSON is well under 1 KB

# Regex to strip ```json ... ``` or ``` ... ``` code fences that some models add
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a trading strategy parser. Convert the user's natural-language strategy
description into a single JSON object that matches the schema below.

RULES:
- Output ONLY the JSON object — no markdown, no explanation, no extra text.
- If you cannot map the description to the schema (ambiguous, impossible,
  or missing required information), output {"error": "<clear reason in Spanish>"}.
- Never invent rules that were not stated by the user.
- Infer `eod_policy` from `horizon` when the user does not mention it:
    intraday → "close_all", swing → "hold", position → "hold"
- `max_position_pct * max_concurrent_positions` MUST be <= `max_total_exposure_pct`.
  Choose sensible defaults if the user does not specify (e.g., max_position_pct=10,
  max_total_exposure_pct=50, max_concurrent_positions=4).
- `stop_loss_pct` is REQUIRED in exit_rules (range 0.1–50). Infer a default of 2.0
  if the user does not mention it.

SCHEMA:
{
  "name": "string (short strategy name)",
  "universe": ["TICKER", ...],
  "timeframe": "1Min" | "5Min" | "15Min" | "1H" | "1D",
  "session": "regular" | "extended",
  "horizon": "intraday" | "swing" | "position",
  "eod_policy": "close_all" | "hold",
  "entry_rules": {
    "logic": "AND" | "OR",
    "conditions": [
      {
        "left": <IndicatorRef or number>,
        "op": "<" | "<=" | ">" | ">=" | "==" | "crosses_above" | "crosses_below",
        "right": <IndicatorRef or number>
      },
      ...
    ]
  },
  "exit_rules": {
    "stop_loss_pct": <number 0.1–50>,
    "take_profit_pct": <number or null>,
    "trailing_stop_pct": <number or null>,
    "inverse_signal": <RuleGroup or null>
  },
  "position_sizing": {
    "max_position_pct": <number 0–100>,
    "max_total_exposure_pct": <number 0–100>,
    "max_concurrent_positions": <integer 1–50>
  }
}

IndicatorRef format:
  {"type": "rsi", "params": {"period": 14}}
  {"type": "ema", "params": {"period": 20}}
  {"type": "sma", "params": {"period": 50}}
  {"type": "macd", "params": {"fast": 12, "slow": 26}}
  {"type": "bbands", "params": {"period": 20}}
  {"type": "atr", "params": {"period": 14}}
  {"type": "vwap", "params": {}}
  {"type": "volume", "params": {"period": 20}}
  {"type": "price", "params": {}}
  {"type": "breakout", "params": {"lookback": 20}}

RuleGroup format (for nested logic or inverse_signal):
  {"logic": "AND"|"OR", "conditions": [...]}

EXAMPLE INPUT:
"Compra QQQ cuando el RSI 14 baje de 30 en velas de 15 minutos. Vende cuando
suba sobre 70. Máximo 10% del capital por operación, stop-loss 2%."

EXAMPLE OUTPUT:
{
  "name": "RSI Oversold QQQ",
  "universe": ["QQQ"],
  "timeframe": "15Min",
  "session": "regular",
  "horizon": "intraday",
  "eod_policy": "close_all",
  "entry_rules": {
    "logic": "AND",
    "conditions": [
      {"left": {"type": "rsi", "params": {"period": 14}}, "op": "<", "right": 30}
    ]
  },
  "exit_rules": {
    "stop_loss_pct": 2.0,
    "take_profit_pct": null,
    "trailing_stop_pct": null,
    "inverse_signal": {
      "logic": "AND",
      "conditions": [
        {"left": {"type": "rsi", "params": {"period": 14}}, "op": ">", "right": 70}
      ]
    }
  },
  "position_sizing": {
    "max_position_pct": 10.0,
    "max_total_exposure_pct": 50.0,
    "max_concurrent_positions": 4
  }
}
"""

# ── ParseError ────────────────────────────────────────────────────────────────


class ParseError(Exception):
    """Raised when the LLM response cannot be converted to a valid Strategy.

    ``user_message`` is safe to show directly in Telegram (no stack traces).
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_json_text(raw: str) -> str:
    """Strip optional markdown code fences and return the inner text."""
    m = _CODE_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _parse_raw_response(raw: str) -> dict:
    """Extract and parse the JSON from the LLM text response.

    Raises:
        ParseError: If the response is not valid JSON.
    """
    text = _extract_json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON response: {!r} (error: {})", text[:200], exc)
        raise ParseError(
            "No pude interpretar la respuesta del modelo como JSON. "
            "Por favor, intenta reformular la estrategia."
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────


async def parse_strategy(
    raw_text: str,
    *,
    client: LLMClient,
) -> Strategy:
    """Parse a natural-language strategy description into a Strategy object.

    This is the ONLY function in the trading loop allowed to call the LLM for
    strategy parsing (claude-sonnet-4-5, one call per /strategy command).

    Args:
        raw_text: Free-text strategy description from the user.
        client:   LLMClient instance (injected by caller; never created here
                  so the caller controls API key and retry settings).

    Returns:
        A fully validated Strategy object.

    Raises:
        ParseError: If the LLM cannot parse the strategy, returns an error JSON,
                    or the resulting JSON fails Pydantic validation.
    """
    logger.info("Parsing strategy from {} chars of input", len(raw_text))

    raw_response = await client.complete(
        model=SONNET_MODEL,
        system=_SYSTEM_PROMPT,
        user=raw_text,
        max_tokens=_MAX_TOKENS,
    )

    data = _parse_raw_response(raw_response)

    # LLM signals inability to parse via {"error": "..."}
    if "error" in data and len(data) == 1:
        user_msg = data["error"]
        logger.info("LLM rejected strategy input: {}", user_msg)
        raise ParseError(f"No pude parsear la estrategia: {user_msg}")

    try:
        strategy = Strategy.model_validate(data)
    except ValidationError as exc:
        # Produce a readable summary of Pydantic errors without exposing internals.
        errors_summary = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        logger.warning("Strategy JSON failed validation: {}", errors_summary)
        raise ParseError(
            f"La estrategia generada no es válida: {errors_summary}. "
            "Por favor, revisa los parámetros e intenta de nuevo."
        ) from exc

    logger.info(
        "Strategy parsed successfully: name={!r} universe={} timeframe={}",
        strategy.name,
        strategy.universe,
        strategy.timeframe.value,
    )
    return strategy
