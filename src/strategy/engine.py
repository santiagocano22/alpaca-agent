"""Strategy evaluation engine.

Pure, deterministic: no I/O, no LLM calls, no Alpaca calls.
Receives already-loaded OHLCV DataFrames and evaluates the strategy's
rule tree against them.

DataFrame contract (for both evaluate_entry and evaluate_exit):
    Columns: open, high, low, close, volume  (numeric)
    Index:   pd.DatetimeIndex (tz-aware, required for VWAP; optional otherwise)
    Length:  >= self.required_lookback_bars()

Exit priority (highest to lowest): stop_loss > take_profit > trailing_stop > inverse_signal.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from src.broker.schemas import PositionSnapshot
from src.strategy.exceptions import InsufficientHistoryError
from src.strategy.indicators import (
    atr,
    bbands,
    breakout,
    ema,
    macd,
    price,
    rsi,
    sma,
    vwap,
    volume_avg,
)
from src.strategy.schema import (
    ComparisonOp,
    Condition,
    EntrySignal,
    ExitSignal,
    ExitRules,
    IndicatorRef,
    IndicatorType,
    RuleGroup,
    Strategy,
)

# ── Lookback period extractor ─────────────────────────────────────────────────


def _period_for_ref(ref: IndicatorRef) -> int:
    """Return the dominant warmup period (in bars) for a single IndicatorRef."""
    p = ref.params
    t = ref.type
    if t == IndicatorType.RSI:
        return int(p.get("period", 14))
    if t in (IndicatorType.EMA, IndicatorType.SMA):
        return int(p.get("period", 20))
    if t == IndicatorType.MACD:
        return max(int(p.get("fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9)))
    if t == IndicatorType.BBANDS:
        return int(p.get("period", 20))
    if t == IndicatorType.ATR:
        return int(p.get("period", 14)) + 1  # +1 for prev-close shift
    if t == IndicatorType.VOLUME:
        return int(p.get("period", 20))
    if t == IndicatorType.BREAKOUT:
        return int(p.get("lookback", 20))
    # VWAP, PRICE: no fixed period
    return 0


def _collect_refs(node: Union[RuleGroup, Condition]) -> list[IndicatorRef]:
    """Recursively collect all IndicatorRef leaves in a rule tree."""
    if isinstance(node, RuleGroup):
        refs: list[IndicatorRef] = []
        for child in node.conditions:
            refs.extend(_collect_refs(child))
        return refs
    # Condition
    refs = []
    if isinstance(node.left, IndicatorRef):
        refs.append(node.left)
    if isinstance(node.right, IndicatorRef):
        refs.append(node.right)
    return refs


# ── Indicator computation ─────────────────────────────────────────────────────


def _compute_indicator(ref: IndicatorRef, bars: pd.DataFrame) -> pd.Series:
    """Dispatch an IndicatorRef to the corresponding indicator function."""
    p = ref.params
    t = ref.type

    if t == IndicatorType.RSI:
        return rsi(bars, period=int(p.get("period", 14)))
    if t == IndicatorType.EMA:
        return ema(bars, period=int(p.get("period", 20)), column=p.get("column", "close"))
    if t == IndicatorType.SMA:
        return sma(bars, period=int(p.get("period", 20)), column=p.get("column", "close"))
    if t == IndicatorType.MACD:
        df_macd = macd(
            bars,
            fast=int(p.get("fast", 12)),
            slow=int(p.get("slow", 26)),
            signal=int(p.get("signal", 9)),
        )
        return df_macd[p.get("column", "macd")]
    if t == IndicatorType.BBANDS:
        df_bb = bbands(bars, period=int(p.get("period", 20)), std_dev=float(p.get("std_dev", 2.0)))
        return df_bb[p.get("column", "middle")]
    if t == IndicatorType.ATR:
        return atr(bars, period=int(p.get("period", 14)))
    if t == IndicatorType.VWAP:
        return vwap(bars)
    if t == IndicatorType.VOLUME:
        return volume_avg(bars, period=int(p.get("period", 20)))
    if t == IndicatorType.PRICE:
        return price(bars, ohlc=p.get("ohlc", "close"))
    if t == IndicatorType.BREAKOUT:
        df_brk = breakout(bars, lookback=int(p.get("lookback", 20)))
        return df_brk[p.get("column", "high_n")]
    raise ValueError(f"Unknown IndicatorType: {t}")  # pragma: no cover


def _as_series(val: Union[IndicatorRef, float], bars: pd.DataFrame) -> pd.Series:
    """Return a constant Series for a float or compute the indicator Series."""
    if isinstance(val, (int, float)):
        return pd.Series(float(val), index=bars.index)
    return _compute_indicator(val, bars)


# ── Rule evaluation ───────────────────────────────────────────────────────────


def _eval_condition(cond: Condition, bars: pd.DataFrame) -> bool:
    """Evaluate a single Condition against the last bar of ``bars``.

    Returns False (never raises) if any value is NaN.
    Crossing operators check bar[−2] vs bar[−1] and return False if < 2 bars.
    """
    left_s = _as_series(cond.left, bars)
    right_s = _as_series(cond.right, bars)
    op = cond.op

    if op in (ComparisonOp.CROSSES_ABOVE, ComparisonOp.CROSSES_BELOW):
        if len(bars) < 2:
            return False
        transitions = min(cond.lookback_bars, len(bars) - 1)
        for back in range(1, transitions + 1):
            cur_idx = -back
            prev_idx = cur_idx - 1
            l_cur, l_prev = left_s.iloc[cur_idx], left_s.iloc[prev_idx]
            r_cur, r_prev = right_s.iloc[cur_idx], right_s.iloc[prev_idx]
            if any(pd.isna(v) for v in (l_cur, l_prev, r_cur, r_prev)):
                continue
            if op == ComparisonOp.CROSSES_ABOVE and l_prev <= r_prev and l_cur > r_cur:
                return True
            if op == ComparisonOp.CROSSES_BELOW and l_prev >= r_prev and l_cur < r_cur:
                return True
        return False

    l_val = left_s.iloc[-1]
    r_val = right_s.iloc[-1]
    if pd.isna(l_val) or pd.isna(r_val):
        return False

    if op == ComparisonOp.LT:
        return bool(l_val < r_val)
    if op == ComparisonOp.LE:
        return bool(l_val <= r_val)
    if op == ComparisonOp.GT:
        return bool(l_val > r_val)
    if op == ComparisonOp.GE:
        return bool(l_val >= r_val)
    if op == ComparisonOp.EQ:
        return bool(l_val == r_val)
    raise ValueError(f"Unhandled op: {op}")  # pragma: no cover


def _eval_group(group: RuleGroup, bars: pd.DataFrame) -> bool:
    """Recursively evaluate a RuleGroup tree."""
    results = [
        _eval_group(item, bars) if isinstance(item, RuleGroup) else _eval_condition(item, bars)
        for item in group.conditions
    ]
    return all(results) if group.logic == "AND" else any(results)


# ── Description helpers ───────────────────────────────────────────────────────


def _desc_val(v: Union[IndicatorRef, float]) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    if not v.params:
        return v.type.value
    params = ", ".join(f"{key}={value}" for key, value in v.params.items())
    return f"{v.type.value}({params})"


def _desc_condition(c: Condition) -> str:
    return f"{_desc_val(c.left)} {c.op.value} {_desc_val(c.right)}"


def _desc_group(g: RuleGroup) -> str:
    parts = [
        _desc_group(item) if isinstance(item, RuleGroup) else _desc_condition(item)
        for item in g.conditions
    ]
    sep = f" {g.logic} "
    inner = sep.join(parts)
    return f"({inner})" if len(parts) > 1 else inner


# ── StrategyEngine ────────────────────────────────────────────────────────────


class StrategyEngine:
    """Evaluates entry and exit rules for a given Strategy."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
        self._required: int | None = None  # cached

    def required_lookback_bars(self) -> int:
        """Minimum bars needed for all indicators to be fully warmed up.

        Calculated as ``max_indicator_period × 2`` across all IndicatorRefs
        referenced in entry_rules and exit_rules.inverse_signal.
        Minimum of 2 to support crossing operators.
        """
        if self._required is not None:
            return self._required

        refs = _collect_refs(self._strategy.entry_rules)
        if self._strategy.exit_rules.inverse_signal is not None:
            refs.extend(_collect_refs(self._strategy.exit_rules.inverse_signal))

        if not refs:
            self._required = 2
        else:
            max_period = max(_period_for_ref(r) for r in refs)
            self._required = max(max_period * 2, 2)

        return self._required

    def evaluate_entry(self, symbol: str, bars: pd.DataFrame) -> EntrySignal | None:
        """Evaluate entry_rules against the last bar of ``bars``.

        Returns an EntrySignal if all conditions are met, None otherwise.

        Raises:
            InsufficientHistoryError: if ``len(bars) < required_lookback_bars()``.
        """
        self._check_history(bars)

        if not _eval_group(self._strategy.entry_rules, bars):
            return None

        reason = _desc_group(self._strategy.entry_rules)
        bar_ts = self._bar_timestamp(bars)
        logger.debug("Entry signal for {}: {}", symbol, reason)
        return EntrySignal(symbol=symbol, reason=reason, bar_timestamp=bar_ts)

    def explain_entry(self, bars: pd.DataFrame) -> list[str]:
        """Return current values and outcomes for every atomic entry condition."""
        self._check_history(bars)
        snapshots: list[str] = []

        def walk(node: Union[RuleGroup, Condition]) -> None:
            if isinstance(node, RuleGroup):
                for child in node.conditions:
                    walk(child)
                return
            left = _as_series(node.left, bars).iloc[-1]
            right = _as_series(node.right, bars).iloc[-1]
            outcome = _eval_condition(node, bars)
            left_text = "NaN" if pd.isna(left) else f"{float(left):.6g}"
            right_text = "NaN" if pd.isna(right) else f"{float(right):.6g}"
            snapshots.append(
                f"{_desc_val(node.left)}={left_text} {node.op.value} "
                f"{_desc_val(node.right)}={right_text} → {outcome}"
            )

        walk(self._strategy.entry_rules)
        return snapshots

    def entry_condition_outcomes(self, bars: pd.DataFrame) -> list[tuple[str, bool]]:
        """Return stable condition labels and their current boolean outcomes.

        Unlike :meth:`explain_entry`, labels do not contain changing indicator
        values, which makes this suitable for aggregating backtest diagnostics.
        """
        self._check_history(bars)
        outcomes: list[tuple[str, bool]] = []

        def walk(node: Union[RuleGroup, Condition]) -> None:
            if isinstance(node, RuleGroup):
                for child in node.conditions:
                    walk(child)
                return
            outcomes.append((_desc_condition(node), _eval_condition(node, bars)))

        walk(self._strategy.entry_rules)
        return outcomes

    def evaluate_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: PositionSnapshot,
        *,
        high_since_entry: float | None = None,
    ) -> ExitSignal | None:
        """Evaluate exit rules against the last bar of ``bars``.

        Priority order: stop_loss > take_profit > trailing_stop > inverse_signal.

        Returns an ExitSignal if any exit criterion is met, None otherwise.

        Raises:
            InsufficientHistoryError: if ``len(bars) < required_lookback_bars()``.
        """
        self._check_history(bars)

        current_price = float(bars["close"].iloc[-1])
        entry_price = float(position.avg_entry_price)
        er: ExitRules = self._strategy.exit_rules
        bar_ts = self._bar_timestamp(bars)

        # 1. Stop loss (mandatory, highest priority)
        sl_price = entry_price * (1.0 - er.stop_loss_pct / 100.0)
        if current_price <= sl_price:
            reason = (
                f"Stop loss triggered: close {current_price:.4f} ≤ "
                f"stop {sl_price:.4f} (entry {entry_price:.4f} - {er.stop_loss_pct}%)"
            )
            logger.debug("Exit[stop_loss] for {}: {}", symbol, reason)
            return ExitSignal(
                symbol=symbol, reason=reason, exit_type="stop_loss", bar_timestamp=bar_ts
            )

        # 2. Take profit
        if er.take_profit_pct is not None:
            tp_price = entry_price * (1.0 + er.take_profit_pct / 100.0)
            if current_price >= tp_price:
                reason = (
                    f"Take profit triggered: close {current_price:.4f} ≥ "
                    f"target {tp_price:.4f} (entry {entry_price:.4f} + {er.take_profit_pct}%)"
                )
                logger.debug("Exit[take_profit] for {}: {}", symbol, reason)
                return ExitSignal(
                    symbol=symbol, reason=reason, exit_type="take_profit", bar_timestamp=bar_ts
                )

        # 3. Trailing stop
        if er.trailing_stop_pct is not None:
            peak = (
                high_since_entry
                if high_since_entry is not None
                else float(bars["high"].max())
            )
            trail_price = peak * (1.0 - er.trailing_stop_pct / 100.0)
            if current_price <= trail_price:
                reason = (
                    f"Trailing stop triggered: close {current_price:.4f} ≤ "
                    f"trail {trail_price:.4f} "
                    f"(peak {peak:.4f} - {er.trailing_stop_pct}%)"
                )
                logger.debug("Exit[trailing_stop] for {}: {}", symbol, reason)
                return ExitSignal(
                    symbol=symbol,
                    reason=reason,
                    exit_type="trailing_stop",
                    bar_timestamp=bar_ts,
                )

        # 4. Inverse signal
        if er.inverse_signal is not None and _eval_group(er.inverse_signal, bars):
            reason = f"Inverse signal: {_desc_group(er.inverse_signal)}"
            logger.debug("Exit[inverse_signal] for {}: {}", symbol, reason)
            return ExitSignal(
                symbol=symbol, reason=reason, exit_type="inverse_signal", bar_timestamp=bar_ts
            )

        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_history(self, bars: pd.DataFrame) -> None:
        required = self.required_lookback_bars()
        got = len(bars)
        if got < required:
            raise InsufficientHistoryError(
                f"Need {required} bars, got {got}",
                required=required,
                got=got,
            )

    @staticmethod
    def _bar_timestamp(bars: pd.DataFrame) -> datetime:
        last = bars.index[-1]
        if isinstance(last, pd.Timestamp):
            return last.to_pydatetime()
        return datetime.now(UTC)
