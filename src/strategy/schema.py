"""Strategy schema — pure Pydantic value objects.

The LLM parser (module d) MUST produce objects that satisfy every validator
here.  If it generates an IndicatorRef without required params or a Strategy
with an incoherent timeframe/horizon pair, Pydantic raises ValidationError
before the engine ever sees the data.

All enums are closed lists; adding a new indicator or operator requires a
deliberate schema change, not just an LLM instruction.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class Timeframe(str, enum.Enum):
    M1 = "1Min"
    M5 = "5Min"
    M15 = "15Min"
    H1 = "1H"
    D1 = "1D"


class Session(str, enum.Enum):
    REGULAR = "regular"
    EXTENDED = "extended"
    CRYPTO_24_7 = "24/7"


class Horizon(str, enum.Enum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


class EodPolicy(str, enum.Enum):
    CLOSE_ALL = "close_all"
    HOLD = "hold"


class IndicatorType(str, enum.Enum):
    RSI = "rsi"
    EMA = "ema"
    SMA = "sma"
    MACD = "macd"
    BBANDS = "bbands"
    ATR = "atr"
    VWAP = "vwap"
    VOLUME = "volume"
    PRICE = "price"
    BREAKOUT = "breakout"


class ComparisonOp(str, enum.Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


# ── Param requirements per indicator ──────────────────────────────────────────

# Maps IndicatorType value → list of required param keys.
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "rsi": ["period"],
    "ema": ["period"],
    "sma": ["period"],
    "macd": ["fast", "slow"],
    "bbands": ["period"],
    "atr": ["period"],
    "vwap": [],
    "volume": ["period"],
    "price": [],
    "breakout": ["lookback"],
}


# ── IndicatorRef ──────────────────────────────────────────────────────────────


class IndicatorRef(BaseModel):
    """Reference to a technical indicator with its configuration."""

    type: IndicatorType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> IndicatorRef:
        required = _REQUIRED_PARAMS.get(self.type.value, [])
        missing = [p for p in required if p not in self.params]
        if missing:
            raise ValueError(
                f"IndicatorRef({self.type.value}) is missing required params: {missing}"
            )
        return self


# ── Condition ─────────────────────────────────────────────────────────────────


class Condition(BaseModel):
    """Atomic comparison: left op right.

    ``left`` and ``right`` are either an IndicatorRef (computed from bar data)
    or a plain numeric threshold.  ``lookback_bars`` controls how many bars
    back to search for crossover events (default 1 = only the previous bar).
    """

    left: Union[IndicatorRef, float]
    op: ComparisonOp
    right: Union[IndicatorRef, float]
    lookback_bars: int = Field(default=1, ge=1)


# ── RuleGroup (recursive) ─────────────────────────────────────────────────────


class RuleGroup(BaseModel):
    """Tree node combining conditions with AND or OR logic.

    ``conditions`` may contain any mix of leaf ``Condition`` and nested
    ``RuleGroup`` nodes, enabling arbitrary depth rule trees.
    """

    logic: Literal["AND", "OR"] = "AND"
    conditions: list[Union[Condition, RuleGroup]]

    @field_validator("conditions")
    @classmethod
    def _not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("conditions must not be empty")
        return v


# Resolve the forward reference to RuleGroup inside itself.
RuleGroup.model_rebuild()


# ── ExitRules ─────────────────────────────────────────────────────────────────


class ExitRules(BaseModel):
    """Exit criteria evaluated in priority order: SL > TP > trailing > signal."""

    take_profit_pct: float | None = Field(default=None, gt=0.0, le=100.0)
    stop_loss_pct: float = Field(..., gt=0.1, le=50.0)
    trailing_stop_pct: float | None = Field(default=None, gt=0.0, le=50.0)
    inverse_signal: RuleGroup | None = None


# ── PositionSizing ────────────────────────────────────────────────────────────


class PositionSizing(BaseModel):
    """Portfolio-level risk limits.

    ``max_position_pct * max_concurrent_positions`` must not exceed
    ``max_total_exposure_pct`` — the engine validates this so the risk_manager
    never faces an impossible constraint set.
    """

    max_position_pct: float = Field(..., gt=0.0, le=100.0)
    max_total_exposure_pct: float = Field(..., gt=0.0, le=100.0)
    max_concurrent_positions: int = Field(..., ge=1, le=50)

    @model_validator(mode="after")
    def _coherent(self) -> PositionSizing:
        product = self.max_position_pct * self.max_concurrent_positions
        if product > self.max_total_exposure_pct:
            raise ValueError(
                f"max_position_pct ({self.max_position_pct}) × "
                f"max_concurrent_positions ({self.max_concurrent_positions}) = "
                f"{product:.4f} exceeds max_total_exposure_pct ({self.max_total_exposure_pct})"
            )
        return self


# ── Signal value objects (produced by StrategyEngine) ────────────────────────


class EntrySignal(BaseModel):
    """Signal to open a position, returned by StrategyEngine.evaluate_entry()."""

    symbol: str
    reason: str
    bar_timestamp: datetime


class ExitSignal(BaseModel):
    """Signal to close or reduce a position, returned by StrategyEngine.evaluate_exit()."""

    symbol: str
    reason: str
    exit_type: Literal["take_profit", "stop_loss", "trailing_stop", "inverse_signal"]
    bar_timestamp: datetime


# ── Strategy ──────────────────────────────────────────────────────────────────

# Timeframes that are NOT appropriate for each horizon.
# Checked in Strategy._coherent_timeframe_horizon.
_INTRADAY_FORBIDDEN_TF = {Timeframe.D1}
_POSITION_FORBIDDEN_TF = {Timeframe.M1}


class Strategy(BaseModel):
    """Complete, validated trading strategy definition.

    Produced by the LLM parser and consumed by StrategyEngine.
    Every field is immutable after construction.
    """

    name: str
    universe: list[str]
    timeframe: Timeframe
    session: Session = Session.REGULAR
    horizon: Horizon
    eod_policy: EodPolicy
    entry_rules: RuleGroup
    exit_rules: ExitRules
    position_sizing: PositionSizing

    # ── universe validators ────────────────────────────────────────────────

    @field_validator("universe")
    @classmethod
    def _validate_universe(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("universe must contain at least one ticker")
        if len(v) > 50:
            raise ValueError(f"universe may contain at most 50 tickers, got {len(v)}")
        cleaned = []
        for ticker in v:
            t = ticker.strip().upper()
            if not t or " " in t:
                raise ValueError(f"invalid ticker: {ticker!r}")
            cleaned.append(t)
        return cleaned

    # ── cross-field validators ─────────────────────────────────────────────

    @model_validator(mode="after")
    def _coherent_timeframe_horizon(self) -> Strategy:
        if self.timeframe in _INTRADAY_FORBIDDEN_TF and self.horizon == Horizon.INTRADAY:
            raise ValueError(
                f"Timeframe {self.timeframe.value} is incompatible with horizon INTRADAY "
                f"(daily bars cannot feed an intraday strategy)"
            )
        if self.timeframe in _POSITION_FORBIDDEN_TF and self.horizon == Horizon.POSITION:
            raise ValueError(
                f"Timeframe {self.timeframe.value} is incompatible with horizon POSITION "
                f"(1-minute bars are too granular for a position/long-term strategy)"
            )
        return self

    @model_validator(mode="after")
    def _coherent_session_horizon(self) -> Strategy:
        if self.session == Session.EXTENDED and self.horizon != Horizon.INTRADAY:
            raise ValueError(
                f"Extended-hours session requires horizon=INTRADAY, got {self.horizon.value}"
            )
        return self

    @model_validator(mode="after")
    def _entry_rules_not_empty(self) -> Strategy:
        # entry_rules.conditions is already validated non-empty by RuleGroup itself,
        # but we assert it here as a belt-and-suspenders strategy-level check.
        if not self.entry_rules.conditions:
            raise ValueError("entry_rules must contain at least one condition")
        return self
