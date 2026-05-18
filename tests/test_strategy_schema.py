"""Tests for src/strategy/schema.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.schema import (
    ComparisonOp,
    Condition,
    EodPolicy,
    ExitRules,
    Horizon,
    IndicatorRef,
    IndicatorType,
    PositionSizing,
    RuleGroup,
    Session,
    Strategy,
    Timeframe,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _simple_rsi_ref(period: int = 14) -> IndicatorRef:
    return IndicatorRef(type=IndicatorType.RSI, params={"period": period})


def _simple_condition(
    left=None,
    op: ComparisonOp = ComparisonOp.LT,
    right: float = 30.0,
) -> Condition:
    if left is None:
        left = _simple_rsi_ref()
    return Condition(left=left, op=op, right=right)


def _simple_rule_group(*conditions) -> RuleGroup:
    return RuleGroup(logic="AND", conditions=list(conditions) or [_simple_condition()])


def _simple_exit_rules(stop_loss_pct: float = 5.0, take_profit_pct: float | None = 10.0) -> ExitRules:
    return ExitRules(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)


def _simple_position_sizing(
    max_position_pct: float = 25.0,
    max_total_exposure_pct: float = 100.0,
    max_concurrent_positions: int = 4,
) -> PositionSizing:
    return PositionSizing(
        max_position_pct=max_position_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        max_concurrent_positions=max_concurrent_positions,
    )


def _full_strategy(**overrides) -> Strategy:
    defaults = dict(
        name="test",
        universe=["AAPL", "GOOG"],
        timeframe=Timeframe.M15,
        session=Session.REGULAR,
        horizon=Horizon.INTRADAY,
        eod_policy=EodPolicy.CLOSE_ALL,
        entry_rules=_simple_rule_group(),
        exit_rules=_simple_exit_rules(),
        position_sizing=_simple_position_sizing(),
    )
    defaults.update(overrides)
    return Strategy(**defaults)


# ── IndicatorRef params validation ────────────────────────────────────────────


def test_indicator_ref_rsi_missing_period_raises():
    with pytest.raises(ValidationError, match="period"):
        IndicatorRef(type=IndicatorType.RSI, params={})


def test_indicator_ref_macd_missing_fast_raises():
    with pytest.raises(ValidationError, match="fast"):
        IndicatorRef(type=IndicatorType.MACD, params={"slow": 26})


def test_indicator_ref_macd_missing_slow_raises():
    with pytest.raises(ValidationError, match="slow"):
        IndicatorRef(type=IndicatorType.MACD, params={"fast": 12})


def test_indicator_ref_vwap_no_params_required():
    ref = IndicatorRef(type=IndicatorType.VWAP, params={})
    assert ref.type == IndicatorType.VWAP


def test_indicator_ref_breakout_requires_lookback():
    with pytest.raises(ValidationError, match="lookback"):
        IndicatorRef(type=IndicatorType.BREAKOUT, params={})


def test_indicator_ref_sma_valid():
    ref = IndicatorRef(type=IndicatorType.SMA, params={"period": 20})
    assert ref.params["period"] == 20


# ── RuleGroup ─────────────────────────────────────────────────────────────────


def test_rule_group_empty_conditions_raises():
    with pytest.raises(ValidationError, match="not be empty"):
        RuleGroup(logic="AND", conditions=[])


def test_rule_group_recursive_and_of_or_of_and():
    """Verify that deeply nested AND/OR/AND trees are accepted."""
    inner = RuleGroup(
        logic="AND",
        conditions=[_simple_condition(), _simple_condition(right=50.0)],
    )
    middle = RuleGroup(logic="OR", conditions=[inner, _simple_condition(right=70.0)])
    outer = RuleGroup(logic="AND", conditions=[middle, _simple_condition(right=20.0)])
    assert outer.logic == "AND"
    assert len(outer.conditions) == 2
    assert isinstance(outer.conditions[0], RuleGroup)
    assert outer.conditions[0].logic == "OR"


# ── PositionSizing coherence ──────────────────────────────────────────────────


def test_position_sizing_product_exceeds_total_exposure_raises():
    """25% × 5 positions = 125% > 100% → error."""
    with pytest.raises(ValidationError, match="exceeds"):
        PositionSizing(
            max_position_pct=25.0,
            max_total_exposure_pct=100.0,
            max_concurrent_positions=5,
        )


def test_position_sizing_product_equals_total_allowed():
    """25% × 4 = 100% ≤ 100% → OK."""
    ps = PositionSizing(
        max_position_pct=25.0,
        max_total_exposure_pct=100.0,
        max_concurrent_positions=4,
    )
    assert ps.max_concurrent_positions == 4


def test_position_sizing_max_positions_above_50_raises():
    with pytest.raises(ValidationError):
        PositionSizing(
            max_position_pct=1.0,
            max_total_exposure_pct=100.0,
            max_concurrent_positions=51,
        )


# ── Strategy cross-field validators ──────────────────────────────────────────


def test_strategy_d1_intraday_raises():
    with pytest.raises(ValidationError, match="[Ii]ntraday|incompatible"):
        _full_strategy(timeframe=Timeframe.D1, horizon=Horizon.INTRADAY)


def test_strategy_m1_position_raises():
    with pytest.raises(ValidationError, match="[Pp]osition|incompatible"):
        _full_strategy(timeframe=Timeframe.M1, horizon=Horizon.POSITION)


def test_strategy_d1_swing_allowed():
    """D1 + SWING is a valid combination."""
    s = _full_strategy(
        timeframe=Timeframe.D1,
        horizon=Horizon.SWING,
        eod_policy=EodPolicy.HOLD,
    )
    assert s.timeframe == Timeframe.D1


def test_strategy_extended_session_swing_raises():
    with pytest.raises(ValidationError, match="[Ee]xtended|[Ii]ntraday"):
        _full_strategy(session=Session.EXTENDED, horizon=Horizon.SWING)


def test_strategy_extended_session_intraday_allowed():
    s = _full_strategy(session=Session.EXTENDED, horizon=Horizon.INTRADAY)
    assert s.session == Session.EXTENDED


def test_strategy_universe_empty_raises():
    with pytest.raises(ValidationError, match="at least one"):
        _full_strategy(universe=[])


def test_strategy_universe_too_many_raises():
    with pytest.raises(ValidationError, match="at most 50"):
        _full_strategy(universe=[f"TK{i}" for i in range(51)])


def test_strategy_universe_uppercased():
    s = _full_strategy(universe=["aapl", "goog"])
    assert s.universe == ["AAPL", "GOOG"]


# ── Round-trip ────────────────────────────────────────────────────────────────


def test_strategy_model_dump_round_trip():
    """model_dump() → Strategy(**...) must produce an equal object."""
    s1 = _full_strategy()
    dump = s1.model_dump()
    s2 = Strategy(**dump)
    assert s1.model_dump() == s2.model_dump()
