from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpaca.data.enums import Adjustment

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    ConditionStat,
    format_backtest_report,
    run_backtest,
)
from src.broker.schemas import BarData
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
from src.utils.market_hours import ET


def _strategy() -> Strategy:
    return Strategy(
        name="daily-test",
        universe=["SPY"],
        timeframe=Timeframe.D1,
        session=Session.REGULAR,
        horizon=Horizon.SWING,
        eod_policy=EodPolicy.HOLD,
        entry_rules=RuleGroup(
            logic="AND",
            conditions=[
                Condition(
                    left=IndicatorRef(type=IndicatorType.PRICE),
                    op=ComparisonOp.GT,
                    right=10.0,
                )
            ],
        ),
        exit_rules=ExitRules(stop_loss_pct=5.0),
        position_sizing=PositionSizing(
            max_position_pct=10,
            max_total_exposure_pct=10,
            max_concurrent_positions=1,
        ),
    )


def _bar(day: date, open_price: float, close_price: float) -> BarData:
    return BarData(
        symbol="SPY",
        timestamp=datetime.combine(day, time.min, tzinfo=ET).astimezone(UTC),
        open=open_price,
        high=max(open_price, close_price),
        low=min(open_price, close_price),
        close=close_price,
        volume=1_000,
    )


@pytest.mark.asyncio
async def test_backtest_executes_signals_at_next_open_without_order_calls():
    start = date(2025, 1, 6)
    bars = [
        _bar(start - timedelta(days=2), 9, 9),
        _bar(start - timedelta(days=1), 9, 9),
        _bar(start, 10, 11),
        _bar(start + timedelta(days=1), 12, 12),
        _bar(start + timedelta(days=2), 12, 8),
        _bar(start + timedelta(days=3), 8, 8),
    ]
    alpaca = MagicMock()
    alpaca.get_bars = AsyncMock(return_value=bars)
    alpaca.submit_order = AsyncMock()

    result = await run_backtest(
        _strategy(),
        alpaca,
        BacktestConfig(
            start_date=start,
            end_date=start + timedelta(days=3),
            initial_cash=10_000,
            slippage_bps=0,
            warmup_calendar_days=10,
        ),
    )

    assert [(fill.side, fill.trade_date, fill.price) for fill in result.fills] == [
        ("BUY", start + timedelta(days=1), 12.0),
        ("SELL", start + timedelta(days=3), 8.0),
    ]
    assert result.entries == 1
    assert result.exits == 1
    assert result.realized_pnl == pytest.approx(-333.333333)
    assert result.final_equity == pytest.approx(9_666.666667)
    alpaca.submit_order.assert_not_awaited()
    assert alpaca.get_bars.call_args.kwargs["adjustment"] == Adjustment.ALL


@pytest.mark.asyncio
async def test_backtest_rejects_non_daily_strategy():
    strategy = _strategy().model_copy(update={"timeframe": Timeframe.M15})

    with pytest.raises(RuntimeError, match="1D"):
        await run_backtest(
            strategy,
            MagicMock(),
            BacktestConfig(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)),
        )


def test_backtest_report_contains_operational_diagnostics():
    result = BacktestResult(
        strategy_name="test",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        initial_equity=100_000,
        final_equity=101_000,
        total_return_pct=1,
        max_drawdown_pct=-2,
        benchmark_return_pct=0.5,
        entry_signals=1,
        fills=(),
        open_positions=(),
        condition_stats=(ConditionStat("RSI crosses 40", 1, 20),),
        rejection_counts={"máximo de posiciones": 1},
        data_errors={},
        trading_days=20,
    )

    report = format_backtest_report(result)

    assert "Retorno: +1.00%" in report
    assert "Drawdown máximo" in report
    assert "1/20 (5.0%)" in report
    assert "Señales bloqueadas" in report
    assert len(report) <= 4000
