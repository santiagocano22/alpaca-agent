from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from src.broker.schemas import BarData
from src.strategy.schema import Session, Timeframe
from src.strategy.timeframe import TimeframeAggregator


def _bar(minute: int, *, close: float, volume: float = 100.0) -> BarData:
    ts = datetime(2026, 1, 5, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    return BarData(
        symbol="AAPL",
        timestamp=ts,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        vwap=close,
    )


def test_fifteen_minute_bar_emits_only_when_complete() -> None:
    aggregator = TimeframeAggregator()
    emitted = []
    for minute in range(14):
        emitted.extend(
            aggregator.push(_bar(minute, close=100 + minute), Timeframe.M15)
        )
    assert emitted == []

    emitted = aggregator.push(_bar(14, close=114), Timeframe.M15)
    assert len(emitted) == 1
    result = emitted[0]
    assert result.timestamp == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    assert result.open == 99.5
    assert result.close == 114
    assert result.high == 115
    assert result.low == 99
    assert result.volume == 1500


def test_hour_bucket_is_anchored_to_regular_open() -> None:
    aggregator = TimeframeAggregator()
    emitted = []
    for minute in range(60):
        emitted.extend(
            aggregator.push(
                _bar(minute, close=100 + minute),
                Timeframe.H1,
                Session.REGULAR,
            )
        )
    assert len(emitted) == 1
    assert emitted[0].timestamp == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def test_duplicate_and_out_of_order_minutes_are_ignored() -> None:
    aggregator = TimeframeAggregator()
    first = _bar(0, close=100)
    assert aggregator.push(first, Timeframe.M1) == [first]
    assert aggregator.push(first, Timeframe.M1) == []
    assert aggregator.push(_bar(-1, close=99), Timeframe.M1) == []


def test_reset_discards_partial_bucket() -> None:
    aggregator = TimeframeAggregator()
    aggregator.push(_bar(0, close=100), Timeframe.M5)
    aggregator.reset()
    emitted = []
    for minute in range(1, 5):
        emitted.extend(aggregator.push(_bar(minute, close=100 + minute), Timeframe.M5))
    assert len(emitted) == 1
    assert emitted[0].open == 100.5


def test_hour_partial_is_finalized_at_early_session_close() -> None:
    aggregator = TimeframeAggregator()
    # 09:30 ET start, early close supplied as 10:00 ET for a compact test.
    emitted = []
    for minute in range(30):
        emitted.extend(
            aggregator.push(
                _bar(minute, close=100 + minute),
                Timeframe.H1,
                Session.REGULAR,
                session_close=time(10, 0),
            )
        )
    assert len(emitted) == 1
    assert emitted[0].volume == 3000
