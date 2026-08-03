"""Aggregate Alpaca minute bars into completed strategy timeframes.

Alpaca's ``bars`` websocket channel always emits one-minute bars.  This
module keeps those transport bars separate from the historical strategy-bar
cache and emits a bar only when the configured bucket is complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.broker.schemas import BarData
from src.strategy.schema import Session, Timeframe

ET = ZoneInfo("America/New_York")

_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
}


@dataclass
class _PartialBar:
    bucket_start: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap_numerator: float
    vwap_volume: float
    last_source_timestamp: datetime

    @classmethod
    def from_bar(cls, bucket_start: datetime, bar: BarData) -> _PartialBar:
        vwap_volume = bar.volume if bar.vwap is not None else 0.0
        return cls(
            bucket_start=bucket_start,
            symbol=bar.symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap_numerator=(bar.vwap or 0.0) * vwap_volume,
            vwap_volume=vwap_volume,
            last_source_timestamp=bar.timestamp,
        )

    def update(self, bar: BarData) -> None:
        # Updated/duplicate bars must not inflate volume.  The live manager is
        # subscribed only to finalized minute bars, so exact duplicates are
        # ignored and out-of-order bars are rejected by the aggregator.
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        if bar.vwap is not None:
            self.vwap_numerator += bar.vwap * bar.volume
            self.vwap_volume += bar.volume
        self.last_source_timestamp = bar.timestamp

    def build(self) -> BarData:
        return BarData(
            symbol=self.symbol,
            timestamp=self.bucket_start,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=(
                self.vwap_numerator / self.vwap_volume
                if self.vwap_volume > 0
                else None
            ),
        )


class TimeframeAggregator:
    """Stateful per-symbol minute-bar aggregator.

    ``push`` returns zero or more completed bars.  Normally it returns one bar
    at the end of a bucket.  If minute data has gaps, the previous partial bar
    is emitted when a bar from the next bucket arrives.
    """

    def __init__(self) -> None:
        self._partials: dict[str, _PartialBar] = {}
        self._last_seen: dict[str, datetime] = {}

    def reset(self, symbols: set[str] | None = None) -> None:
        if symbols is None:
            self._partials.clear()
            self._last_seen.clear()
            return
        for symbol in symbols:
            self._partials.pop(symbol.upper(), None)
            self._last_seen.pop(symbol.upper(), None)

    def push(
        self,
        bar: BarData,
        timeframe: Timeframe,
        session: Session = Session.REGULAR,
        session_close: time | None = None,
    ) -> list[BarData]:
        if timeframe == Timeframe.D1:
            return []
        if timeframe == Timeframe.M1:
            last = self._last_seen.get(bar.symbol)
            if last is not None and bar.timestamp <= last:
                return []
            self._last_seen[bar.symbol] = bar.timestamp
            return [bar]

        minutes = _MINUTES[timeframe]
        symbol = bar.symbol.upper()
        last = self._last_seen.get(symbol)
        if last is not None and bar.timestamp <= last:
            return []
        self._last_seen[symbol] = bar.timestamp

        bucket_start = self._bucket_start(bar.timestamp, minutes, session)
        partial = self._partials.get(symbol)
        completed: list[BarData] = []

        if partial is not None and partial.bucket_start != bucket_start:
            completed.append(partial.build())
            partial = None

        if partial is None:
            partial = _PartialBar.from_bar(bucket_start, bar)
            self._partials[symbol] = partial
        else:
            partial.update(bar)

        # A minute bar timestamp denotes the start of that minute.  When its
        # end reaches the bucket boundary, the aggregate is complete now.
        bar_end = bar.timestamp + timedelta(minutes=1)
        closes_session = False
        if session_close is not None:
            bar_end_et = bar_end.astimezone(ET)
            session_close_dt = datetime.combine(
                bar_end_et.date(), session_close, tzinfo=ET
            )
            closes_session = bar_end_et >= session_close_dt
        if bar_end >= bucket_start + timedelta(minutes=minutes) or closes_session:
            completed.append(partial.build())
            self._partials.pop(symbol, None)

        return completed

    @staticmethod
    def _bucket_start(timestamp: datetime, minutes: int, session: Session) -> datetime:
        ts_et = timestamp.astimezone(ET)
        anchor_time = time(4, 0) if session == Session.EXTENDED else time(9, 30)
        anchor = datetime.combine(ts_et.date(), anchor_time, tzinfo=ET)
        elapsed = int((ts_et - anchor).total_seconds() // 60)
        bucket_index = elapsed // minutes
        return (anchor + timedelta(minutes=bucket_index * minutes)).astimezone(
            timestamp.tzinfo
        )
