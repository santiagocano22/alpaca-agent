"""Alpaca WebSocket stream manager.

Wraps TradingStream (trade updates, JSON/WebSocket) and StockDataStream
(market bars, msgpack/WebSocket) with:
- Independent asyncio.Task per stream so one crash never silences the other
- Exponential backoff with jitter on disconnect or unexpected exit
- Injectable sleeper and jitter for deterministic unit tests
- Idempotent stop(): safe to call multiple times or before start()

Verified against paper API 2026-05-10:
  TradeUpdate.timestamp  → UTC-aware (pydantic parses ISO-8601 Z as UTC)
  Bar.timestamp          → UTC-aware (msgpack.Timestamp.to_datetime() always UTC)
  TradingStream._run_forever()   → async, blocks until stop_ws() signals stop
  StockDataStream._run_forever() → async, blocks until stop_ws() signals stop
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from src.broker.schemas import (
    BarData,
    BarEvent,
    OrderSide as BrokerOrderSide,
    OrderStatusResult,
    TradeUpdateEvent,
)

TradeUpdateCallback = Callable[[TradeUpdateEvent], Awaitable[None]]
BarCallback = Callable[[BarEvent], Awaitable[None]]

_MAX_BACKOFF_SECONDS = 60.0


class AlpacaStreamManager:
    """Lifecycle manager for Alpaca's TradingStream and StockDataStream."""

    def __init__(
        self,
        trading_stream,               # alpaca.trading.stream.TradingStream
        data_stream,                  # alpaca.data.live.StockDataStream
        settings,                     # src.config.Settings
        max_reconnect_attempts: int = 10,
        base_backoff_seconds: float = 1.0,
        sleeper=None,
        jitter=None,
    ) -> None:
        self._ts = trading_stream
        self._ds = data_stream
        self._settings = settings
        self._max_reconnect_attempts = max_reconnect_attempts
        self._base_backoff_seconds = base_backoff_seconds
        self._sleeper = sleeper if sleeper is not None else asyncio.sleep
        self._jitter = jitter if jitter is not None else (lambda: random.uniform(0.75, 1.25))

        self._subscribed_symbols: set[str] = set()
        self._trade_callback: TradeUpdateCallback | None = None
        self._bar_callback: BarCallback | None = None
        self._trading_task: asyncio.Task | None = None
        self._data_task: asyncio.Task | None = None
        self._stopped = False

    # ── Public subscription API ───────────────────────────────────────────────

    def subscribe_trade_updates(self, callback: TradeUpdateCallback) -> None:
        """Register an async callback for trade lifecycle events."""
        self._trade_callback = callback
        self._ts.subscribe_trade_updates(self._handle_trade_update)

    def subscribe_bars(self, callback: BarCallback, *symbols: str) -> None:
        """Register an async callback for bar data and subscribe to symbols."""
        self._bar_callback = callback
        syms = [s.upper() for s in symbols]
        self._subscribed_symbols.update(syms)
        self._ds.subscribe_bars(self._handle_bar, *syms)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn both stream tasks. Call once; not restartable after stop()."""
        if self._stopped:
            raise RuntimeError("AlpacaStreamManager cannot be restarted after stop()")
        self._trading_task = asyncio.create_task(
            self._run_trading_stream(), name="trading_stream"
        )
        self._data_task = asyncio.create_task(
            self._run_data_stream(), name="data_stream"
        )

    async def stop(self) -> None:
        """Gracefully stop both streams and wait for tasks to finish.

        Idempotent: safe to call multiple times or before start().
        Signals stop via stop_ws()/close(), cancels tasks, and awaits
        termination with a 5-second timeout.
        """
        if self._stopped:
            return
        self._stopped = True

        for stream in (self._ts, self._ds):
            try:
                stream.stop_ws()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        await asyncio.sleep(0)  # yield so tasks can observe _stopped before cancel

        tasks = [t for t in (self._trading_task, self._data_task) if t is not None]
        for t in tasks:
            t.cancel()

        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=5.0)
            if pending:
                logger.error(
                    "Stream tasks did not stop within 5s: {}",
                    [t.get_name() for t in pending],
                )

    # ── Internal stream loops ─────────────────────────────────────────────────

    async def _run_trading_stream(self) -> None:
        reconnect = 0
        while not self._stopped:
            try:
                await self._ts._run_forever()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopped:
                    return
                logger.error("Trading stream error (reconnect {}): {}", reconnect, exc)
            else:
                if self._stopped:
                    return
                logger.warning(
                    "Trading stream ended unexpectedly (reconnect {}), reconnecting", reconnect
                )

            if reconnect >= self._max_reconnect_attempts:
                logger.error(
                    "Trading stream: max reconnect attempts ({}) reached",
                    self._max_reconnect_attempts,
                )
                return

            delay = min(
                self._base_backoff_seconds * (2 ** reconnect),
                _MAX_BACKOFF_SECONDS,
            ) * self._jitter()
            await self._sleeper(delay)
            reconnect += 1

    async def _run_data_stream(self) -> None:
        reconnect = 0
        while not self._stopped:
            try:
                await self._ds._run_forever()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopped:
                    return
                logger.error("Data stream error (reconnect {}): {}", reconnect, exc)
            else:
                if self._stopped:
                    return
                logger.warning(
                    "Data stream ended unexpectedly (reconnect {}), reconnecting", reconnect
                )

            if reconnect >= self._max_reconnect_attempts:
                logger.error(
                    "Data stream: max reconnect attempts ({}) reached",
                    self._max_reconnect_attempts,
                )
                return

            delay = min(
                self._base_backoff_seconds * (2 ** reconnect),
                _MAX_BACKOFF_SECONDS,
            ) * self._jitter()
            await self._sleeper(delay)
            reconnect += 1

            # Resubscribe before next _run_forever so the stream receives events
            if self._subscribed_symbols and self._bar_callback is not None:
                self._ds.subscribe_bars(self._handle_bar, *self._subscribed_symbols)

    # ── SDK event handlers ────────────────────────────────────────────────────

    async def _handle_trade_update(self, tu: Any) -> None:
        if self._trade_callback is None:
            return
        await self._trade_callback(self._map_trade_update(tu))

    async def _handle_bar(self, bar: Any) -> None:
        if self._bar_callback is None:
            return
        await self._bar_callback(self._map_bar(bar))

    # ── SDK → schema mappers ──────────────────────────────────────────────────

    def _map_trade_update(self, tu: Any) -> TradeUpdateEvent:
        ts = tu.timestamp
        if ts is not None and ts.tzinfo is None:
            raise RuntimeError(
                "TradeUpdate.timestamp is naive — SDK changed format. "
                "Verified UTC-aware against paper API 2026-05-10."
            )
        return TradeUpdateEvent(
            event=tu.event,
            order=self._map_order_status(tu.order),
            timestamp=ts,
        )

    def _map_bar(self, bar: Any) -> BarEvent:
        ts = bar.timestamp
        if ts is not None and ts.tzinfo is None:
            raise RuntimeError(
                "Bar.timestamp is naive — SDK changed format. "
                "Verified UTC-aware via msgpack.Timestamp.to_datetime() 2026-05-10."
            )
        return BarEvent(
            bar=BarData(
                symbol=bar.symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if getattr(bar, "vwap", None) is not None else None,
            ),
            received_at=datetime.now(UTC),
        )

    @staticmethod
    def _map_order_status(raw: Any) -> OrderStatusResult:
        def _tz(dt):
            if dt is not None and dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt

        raw_side = raw.side
        side_val = raw_side.value if hasattr(raw_side, "value") else str(raw_side)

        return OrderStatusResult(
            alpaca_order_id=str(raw.id),
            client_order_id=str(getattr(raw, "client_order_id", "")),
            status=raw.status.value if hasattr(raw.status, "value") else str(raw.status),
            submitted_at=_tz(raw.submitted_at) or datetime.now(UTC),
            filled_at=_tz(getattr(raw, "filled_at", None)),
            canceled_at=_tz(getattr(raw, "canceled_at", None)),
            filled_qty=float(getattr(raw, "filled_qty", 0) or 0),
            filled_avg_price=(
                float(raw.filled_avg_price)
                if getattr(raw, "filled_avg_price", None) is not None
                else None
            ),
            side=BrokerOrderSide(side_val),
            symbol=str(raw.symbol),
            qty=float(raw.qty),
        )
