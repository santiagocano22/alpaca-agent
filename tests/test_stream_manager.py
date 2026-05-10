"""Tests for src/broker/stream_manager.py.

All tests use an injectable sleeper (AsyncMock or explicit async fn) so no
real asyncio.sleep calls are made; wall time stays < 2 s.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.broker.schemas import BarEvent, TradeUpdateEvent
from src.broker.stream_manager import AlpacaStreamManager


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_streams():
    ts = MagicMock()
    ts._run_forever = AsyncMock()
    ts.stop_ws = MagicMock()
    ts.close = MagicMock()
    ts.subscribe_trade_updates = MagicMock()

    ds = MagicMock()
    ds._run_forever = AsyncMock()
    ds.stop_ws = MagicMock()
    ds.close = MagicMock()
    ds.subscribe_bars = MagicMock()

    return ts, ds


def _make_manager(
    *,
    sleeper=None,
    jitter=None,
    max_reconnect_attempts=10,
    base_backoff_seconds=1.0,
):
    ts, ds = _make_streams()
    sleeper = sleeper if sleeper is not None else AsyncMock()
    jitter = jitter if jitter is not None else (lambda: 1.0)
    mgr = AlpacaStreamManager(
        trading_stream=ts,
        data_stream=ds,
        settings=MagicMock(),
        max_reconnect_attempts=max_reconnect_attempts,
        base_backoff_seconds=base_backoff_seconds,
        sleeper=sleeper,
        jitter=jitter,
    )
    return mgr, ts, ds, sleeper


def _make_raw_order():
    o = MagicMock()
    o.id = "order-abc"
    o.client_order_id = "client-abc"
    o.status = MagicMock(value="new")
    o.submitted_at = datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)
    o.filled_at = None
    o.canceled_at = None
    o.filled_qty = "0"
    o.filled_avg_price = None
    o.side = MagicMock(value="buy")
    o.symbol = "AAPL"
    o.qty = "10"
    return o


def _make_raw_trade_update(event: str = "fill"):
    tu = MagicMock()
    tu.event = event
    tu.timestamp = datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)
    tu.order = _make_raw_order()
    return tu


def _make_raw_bar():
    b = MagicMock()
    b.symbol = "AAPL"
    b.timestamp = datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)
    b.open = 150.0
    b.high = 152.0
    b.low = 149.0
    b.close = 151.0
    b.volume = 10000.0
    b.vwap = None
    return b


async def _blocking_run():
    """Simulates a running stream: blocks until cancelled."""
    await asyncio.Event().wait()


# ── 1. Construction ───────────────────────────────────────────────────────────


def test_default_sleeper_is_asyncio_sleep():
    ts, ds = _make_streams()
    mgr = AlpacaStreamManager(ts, ds, MagicMock())
    assert mgr._sleeper is asyncio.sleep


def test_injectable_sleeper_and_jitter_stored():
    ts, ds = _make_streams()
    fake_sleep = AsyncMock()
    jitter_fn = lambda: 2.0
    mgr = AlpacaStreamManager(ts, ds, MagicMock(), sleeper=fake_sleep, jitter=jitter_fn)
    assert mgr._sleeper is fake_sleep
    assert mgr._jitter is jitter_fn


# ── 2. Subscriptions ──────────────────────────────────────────────────────────


def test_subscribe_trade_updates_delegates_to_sdk():
    mgr, ts, ds, _ = _make_manager()
    mgr.subscribe_trade_updates(AsyncMock())
    ts.subscribe_trade_updates.assert_called_once_with(mgr._handle_trade_update)


def test_subscribe_bars_tracks_symbols_and_delegates():
    mgr, ts, ds, _ = _make_manager()
    mgr.subscribe_bars(AsyncMock(), "aapl", "GOOG")
    assert "AAPL" in mgr._subscribed_symbols
    assert "GOOG" in mgr._subscribed_symbols
    ds.subscribe_bars.assert_called_once()


# ── 3. Mappers ────────────────────────────────────────────────────────────────


def test_map_trade_update_returns_event_with_tz_aware_timestamp():
    mgr, _, _, _ = _make_manager()
    result = mgr._map_trade_update(_make_raw_trade_update("fill"))
    assert isinstance(result, TradeUpdateEvent)
    assert result.event == "fill"
    assert result.timestamp.tzinfo is not None


def test_map_trade_update_naive_timestamp_raises():
    mgr, _, _, _ = _make_manager()
    tu = _make_raw_trade_update()
    tu.timestamp = datetime(2026, 5, 10, 12, 0, 0)  # naive
    with pytest.raises(RuntimeError, match="naive"):
        mgr._map_trade_update(tu)


def test_map_bar_returns_event_with_received_at_tz_aware():
    mgr, _, _, _ = _make_manager()
    result = mgr._map_bar(_make_raw_bar())
    assert isinstance(result, BarEvent)
    assert result.bar.symbol == "AAPL"
    assert result.received_at.tzinfo is not None


def test_map_bar_naive_timestamp_raises():
    mgr, _, _, _ = _make_manager()
    bar = _make_raw_bar()
    bar.timestamp = datetime(2026, 5, 10, 12, 0, 0)  # naive
    with pytest.raises(RuntimeError, match="naive"):
        mgr._map_bar(bar)


# ── 4. Lifecycle ──────────────────────────────────────────────────────────────


async def test_start_creates_two_running_tasks():
    mgr, ts, ds, _ = _make_manager()
    ts._run_forever = _blocking_run
    ds._run_forever = _blocking_run

    await mgr.start()
    assert mgr._trading_task is not None
    assert mgr._data_task is not None
    assert not mgr._trading_task.done()
    assert not mgr._data_task.done()
    await mgr.stop()


async def test_start_after_stop_raises():
    mgr, ts, ds, _ = _make_manager()
    ts._run_forever = _blocking_run
    ds._run_forever = _blocking_run

    await mgr.start()
    await mgr.stop()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await mgr.start()


async def test_stop_without_start_is_noop():
    mgr, _, _, _ = _make_manager()
    await mgr.stop()
    assert mgr._stopped is True


async def test_stop_idempotent():
    mgr, ts, ds, _ = _make_manager()
    ts._run_forever = _blocking_run
    ds._run_forever = _blocking_run

    await mgr.start()
    await mgr.stop()
    await mgr.stop()  # second call must not raise or double-signal
    assert ts.stop_ws.call_count == 1


async def test_stop_calls_stop_ws_on_both_streams():
    mgr, ts, ds, _ = _make_manager()
    ts._run_forever = _blocking_run
    ds._run_forever = _blocking_run

    await mgr.start()
    await mgr.stop()
    ts.stop_ws.assert_called_once()
    ds.stop_ws.assert_called_once()


# ── 5. Backoff and reconnect ──────────────────────────────────────────────────


async def test_trading_stream_backoff_delays_are_exponential():
    """_run_trading_stream sleeps with exponential backoff between reconnects."""
    sleep_calls: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)

    async def always_fails():
        raise ConnectionError("disconnected")

    ts, ds = _make_streams()
    ts._run_forever = always_fails

    mgr = AlpacaStreamManager(
        trading_stream=ts,
        data_stream=ds,
        settings=MagicMock(),
        max_reconnect_attempts=3,
        base_backoff_seconds=1.0,
        sleeper=fake_sleep,
        jitter=lambda: 1.0,
    )
    # Run the internal coroutine directly — no asyncio.Task overhead
    await mgr._run_trading_stream()

    # max_reconnect_attempts=3 → 3 sleeps before giving up (reconnect 0,1,2)
    assert sleep_calls == [1.0, 2.0, 4.0]


async def test_data_stream_resubscribes_symbols_after_reconnect():
    """After a disconnect, _run_data_stream re-registers bar subscriptions."""
    run_count = 0

    async def fail_then_succeed():
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            raise ConnectionError("initial disconnect")
        # Second call returns normally; max_reconnect_attempts=1 exits after this

    ts, ds = _make_streams()
    ds._run_forever = fail_then_succeed

    mgr = AlpacaStreamManager(
        trading_stream=ts,
        data_stream=ds,
        settings=MagicMock(),
        max_reconnect_attempts=1,
        base_backoff_seconds=0.0,
        sleeper=AsyncMock(),
        jitter=lambda: 1.0,
    )
    mgr.subscribe_bars(AsyncMock(), "AAPL")
    ds.subscribe_bars.reset_mock()  # ignore the initial subscribe call

    await mgr._run_data_stream()

    # Must have resubscribed once after the disconnect, before the second run
    ds.subscribe_bars.assert_called_once()
    call_args = ds.subscribe_bars.call_args
    # First positional arg is the handler; remaining are symbols
    assert "AAPL" in call_args.args or "AAPL" in str(call_args)
