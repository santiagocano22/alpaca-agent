"""Tests for src/broker/alpaca_client.py.

Strategy: all SDK clients are MagicMocks injected at construction time.
No real network calls, no asyncio.sleep delays (retry sleeps are patched
where needed).  Tests stay deterministic and run in well under 2 s.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from alpaca.data.enums import Adjustment

from src.broker.alpaca_client import AlpacaClient, _classify_422, _map_calendar_day
from src.broker.exceptions import (
    AlpacaConnectionError,
    AlpacaDuplicateOrderError,
    AlpacaInsufficientFundsError,
    AlpacaOrderRejectedError,
    AlpacaRateLimitError,
    AlpacaServerError,
    AlpacaSymbolNotFoundError,
)
from src.broker.rate_limiter import TokenBucketLimiter
from src.broker.schemas import (
    AccountSnapshot,
    BarData,
    ClockSnapshot,
    MarketCalendarDay,
    OrderIntent,
    OrderSide,
    OrderStatusResult,
    OrderSubmitResult,
    OrderType,
    PositionSnapshot,
    TimeInForce,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_settings(dry_run: bool = False) -> MagicMock:
    s = MagicMock()
    s.dry_run = dry_run
    return s


def _make_rate_limiter() -> MagicMock:
    rl = MagicMock(spec=TokenBucketLimiter)
    rl.acquire = AsyncMock()
    rl.backoff_on_429 = AsyncMock()
    return rl


def _make_client(
    *,
    tc: MagicMock | None = None,
    dc: MagicMock | None = None,
    dry_run: bool = False,
    rl: MagicMock | None = None,
) -> AlpacaClient:
    return AlpacaClient(
        trading_client=tc or MagicMock(),
        data_client=dc or MagicMock(),
        settings=_make_settings(dry_run=dry_run),
        rate_limiter=rl or _make_rate_limiter(),
    )


def _make_intent(
    symbol: str = "QQQ",
    side: OrderSide = OrderSide.BUY,
    qty: float = 1.0,
    order_type: OrderType = OrderType.MARKET,
    client_order_id: str = "strat1-QQQ-123-abcd1234",
) -> OrderIntent:
    kwargs: dict = dict(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )
    if order_type == OrderType.LIMIT:
        kwargs["limit_price"] = 100.0
    return OrderIntent(**kwargs)


def _raw_order(
    alpaca_id: str = "alpaca-uuid-1",
    client_id: str = "strat1-QQQ-123-abcd1234",
    status: str = "new",
    side: str = "buy",
    symbol: str = "QQQ",
    qty: float = 1.0,
    submitted_at: datetime | None = None,
) -> MagicMock:
    o = MagicMock()
    o.id = alpaca_id
    o.client_order_id = client_id
    o.status = MagicMock()
    o.status.value = status
    o.side = MagicMock()
    o.side.value = side
    o.symbol = symbol
    o.qty = qty
    o.submitted_at = submitted_at or datetime(2026, 5, 9, 14, 0, 0, tzinfo=UTC)
    o.filled_at = None
    o.canceled_at = None
    o.filled_qty = 0.0
    o.filled_avg_price = None
    return o


# ── 422 classification ────────────────────────────────────────────────────────


class TestClassify422:
    def test_insufficient_buying_power(self) -> None:
        assert _classify_422("insufficient buying power") is AlpacaInsufficientFundsError

    def test_insufficient_funds(self) -> None:
        assert _classify_422("Insufficient funds for order") is AlpacaInsufficientFundsError

    def test_duplicate_client_order_id(self) -> None:
        assert _classify_422("duplicate client_order_id") is AlpacaDuplicateOrderError

    def test_client_order_id_exists(self) -> None:
        assert _classify_422("client_order_id already exists") is AlpacaDuplicateOrderError

    def test_order_already_exists(self) -> None:
        assert _classify_422("order already exists") is AlpacaDuplicateOrderError

    def test_unknown_falls_back_to_rejected(self) -> None:
        assert _classify_422("some novel rejection reason") is AlpacaOrderRejectedError


# ── _map_calendar_day ─────────────────────────────────────────────────────────


class TestMapCalendarDay:
    def test_converts_naive_et_to_utc(self) -> None:
        sdk = MagicMock()
        sdk.date = date(2026, 5, 9)
        sdk.open = datetime(2026, 5, 9, 9, 30)   # naive ET
        sdk.close = datetime(2026, 5, 9, 16, 0)  # naive ET

        day = _map_calendar_day(sdk)

        assert isinstance(day, MarketCalendarDay)
        # ET offset on 2026-05-09 is UTC-4 (EDT)
        assert day.session_open_utc == datetime(2026, 5, 9, 13, 30, tzinfo=UTC)
        assert day.session_close_utc == datetime(2026, 5, 9, 20, 0, tzinfo=UTC)

    def test_et_properties_round_trip(self) -> None:
        sdk = MagicMock()
        sdk.date = date(2026, 5, 9)
        sdk.open = datetime(2026, 5, 9, 9, 30)
        sdk.close = datetime(2026, 5, 9, 16, 0)

        day = _map_calendar_day(sdk)

        from datetime import time
        assert day.open_et == time(9, 30)
        assert day.close_et == time(16, 0)

    def test_raises_if_open_tz_aware(self) -> None:
        sdk = MagicMock()
        sdk.open = datetime(2026, 5, 9, 9, 30, tzinfo=UTC)
        sdk.close = datetime(2026, 5, 9, 16, 0)

        with pytest.raises(RuntimeError, match="Calendar.open is now tz-aware"):
            _map_calendar_day(sdk)

    def test_raises_if_close_tz_aware(self) -> None:
        sdk = MagicMock()
        sdk.open = datetime(2026, 5, 9, 9, 30)
        sdk.close = datetime(2026, 5, 9, 16, 0, tzinfo=UTC)

        with pytest.raises(RuntimeError, match="Calendar.close is now tz-aware"):
            _map_calendar_day(sdk)


# ── get_account ───────────────────────────────────────────────────────────────


class TestGetAccount:
    async def test_maps_fields_correctly(self) -> None:
        tc = MagicMock()
        tc.get_account.return_value = MagicMock(
            buying_power="50000.00",
            cash="20000.00",
            portfolio_value="80000.00",
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
        )
        client = _make_client(tc=tc)
        snap = await client.get_account()

        assert isinstance(snap, AccountSnapshot)
        assert snap.buying_power == 50000.0
        assert snap.portfolio_value == 80000.0
        assert snap.pattern_day_trader is False


# ── get_positions ─────────────────────────────────────────────────────────────


class TestGetPositions:
    async def test_returns_list_of_position_snapshots(self) -> None:
        raw = MagicMock()
        raw.symbol = "QQQ"
        raw.qty = "5.0"
        raw.side = MagicMock()
        raw.side.value = "long"
        raw.market_value = "1500.00"
        raw.avg_entry_price = "290.00"
        raw.unrealized_pl = "50.00"
        raw.unrealized_plpc = "0.034"
        raw.current_price = "300.00"

        tc = MagicMock()
        tc.get_all_positions.return_value = [raw]

        client = _make_client(tc=tc)
        positions = await client.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert isinstance(pos, PositionSnapshot)
        assert pos.symbol == "QQQ"
        assert pos.qty == 5.0
        assert pos.side == "long"

    async def test_get_position_returns_none_when_flat(self) -> None:
        err = Exception("Not Found")
        err.status_code = 404

        tc = MagicMock()
        tc.get_open_position.side_effect = err

        client = _make_client(tc=tc)
        result = await client.get_position("QQQ")

        assert result is None


# ── submit_order ──────────────────────────────────────────────────────────────


class TestSubmitOrder:
    async def test_dry_run_returns_simulated_result(self) -> None:
        client = _make_client(dry_run=True)
        intent = _make_intent()
        result = await client.submit_order(intent)

        assert result.simulated is True
        assert result.alpaca_order_id.startswith("dry-")
        assert result.status == "simulated"

    async def test_dry_run_does_not_call_api(self) -> None:
        tc = MagicMock()
        client = _make_client(tc=tc, dry_run=True)
        await client.submit_order(_make_intent())
        tc.submit_order.assert_not_called()

    async def test_successful_submit_returns_result(self) -> None:
        tc = MagicMock()
        tc.submit_order.return_value = _raw_order(status="new")

        rl = _make_rate_limiter()
        client = _make_client(tc=tc, rl=rl)

        with patch("src.broker.alpaca_client.AlpacaClient._build_order_request"):
            result = await client.submit_order(_make_intent())

        assert isinstance(result, OrderSubmitResult)
        assert result.simulated is False
        assert result.alpaca_order_id == "alpaca-uuid-1"

    async def test_duplicate_order_fetches_existing(self) -> None:
        existing = _raw_order(alpaca_id="existing-uuid")
        dup_exc = Exception("duplicate client_order_id")
        dup_exc.status_code = 422

        tc = MagicMock()
        tc.submit_order.side_effect = dup_exc
        tc.get_order_by_client_id.return_value = existing

        rl = _make_rate_limiter()
        client = _make_client(tc=tc, rl=rl)

        with patch("src.broker.alpaca_client.AlpacaClient._build_order_request"):
            result = await client.submit_order(_make_intent())

        assert result.alpaca_order_id == "existing-uuid"
        tc.get_order_by_client_id.assert_called_once()

    async def test_insufficient_funds_raises(self) -> None:
        exc = Exception("insufficient buying power")
        exc.status_code = 422

        tc = MagicMock()
        tc.submit_order.side_effect = exc

        rl = _make_rate_limiter()
        client = _make_client(tc=tc, rl=rl)

        with patch("src.broker.alpaca_client.AlpacaClient._build_order_request"):
            with pytest.raises(AlpacaInsufficientFundsError):
                await client.submit_order(_make_intent())

    async def test_rate_limiter_acquire_called_before_submit(self) -> None:
        call_order: list[str] = []

        tc = MagicMock()
        tc.submit_order.return_value = _raw_order()
        tc.submit_order.side_effect = lambda *a, **kw: call_order.append("submit") or _raw_order()

        rl = _make_rate_limiter()
        original_acquire = rl.acquire

        async def _tracked_acquire():
            call_order.append("acquire")
            await original_acquire()

        rl.acquire = _tracked_acquire

        client = _make_client(tc=tc, rl=rl)
        with patch("src.broker.alpaca_client.AlpacaClient._build_order_request"):
            await client.submit_order(_make_intent())

        assert call_order.index("acquire") < call_order.index("submit")


# ── get_order ─────────────────────────────────────────────────────────────────


class TestGetOrder:
    async def test_returns_order_status_result(self) -> None:
        tc = MagicMock()
        tc.get_order_by_id.return_value = _raw_order(status="filled", side="buy")

        client = _make_client(tc=tc)
        result = await client.get_order("alpaca-uuid-1")

        assert isinstance(result, OrderStatusResult)
        assert result.status == "filled"
        assert result.side == OrderSide.BUY

    async def test_returns_none_when_not_found(self) -> None:
        err = Exception("Not Found")
        err.status_code = 404
        tc = MagicMock()
        tc.get_order_by_id.side_effect = err

        client = _make_client(tc=tc)
        result = await client.get_order("does-not-exist")

        assert result is None

    async def test_get_order_by_client_id_supports_crash_reconciliation(self) -> None:
        tc = MagicMock()
        tc.get_order_by_client_id.return_value = _raw_order(status="accepted")
        client = _make_client(tc=tc)

        result = await client.get_order_by_client_id("client-123")

        assert result is not None
        assert result.status == "accepted"
        tc.get_order_by_client_id.assert_called_once_with("client-123")

    async def test_get_order_by_client_id_returns_none_when_absent(self) -> None:
        err = Exception("Not Found")
        err.status_code = 404
        tc = MagicMock()
        tc.get_order_by_client_id.side_effect = err
        client = _make_client(tc=tc)

        assert await client.get_order_by_client_id("missing") is None


# ── get_clock ─────────────────────────────────────────────────────────────────


class TestGetClock:
    async def test_drift_positive_when_local_ahead(self) -> None:
        alpaca_ts = datetime(2026, 5, 9, 14, 0, 0, tzinfo=UTC)
        local_ts = alpaca_ts + timedelta(seconds=2)

        raw = MagicMock()
        raw.timestamp = alpaca_ts
        raw.is_open = True
        raw.next_open = None
        raw.next_close = None

        tc = MagicMock()
        tc.get_clock.return_value = raw

        client = _make_client(tc=tc)

        with patch("src.broker.alpaca_client.datetime") as mock_dt:
            mock_dt.now.return_value = local_ts
            snap = await client.get_clock()

        assert isinstance(snap, ClockSnapshot)
        assert snap.drift_seconds == pytest.approx(2.0, abs=0.01)
        assert snap.is_open is True

    async def test_local_ts_captured_before_sdk_call(self) -> None:
        """Verify local_ts precedes the SDK call by checking call order."""
        call_log: list[str] = []

        alpaca_ts = datetime(2026, 5, 9, 14, 0, 0, tzinfo=UTC)
        raw = MagicMock()
        raw.timestamp = alpaca_ts
        raw.is_open = False
        raw.next_open = None
        raw.next_close = None

        tc = MagicMock()
        def _get_clock_side_effect():
            call_log.append("sdk_call")
            return raw
        tc.get_clock.side_effect = _get_clock_side_effect

        import src.broker.alpaca_client as mod

        original_datetime = mod.datetime

        class _TrackingDatetime:
            @staticmethod
            def now(tz=None):
                call_log.append("local_ts")
                return original_datetime.now(tz)

        client = _make_client(tc=tc)

        with patch.object(mod, "datetime", _TrackingDatetime):
            await client.get_clock()

        assert call_log[0] == "local_ts", "local_ts must be captured before SDK call"
        assert "sdk_call" in call_log


# ── get_bars ──────────────────────────────────────────────────────────────────


class TestGetBars:
    async def test_raises_on_naive_start(self) -> None:
        client = _make_client()
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.get_bars(
                "QQQ",
                start=datetime(2026, 5, 1, 9, 30),   # naive
                end=datetime(2026, 5, 1, 16, 0, tzinfo=UTC),
            )

    async def test_raises_on_naive_end(self) -> None:
        client = _make_client()
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.get_bars(
                "QQQ",
                start=datetime(2026, 5, 1, 9, 30, tzinfo=UTC),
                end=datetime(2026, 5, 1, 16, 0),   # naive
            )

    async def test_returns_bar_data_list(self) -> None:
        raw_bar = MagicMock()
        raw_bar.timestamp = datetime(2026, 5, 9, 13, 30, tzinfo=UTC)
        raw_bar.open = 480.0
        raw_bar.high = 482.0
        raw_bar.low = 479.0
        raw_bar.close = 481.0
        raw_bar.volume = 1_000_000.0
        raw_bar.vwap = 480.5

        bar_set = MagicMock()
        bar_set.data = {"QQQ": [raw_bar]}

        dc = MagicMock()
        dc.get_stock_bars.return_value = bar_set

        client = _make_client(dc=dc)

        bars = await client.get_bars(
            "QQQ",
            start=datetime(2026, 5, 9, 13, 30, tzinfo=UTC),
            end=datetime(2026, 5, 9, 20, 0, tzinfo=UTC),
        )

        assert len(bars) == 1
        bar = bars[0]
        assert isinstance(bar, BarData)
        assert bar.symbol == "QQQ"
        assert bar.open == 480.0
        assert bar.vwap == 480.5

    async def test_empty_symbol_returns_empty_list(self) -> None:
        bar_set = MagicMock()
        bar_set.data = {}

        dc = MagicMock()
        dc.get_stock_bars.return_value = bar_set

        client = _make_client(dc=dc)

        bars = await client.get_bars(
            "UNKNOWN",
            start=datetime(2026, 5, 9, 13, 30, tzinfo=UTC),
            end=datetime(2026, 5, 9, 20, 0, tzinfo=UTC),
        )

        assert bars == []

    async def test_passes_explicit_adjustment_to_historical_request(self) -> None:
        bar_set = MagicMock()
        bar_set.data = {}
        dc = MagicMock()
        dc.get_stock_bars.return_value = bar_set
        client = _make_client(dc=dc)

        await client.get_bars(
            "SPY",
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 1, tzinfo=UTC),
            adjustment=Adjustment.ALL,
        )

        request = dc.get_stock_bars.call_args.args[0]
        assert request.adjustment == Adjustment.ALL


# ── get_market_calendar ───────────────────────────────────────────────────────


class TestGetMarketCalendar:
    async def test_returns_market_calendar_days(self) -> None:
        sdk_day = MagicMock()
        sdk_day.date = date(2026, 5, 9)
        sdk_day.open = datetime(2026, 5, 9, 9, 30)   # naive ET
        sdk_day.close = datetime(2026, 5, 9, 16, 0)  # naive ET

        tc = MagicMock()
        tc.get_calendar.return_value = [sdk_day]

        client = _make_client(tc=tc)
        days = await client.get_market_calendar(date(2026, 5, 9), date(2026, 5, 9))

        assert len(days) == 1
        assert isinstance(days[0], MarketCalendarDay)
        assert days[0].session_open_utc == datetime(2026, 5, 9, 13, 30, tzinfo=UTC)


# ── Retry behaviour ───────────────────────────────────────────────────────────


class TestRetry:
    async def test_retries_on_server_error_then_succeeds(self) -> None:
        server_err = Exception("internal server error")
        server_err.status_code = 500

        tc = MagicMock()
        tc.get_account.side_effect = [
            server_err,
            MagicMock(
                buying_power="50000",
                cash="20000",
                portfolio_value="80000",
                pattern_day_trader=False,
                trading_blocked=False,
                account_blocked=False,
            ),
        ]

        client = _make_client(tc=tc)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            snap = await client.get_account()

        assert snap.buying_power == 50000.0
        assert tc.get_account.call_count == 2

    async def test_raises_after_max_attempts_on_connection_error(self) -> None:
        tc = MagicMock()
        tc.get_account.side_effect = OSError("connection refused")

        client = _make_client(tc=tc)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(AlpacaConnectionError):
                await client.get_account()

        assert tc.get_account.call_count == 3  # _MAX_RETRY_ATTEMPTS

    async def test_429_triggers_backoff_and_retries(self) -> None:
        rate_exc = Exception("too many requests")
        rate_exc.status_code = 429
        rate_exc.response = MagicMock()
        rate_exc.response.headers = {"Retry-After": "2"}

        tc = MagicMock()
        tc.get_account.side_effect = [
            rate_exc,
            MagicMock(
                buying_power="10000",
                cash="5000",
                portfolio_value="15000",
                pattern_day_trader=False,
                trading_blocked=False,
                account_blocked=False,
            ),
        ]

        rl = _make_rate_limiter()
        client = _make_client(tc=tc, rl=rl)

        snap = await client.get_account()
        rl.backoff_on_429.assert_called_once_with(2.0)
        assert snap.buying_power == 10000.0

    async def test_4xx_raises_immediately_without_retry(self) -> None:
        exc = Exception("forbidden")
        exc.status_code = 403

        tc = MagicMock()
        tc.get_account.side_effect = exc

        client = _make_client(tc=tc)
        with pytest.raises(AlpacaOrderRejectedError):
            await client.get_account()

        assert tc.get_account.call_count == 1
