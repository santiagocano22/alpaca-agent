"""Alpaca REST client wrapper.

All SDK calls are run via ``asyncio.to_thread`` (alpaca-py is synchronous
under the hood).  Every exception from the SDK is mapped to our
``BrokerError`` hierarchy before leaving this module so that callers
never need to import alpaca-py exception types.

Retry policy
------------
- HTTP 429           → ``rate_limiter.backoff_on_429(retry_after)`` then retry
- HTTP 5xx           → exponential backoff + retry (up to ``_MAX_RETRY_ATTEMPTS``)
- Network/OS errors  → exponential backoff + retry
- HTTP 4xx (≠ 429)  → raise immediately, no retry

Dry-run mode
------------
When ``settings.dry_run=True``, ``submit_order`` returns a simulated
``OrderSubmitResult`` with ``simulated=True`` and never calls Alpaca.
Downstream code must check the ``simulated`` field — do not rely on the
``alpaca_order_id`` prefix ("dry-…") for logic.

422 classification
------------------
Patterns are matched case-insensitively with ``re.search`` so minor wording
variations in Alpaca's API responses are handled gracefully.  Unknown 422
messages fall back to ``AlpacaOrderRejectedError``.  The raw response body
is always logged at ERROR level before mapping so that new patterns can be
added when they surface in production logs.

Wordings sourced from:
- Alpaca Markets API documentation (v2)
- alpaca-py GitHub issues #189, #214, #301
"""
from __future__ import annotations

import asyncio
import inspect
import random
import re
from datetime import UTC, date, datetime
from typing import TypeVar
from zoneinfo import ZoneInfo

from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
)
from loguru import logger

from src.broker.exceptions import (
    AlpacaClockDriftError,  # noqa: F401 — re-exported for callers' convenience
    AlpacaConnectionError,
    AlpacaDuplicateOrderError,
    AlpacaInsufficientFundsError,
    AlpacaOrderError,
    AlpacaOrderRejectedError,
    AlpacaRateLimitError,
    AlpacaServerError,
    AlpacaSymbolNotFoundError,
    BrokerError,
)
from src.broker.rate_limiter import TokenBucketLimiter
from src.broker.schemas import (
    AccountSnapshot,
    BarData,
    ClockSnapshot,
    MarketCalendarDay,
    OrderIntent,
    OrderStatusResult,
    OrderSubmitResult,
    OrderType,
    PositionSnapshot,
)

T = TypeVar("T")

ET = ZoneInfo("America/New_York")

_MAX_RETRY_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 1.0
_PRICE_CACHE_TTL = 0.5  # seconds

# ── 422 classification ────────────────────────────────────────────────────────
# Extend this list when new patterns appear in production logs.
_422_PATTERNS: list[tuple[re.Pattern, type[AlpacaOrderError]]] = [
    (
        re.compile(r"insufficient.*(buying\s+power|funds)", re.I),
        AlpacaInsufficientFundsError,
    ),
    (
        re.compile(r"duplicate.*(client[_ ]order[_ ]id|order)", re.I),
        AlpacaDuplicateOrderError,
    ),
    (
        re.compile(r"client[_ ]order[_ ]id.*(unique|exists|taken)", re.I),
        AlpacaDuplicateOrderError,
    ),
    (
        re.compile(r"order.*already\s+exists", re.I),
        AlpacaDuplicateOrderError,
    ),
]


def _classify_422(message: str) -> type[AlpacaOrderError]:
    for pattern, exc_class in _422_PATTERNS:
        if pattern.search(message):
            return exc_class
    return AlpacaOrderRejectedError


# ── Calendar mapping ──────────────────────────────────────────────────────────


def _map_calendar_day(sdk_cal) -> MarketCalendarDay:
    """Convert an alpaca-py Calendar object to MarketCalendarDay.

    Verified 2026-05-09: alpaca-py returns Calendar.open/close as naive
    datetime objects in ET (tzinfo=None).  The RuntimeError guards detect
    if the SDK ever starts returning tz-aware values so we can update this.
    """
    if sdk_cal.open.tzinfo is not None:
        raise RuntimeError(
            f"Alpaca SDK changed: Calendar.open is now tz-aware "
            f"({sdk_cal.open.tzinfo}). Review _map_calendar_day."
        )
    if sdk_cal.close.tzinfo is not None:
        raise RuntimeError(
            f"Alpaca SDK changed: Calendar.close is now tz-aware "
            f"({sdk_cal.close.tzinfo}). Review _map_calendar_day."
        )
    open_et = sdk_cal.open.replace(tzinfo=ET)
    close_et = sdk_cal.close.replace(tzinfo=ET)
    return MarketCalendarDay(
        date=sdk_cal.date,
        session_open_utc=open_et.astimezone(UTC),
        session_close_utc=close_et.astimezone(UTC),
    )


# ── AlpacaClient ──────────────────────────────────────────────────────────────


class AlpacaClient:
    """Async wrapper around alpaca-py's synchronous REST clients."""

    def __init__(
        self,
        trading_client,                   # alpaca.trading.client.TradingClient
        data_client,                      # alpaca.data.historical.StockHistoricalDataClient
        settings,                         # src.config.Settings
        rate_limiter: TokenBucketLimiter,
    ) -> None:
        self._tc = trading_client
        self._dc = data_client
        self._settings = settings
        self._rl = rate_limiter
        # Brief price cache: symbol → (price, fetched_at); entries expire after 0.5 s
        self._price_cache: dict[str, tuple[float, datetime]] = {}

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_account(self) -> AccountSnapshot:
        """Return current account state (buying power, PDT flag, etc.)."""
        raw = await self._with_retry(self._tc.get_account)
        return AccountSnapshot(
            buying_power=float(raw.buying_power),
            cash=float(raw.cash),
            portfolio_value=float(raw.portfolio_value),
            pattern_day_trader=bool(raw.pattern_day_trader),
            trading_blocked=bool(raw.trading_blocked),
            account_blocked=bool(raw.account_blocked),
            snapshot_at=datetime.now(UTC),
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[PositionSnapshot]:
        """Return all open positions."""
        raw_list = await self._with_retry(self._tc.get_all_positions)
        return [self._map_position(p) for p in raw_list]

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        """Return the open position for ``symbol``, or ``None`` if flat."""
        sym = symbol.upper()
        try:
            raw = await self._with_retry(lambda: self._tc.get_open_position(sym))
        except AlpacaSymbolNotFoundError:
            return None
        return self._map_position(raw)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def submit_order(self, intent: OrderIntent) -> OrderSubmitResult:
        """Submit an order to Alpaca.

        In dry-run mode returns a simulated result without calling the API.
        On duplicate client_order_id, fetches the existing order and returns
        it (idempotent at-least-once delivery).

        Raises:
            AlpacaRateLimitError, AlpacaOrderRejectedError,
            AlpacaInsufficientFundsError, AlpacaServerError,
            AlpacaConnectionError
        """
        bound = logger.bind(
            method="submit_order",
            symbol=intent.symbol,
            client_order_id=intent.client_order_id,
        )

        if self._settings.dry_run:
            bound.info(
                "DRY RUN: would submit {} {} {} {}",
                intent.side.value.upper(), intent.qty, intent.symbol,
                intent.order_type.value,
            )
            return OrderSubmitResult(
                alpaca_order_id=f"dry-{intent.client_order_id}",
                client_order_id=intent.client_order_id,
                status="simulated",
                submitted_at=datetime.now(UTC),
                simulated=True,
            )

        await self._rl.acquire()
        request = self._build_order_request(intent)

        try:
            raw = await self._with_retry(
                lambda: self._tc.submit_order(request),
                symbol=intent.symbol,
            )
        except AlpacaDuplicateOrderError:
            bound.warning("Duplicate client_order_id; fetching existing order")
            raw = await self._with_retry(
                lambda: self._tc.get_order_by_client_id(intent.client_order_id),
                symbol=intent.symbol,
            )

        return self._map_order_submit(raw, intent.client_order_id)

    async def cancel_order(self, alpaca_order_id: str) -> None:
        """Cancel an order by its Alpaca ID; silent if already cancelled/filled."""
        try:
            await self._with_retry(lambda: self._tc.cancel_order_by_id(alpaca_order_id))
        except AlpacaOrderRejectedError:
            pass  # already in a terminal state

    async def get_order(self, alpaca_order_id: str) -> OrderStatusResult | None:
        """Return current state of an order, or ``None`` if not found."""
        try:
            raw = await self._with_retry(
                lambda: self._tc.get_order_by_id(alpaca_order_id)
            )
        except AlpacaSymbolNotFoundError:
            return None
        return self._map_order_status(raw)

    # ── Market data ───────────────────────────────────────────────────────────

    async def get_latest_price(self, symbol: str) -> float:
        """Return the price of the most recent trade for ``symbol``.

        Results are cached for 500 ms to avoid redundant API calls when
        multiple risk checks reference the same symbol in quick succession.
        """
        symbol = symbol.upper()
        now = datetime.now(UTC)
        cached = self._price_cache.get(symbol)
        if cached and (now - cached[1]).total_seconds() < _PRICE_CACHE_TTL:
            return cached[0]

        req = StockLatestTradeRequest(symbol_or_symbols=symbol)

        resp = await self._with_retry(
            lambda: self._dc.get_latest_stock_trades(req),
            symbol=symbol,
        )
        price = float(resp[symbol].price)
        self._price_cache[symbol] = (price, datetime.now(UTC))
        return price

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe=None,
    ) -> list[BarData]:
        """Return OHLCV bars for ``symbol`` in [start, end].

        ``start`` and ``end`` must be timezone-aware datetimes.
        alpaca-py handles multi-page responses internally; this method
        always returns the complete list across all pages.
        ``timeframe`` defaults to TimeFrame.Minute when not supplied.

        Raises:
            ValueError: if ``start`` or ``end`` is a naive datetime.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError(
                "start and end must be timezone-aware; received naive datetime"
            )

        tf = timeframe if timeframe is not None else AlpacaTimeFrame.Minute
        symbol = symbol.upper()
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=tf,
            limit=10000,
        )

        bar_set = await self._with_retry(
            lambda: self._dc.get_stock_bars(req),
            symbol=symbol,
        )
        bars = bar_set.get(symbol, [])
        return [self._map_bar(b, symbol) for b in bars]

    # ── Clock ─────────────────────────────────────────────────────────────────

    async def get_clock(self) -> ClockSnapshot:
        """Return Alpaca's server clock for drift detection.

        drift_seconds = local_timestamp - alpaca_timestamp
        Positive → local clock is ahead; negative → local is behind.
        local_ts is captured BEFORE the SDK call to avoid inflating drift
        with network round-trip time.
        """
        local_ts = datetime.now(UTC)
        raw = await self._with_retry(self._tc.get_clock)
        alpaca_ts = raw.timestamp
        if alpaca_ts.tzinfo is None:
            alpaca_ts = alpaca_ts.replace(tzinfo=UTC)
        drift = (local_ts - alpaca_ts).total_seconds()
        return ClockSnapshot(
            alpaca_timestamp=alpaca_ts,
            local_timestamp=local_ts,
            drift_seconds=drift,
            is_open=bool(raw.is_open),
            next_open=getattr(raw, "next_open", None),
            next_close=getattr(raw, "next_close", None),
        )

    # ── Calendar ──────────────────────────────────────────────────────────────

    async def get_market_calendar(
        self, start: date, end: date
    ) -> list[MarketCalendarDay]:
        """Return trading schedule for [start, end] as UTC-normalised objects."""
        req = GetCalendarRequest(start=start, end=end)
        raw_days = await self._with_retry(lambda: self._tc.get_calendar(req))
        return [_map_calendar_day(day) for day in raw_days]

    async def close_position(self, symbol: str) -> None:
        """Close an open position for ``symbol`` with a market order.

        Uses Alpaca's ``DELETE /v2/positions/{symbol}`` endpoint which submits
        a market order to liquidate the full position.  No-op when the position
        does not exist (404 is silently ignored).
        """
        if self._settings.dry_run:
            logger.info("dry_run: close_position({}) suppressed", symbol)
            return
        try:
            await self._call(lambda: self._tc.close_position(symbol))
            logger.info("close_position: liquidation order submitted for {}", symbol)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status == 404:
                logger.info("close_position: no open position for {} (404 ignored)", symbol)
                return
            raise self._map_exc(exc, symbol=symbol) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _call(self, fn, *, symbol: str | None = None):
        """Run a sync SDK callable in a thread and map all exceptions."""
        try:
            return await asyncio.to_thread(fn)
        except (OSError, ConnectionError, TimeoutError) as exc:
            raise AlpacaConnectionError(str(exc)) from exc
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 422:
                logger.error(
                    "Alpaca 422 body (symbol={}): {!r}", symbol, str(exc)
                )
            raise self._map_exc(exc, symbol=symbol) from exc

    async def _with_retry(
        self,
        fn,
        *,
        symbol: str | None = None,
        max_attempts: int = _MAX_RETRY_ATTEMPTS,
    ) -> T:
        """Retry ``fn`` on 429 / 5xx / connection errors; propagate 4xx immediately.

        ``fn`` may be a callable (sync SDK function) or an async callable.
        Sync callables are wrapped in ``_call`` for thread dispatch + exception
        mapping; async callables are awaited directly.
        """
        last_exc: BrokerError | None = None
        for attempt in range(max_attempts):
            try:
                if inspect.iscoroutinefunction(fn):
                    return await fn()
                return await self._call(fn, symbol=symbol)
            except AlpacaRateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "429 from Alpaca (attempt {}/{}); backing off {:.1f}s",
                    attempt + 1, max_attempts, exc.retry_after,
                )
                await self._rl.backoff_on_429(exc.retry_after)
            except (AlpacaServerError, AlpacaConnectionError) as exc:
                last_exc = exc
                if attempt == max_attempts - 1:
                    raise
                jitter = random.uniform(0.75, 1.25)
                delay = _BASE_BACKOFF_SECONDS * (2 ** attempt) * jitter
                logger.warning(
                    "Retriable error (attempt {}/{}): {}; sleeping {:.1f}s",
                    attempt + 1, max_attempts, exc, delay,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]  # reached only after 429 exhaustion

    def _map_exc(self, exc: Exception, *, symbol: str | None = None) -> BrokerError:
        """Map an SDK exception to our BrokerError hierarchy."""
        status = getattr(exc, "status_code", None)
        message = str(exc)

        if status == 429:
            response = getattr(exc, "response", None)
            retry_after = 1.0
            if response is not None:
                retry_after = float(
                    getattr(response, "headers", {}).get("Retry-After", 1.0)
                )
            return AlpacaRateLimitError(message, retry_after=retry_after)

        if status == 422:
            cls = _classify_422(message)
            return cls(message, alpaca_message=message)

        if status == 404:
            sym = symbol or "unknown"
            return AlpacaSymbolNotFoundError(message, symbol=sym)

        if status is not None and status >= 500:
            return AlpacaServerError(message, status_code=status)

        if status is not None and 400 <= status < 500:
            return AlpacaOrderRejectedError(message, alpaca_message=message)

        return AlpacaConnectionError(message)

    # ── Mapping helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _map_position(raw) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=raw.symbol,
            qty=float(raw.qty),
            side=raw.side.value if hasattr(raw.side, "value") else str(raw.side),
            market_value=float(raw.market_value or 0),
            avg_entry_price=float(raw.avg_entry_price or 0),
            unrealized_pl=float(raw.unrealized_pl or 0),
            unrealized_plpc=float(raw.unrealized_plpc or 0),
            current_price=float(raw.current_price or 0),
        )

    @staticmethod
    def _map_order_submit(raw, client_order_id: str) -> OrderSubmitResult:
        submitted_at = raw.submitted_at
        if submitted_at is not None and submitted_at.tzinfo is None:
            submitted_at = submitted_at.replace(tzinfo=UTC)
        return OrderSubmitResult(
            alpaca_order_id=str(raw.id),
            client_order_id=client_order_id or str(getattr(raw, "client_order_id", "")),
            status=raw.status.value if hasattr(raw.status, "value") else str(raw.status),
            submitted_at=submitted_at or datetime.now(UTC),
            simulated=False,
        )

    @staticmethod
    def _map_order_status(raw) -> OrderStatusResult:
        from src.broker.schemas import OrderSide as BrokerOrderSide

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

    @staticmethod
    def _map_bar(raw, symbol: str) -> BarData:
        ts = raw.timestamp
        if ts is not None and hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return BarData(
            symbol=symbol,
            timestamp=ts,
            open=float(raw.open),
            high=float(raw.high),
            low=float(raw.low),
            close=float(raw.close),
            volume=float(raw.volume),
            vwap=float(raw.vwap) if getattr(raw, "vwap", None) is not None else None,
        )

    def _build_order_request(self, intent: OrderIntent):
        common = dict(
            symbol=intent.symbol,
            qty=intent.qty,
            side=AlpacaOrderSide(intent.side.value),
            time_in_force=AlpacaTimeInForce(intent.time_in_force.value),
            client_order_id=intent.client_order_id,
        )
        if intent.order_type == OrderType.LIMIT:
            return LimitOrderRequest(**common, limit_price=intent.limit_price)
        if intent.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                **common,
                limit_price=intent.limit_price,
                stop_price=intent.stop_price,
            )
        return MarketOrderRequest(**common)
