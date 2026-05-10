"""Asset metadata cache with TTL.

Wraps Alpaca's ``get_asset`` endpoint with a per-symbol TTL cache so the
risk_manager can call ``cache.get(symbol)`` on every order check without
hammering the API.

Key behaviours:
- Hit: returns cached ``AssetInfo`` if ``now < expires_at``.
- Miss / expired: fetches from Alpaca, caches the result, returns it.
- 404 → raises ``AlpacaSymbolNotFoundError`` and does NOT cache the failure
  (so the next call retries rather than serving a stale "not found").
- ``invalidate()`` clears the whole cache (e.g. after a strategy change).
- ``invalidate_symbol(symbol)`` removes one entry without touching others
  (e.g. when a symbol becomes untradeable at runtime).
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from loguru import logger

from src.broker.exceptions import AlpacaConnectionError, AlpacaSymbolNotFoundError
from src.broker.schemas import AssetInfo


@dataclass(slots=True)
class _CacheEntry:
    asset: AssetInfo
    expires_at: datetime


class AssetCache:
    def __init__(
        self,
        trading_client,                                      # TradingClient from alpaca-py
        ttl_seconds: int = 3600,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = trading_client
        self._ttl = ttl_seconds
        self._clock: Callable[[], datetime] = clock if clock is not None else (
            lambda: datetime.now(UTC)
        )
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, symbol: str) -> AssetInfo:
        """Return ``AssetInfo`` for ``symbol``, from cache or Alpaca.

        Raises:
            AlpacaSymbolNotFoundError: ticker does not exist on Alpaca.
            AlpacaConnectionError: network error while fetching.
        """
        symbol = symbol.upper()
        async with self._lock:
            entry = self._cache.get(symbol)
            if entry is not None and entry.expires_at > self._clock():
                return entry.asset

        # Fetch outside the lock so a slow API call doesn't block other symbols.
        asset_info = await self._fetch(symbol)

        async with self._lock:
            self._cache[symbol] = _CacheEntry(
                asset=asset_info,
                expires_at=self._clock() + timedelta(seconds=self._ttl),
            )
        return asset_info

    async def invalidate(self) -> None:
        """Clear the entire cache (call after a strategy change)."""
        async with self._lock:
            self._cache.clear()
        logger.debug("AssetCache: full invalidation")

    async def invalidate_symbol(self, symbol: str) -> None:
        """Remove a single symbol from the cache without clearing others."""
        async with self._lock:
            removed = self._cache.pop(symbol.upper(), None)
        if removed:
            logger.debug("AssetCache: invalidated {}", symbol.upper())

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch(self, symbol: str) -> AssetInfo:
        try:
            raw = await asyncio.to_thread(self._client.get_asset, symbol)
        except Exception as exc:
            # Check for alpaca-py's APIError via duck-typing to avoid a hard
            # import dependency on the exception class at module load time.
            status = getattr(exc, "status_code", None)
            if status == 404:
                raise AlpacaSymbolNotFoundError(
                    f"Asset not found: {symbol}", symbol=symbol
                ) from exc
            raise AlpacaConnectionError(
                f"Failed to fetch asset {symbol}: {exc}"
            ) from exc

        return AssetInfo(
            symbol=raw.symbol,
            name=getattr(raw, "name", None) or raw.symbol,
            tradeable=bool(raw.tradeable),
            fractionable=bool(raw.fractionable),
            shortable=bool(raw.shortable),
            easy_to_borrow=bool(getattr(raw, "easy_to_borrow", False)),
            status=(
                raw.status.value
                if hasattr(raw.status, "value")
                else str(raw.status)
            ),
        )
