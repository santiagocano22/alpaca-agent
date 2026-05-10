"""Tests for src/broker/asset_cache.py."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.broker.asset_cache import AssetCache
from src.broker.exceptions import AlpacaConnectionError, AlpacaSymbolNotFoundError
from src.broker.schemas import AssetInfo


# ── Helpers ───────────────────────────────────────────────────────────────────


def _raw_asset(symbol: str = "QQQ") -> MagicMock:
    a = MagicMock()
    a.symbol = symbol
    a.name = f"{symbol} ETF"
    a.tradeable = True
    a.fractionable = True
    a.shortable = True
    a.easy_to_borrow = True
    a.status = MagicMock()
    a.status.value = "active"
    return a


def _make_cache(trading_client, *, ttl: int = 3600, now: datetime | None = None) -> AssetCache:
    t = now or datetime.now(UTC)
    return AssetCache(trading_client=trading_client, ttl_seconds=ttl, clock=lambda: t)


# ── Cache hit ─────────────────────────────────────────────────────────────────


class TestCacheHit:
    async def test_second_call_does_not_hit_api(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("QQQ")
        cache = _make_cache(client)

        await cache.get("QQQ")
        await cache.get("QQQ")

        assert client.get_asset.call_count == 1

    async def test_returns_same_asset_info_object(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("SPY")
        cache = _make_cache(client)

        first = await cache.get("SPY")
        second = await cache.get("SPY")

        assert first == second

    async def test_symbol_uppercased_before_lookup(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("SPY")
        cache = _make_cache(client)

        await cache.get("spy")
        await cache.get("SPY")

        assert client.get_asset.call_count == 1


# ── TTL expiry ────────────────────────────────────────────────────────────────


class TestTTLExpiry:
    async def test_expired_entry_refetches(self) -> None:
        t = datetime(2026, 1, 6, 15, 0, 0, tzinfo=UTC)
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("QQQ")

        cache = AssetCache(trading_client=client, ttl_seconds=60, clock=lambda: t)
        await cache.get("QQQ")
        assert client.get_asset.call_count == 1

        # Advance clock past TTL
        t = t + timedelta(seconds=61)
        await cache.get("QQQ")
        assert client.get_asset.call_count == 2

    async def test_fresh_entry_not_refetched(self) -> None:
        t = datetime(2026, 1, 6, 15, 0, 0, tzinfo=UTC)
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("QQQ")

        cache = AssetCache(trading_client=client, ttl_seconds=60, clock=lambda: t)
        await cache.get("QQQ")

        t = t + timedelta(seconds=30)  # still fresh
        await cache.get("QQQ")
        assert client.get_asset.call_count == 1


# ── Invalidation ──────────────────────────────────────────────────────────────


class TestInvalidation:
    async def test_invalidate_clears_all_symbols(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset()

        cache = _make_cache(client)
        await cache.get("QQQ")
        await cache.get("SPY")
        assert client.get_asset.call_count == 2

        await cache.invalidate()
        await cache.get("QQQ")
        await cache.get("SPY")
        assert client.get_asset.call_count == 4

    async def test_invalidate_symbol_removes_only_that_symbol(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset()

        cache = _make_cache(client)
        await cache.get("QQQ")
        await cache.get("SPY")

        await cache.invalidate_symbol("QQQ")
        await cache.get("QQQ")  # re-fetches
        await cache.get("SPY")  # still cached

        assert client.get_asset.call_count == 3

    async def test_invalidate_nonexistent_symbol_is_silent(self) -> None:
        client = MagicMock()
        cache = _make_cache(client)
        await cache.invalidate_symbol("NONEXISTENT")  # must not raise


# ── Error cases ───────────────────────────────────────────────────────────────


class TestErrors:
    async def test_404_raises_symbol_not_found(self) -> None:
        client = MagicMock()
        err = Exception("Not Found")
        err.status_code = 404
        client.get_asset.side_effect = err

        cache = _make_cache(client)
        with pytest.raises(AlpacaSymbolNotFoundError) as exc_info:
            await cache.get("FAKE")

        assert exc_info.value.symbol == "FAKE"

    async def test_404_not_cached_so_next_call_retries(self) -> None:
        t = datetime(2026, 1, 6, 15, 0, 0, tzinfo=UTC)
        client = MagicMock()

        not_found = Exception("Not Found")
        not_found.status_code = 404
        client.get_asset.side_effect = [not_found, _raw_asset("QQQ")]

        cache = AssetCache(trading_client=client, ttl_seconds=60, clock=lambda: t)

        with pytest.raises(AlpacaSymbolNotFoundError):
            await cache.get("QQQ")

        # Second call should hit API again (failure was not cached)
        result = await cache.get("QQQ")
        assert result.symbol == "QQQ"
        assert client.get_asset.call_count == 2

    async def test_network_error_raises_connection_error(self) -> None:
        client = MagicMock()
        client.get_asset.side_effect = OSError("connection refused")

        cache = _make_cache(client)
        with pytest.raises(AlpacaConnectionError):
            await cache.get("QQQ")

    async def test_returned_asset_info_fields(self) -> None:
        client = MagicMock()
        client.get_asset.return_value = _raw_asset("TSLA")

        cache = _make_cache(client)
        info = await cache.get("TSLA")

        assert isinstance(info, AssetInfo)
        assert info.symbol == "TSLA"
        assert info.tradeable is True
        assert info.status == "active"
