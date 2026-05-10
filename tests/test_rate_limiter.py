"""Tests for src/broker/rate_limiter.py.

Strategy: inject a FakeClock whose ``sleep()`` method both advances the
virtual clock AND yields to the event loop.  No patching of asyncio.sleep —
the sleeper is passed directly to TokenBucketLimiter's constructor.
This makes tests deterministic and near-instant (zero real wall-time sleeps).
"""
from __future__ import annotations

import asyncio

import pytest

from src.broker.rate_limiter import TokenBucketLimiter


# ── Helpers ───────────────────────────────────────────────────────────────────


class FakeClock:
    """Virtual clock that advances only when sleep() is called."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)  # yield to event loop without real delay


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            TokenBucketLimiter(rate=0.0, burst=5)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            TokenBucketLimiter(rate=-1.0, burst=5)

    def test_zero_burst_raises(self) -> None:
        with pytest.raises(ValueError, match="burst"):
            TokenBucketLimiter(rate=1.0, burst=0)

    def test_starts_full(self) -> None:
        clock = FakeClock()
        lim = TokenBucketLimiter(rate=2.0, burst=5, clock=clock, sleeper=clock.sleep)
        assert lim._tokens == 5.0

    async def test_acquire_more_than_burst_raises(self) -> None:
        clock = FakeClock()
        lim = TokenBucketLimiter(rate=10.0, burst=5, clock=clock, sleeper=clock.sleep)
        with pytest.raises(ValueError, match="never be satisfied"):
            await lim.acquire(tokens=10)

    async def test_acquire_zero_tokens_raises(self) -> None:
        clock = FakeClock()
        lim = TokenBucketLimiter(rate=10.0, burst=5, clock=clock, sleeper=clock.sleep)
        with pytest.raises(ValueError, match="tokens must be >= 1"):
            await lim.acquire(tokens=0)


# ── Concurrent acquires pace correctly ───────────────────────────────────────


async def test_rate_limiter_concurrent_acquire_paces_correctly() -> None:
    """50 concurrent acquires with fake clock verify that rate is honoured.

    With rate=10/s and burst=5, the first 5 are free and the remaining
    45 each require 1/10 s of simulated time → clock must advance exactly
    45/10 = 4.5 s of virtual time.
    """
    clock = FakeClock()
    limiter = TokenBucketLimiter(
        rate=10.0,
        burst=5,
        clock=clock,
        sleeper=clock.sleep,
    )

    await asyncio.gather(*[limiter.acquire() for _ in range(50)])

    assert 4.4 <= clock.now <= 4.6, f"clock advanced {clock.now}s, expected ~4.5"


# ── Backoff drains and resets the bucket ──────────────────────────────────────


async def test_rate_limiter_backoff_drains_bucket() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate=10.0, burst=10, clock=clock, sleeper=clock.sleep)

    # bucket full → first acquire is instantaneous
    await limiter.acquire()
    t_before = clock.now  # == 0.0

    # backoff drains bucket and sleeps 3 s of virtual time
    await limiter.backoff_on_429(retry_after=3.0)
    assert clock.now == t_before + 3.0

    # bucket is empty after backoff; next acquire must wait 1 token at 10/s = 0.1 s
    await limiter.acquire()
    assert clock.now >= t_before + 3.1


# ── No starvation under sustained load ───────────────────────────────────────


async def test_rate_limiter_no_starvation() -> None:
    """All workers eventually get a token — no coroutine waits forever."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate=20.0, burst=5, clock=clock, sleeper=clock.sleep)

    completion_order: list[int] = []

    async def worker(worker_id: int) -> None:
        await limiter.acquire()
        completion_order.append(worker_id)

    await asyncio.gather(*[worker(i) for i in range(30)])

    assert len(completion_order) == 30
    assert set(completion_order) == set(range(30))


# ── Lock is released during sleep (regression) ───────────────────────────────


async def test_acquire_releases_lock_during_sleep() -> None:
    """Regression: if sleep were inside the lock this test would hang."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate=1.0, burst=1, clock=clock, sleeper=clock.sleep)

    # consume the single burst token
    await limiter.acquire()

    # 5 concurrent acquires; each needs 1 token at 1/s → 5 s of virtual time
    await asyncio.gather(*[limiter.acquire() for _ in range(5)])

    assert 4.9 <= clock.now <= 5.1
