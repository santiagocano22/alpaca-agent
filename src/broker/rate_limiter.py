"""Token bucket rate limiter for Alpaca API calls.

The bucket refills continuously at ``rate`` tokens per second up to a
maximum of ``burst``.  Callers ``await acquire()`` and block until a token is
available.  On HTTP 429, call ``backoff_on_429(retry_after)`` to drain the
bucket and sleep for the prescribed interval before retrying.

Both ``clock`` and ``sleeper`` are injectable so tests can advance simulated
time without real sleeps.  Defaults: ``time.monotonic`` and ``asyncio.sleep``.
Note: do NOT bind ``asyncio.get_event_loop().time`` as a default argument —
that would evaluate before any event loop exists at import time.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from loguru import logger

# Tolerance for IEEE-754 rounding errors in token comparisons.
# The concrete failure case: 0.1 s × 10 tok/s = 0.09999999999999998 × 10
# = 0.9999999999999998 in float64 — not >= 1.0 exactly.  The resulting
# deficit (~2.2e-16) produces a wait so small the FakeClock cannot advance
# past float precision, causing an infinite loop of zero-duration sleeps.
#
# 1e-12 is chosen over 1e-9:
#  - It is ~4 orders of magnitude above float64 machine epsilon (~2.2e-16),
#    which cures the concrete rounding case with room to spare.
#  - It is tight enough that it never "gifts" a detectable token to a caller
#    (1e-12 of a token at any practical rate is immeasurable).
#  - Tests with 50 concurrent acquires pass cleanly with 1e-12 (verified).
_FLOAT_EPSILON = 1e-12


class TokenBucketLimiter:
    """Token bucket rate limiter.

    Thread/coroutine safety: an ``asyncio.Lock`` serialises refill and
    consumption, so concurrent ``acquire()`` calls never double-spend tokens.
    The lock is released BEFORE sleeping so other coroutines can make progress.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        self._rate = rate
        self._burst = burst
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleeper: Callable[[float], Awaitable[None]] = (
            sleeper if sleeper is not None else asyncio.sleep
        )
        self._tokens: float = float(burst)
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def acquire(self, tokens: int = 1) -> None:
        """Block until ``tokens`` tokens are available, then consume them.

        Raises:
            ValueError: if ``tokens`` exceeds ``burst`` (can never be satisfied)
                or if ``tokens < 1``.
        """
        if tokens > self._burst:
            raise ValueError(
                f"requested {tokens} tokens but burst is {self._burst}; "
                "this can never be satisfied"
            )
        if tokens < 1:
            raise ValueError(f"tokens must be >= 1, got {tokens}")

        while True:
            async with self._lock:
                now = self._clock()
                elapsed = now - self._last
                self._tokens = min(
                    float(self._burst),
                    self._tokens + elapsed * self._rate,
                )
                self._last = now
                # Subtract epsilon to absorb IEEE-754 rounding (see _FLOAT_EPSILON).
                if self._tokens >= tokens - _FLOAT_EPSILON:
                    new_tokens = self._tokens - tokens
                    if new_tokens < -_FLOAT_EPSILON:
                        # Should not happen: deficit exceeds rounding tolerance.
                        logger.warning(
                            "rate_limiter: consumed more tokens than available "
                            "(available={}, requested={}, deficit={})",
                            self._tokens, tokens, new_tokens,
                        )
                    self._tokens = max(0.0, new_tokens)
                    return
                deficit = tokens - self._tokens
                wait = deficit / self._rate
            # ← lock released before sleeping
            await self._sleeper(wait)

    async def backoff_on_429(self, retry_after: float) -> None:
        """Drain the bucket to zero and sleep ``retry_after`` seconds.

        Called when Alpaca returns HTTP 429.  After waking, ``_last`` is
        updated so the bucket does not receive free tokens for the time spent
        sleeping — the next ``acquire()`` starts from an empty bucket.
        """
        async with self._lock:
            self._tokens = 0.0
            self._last = self._clock()
        await self._sleeper(retry_after)
        async with self._lock:
            self._last = self._clock()
