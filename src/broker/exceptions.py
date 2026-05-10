"""Broker-layer exception hierarchy.

All exceptions raised by alpaca_client.py are one of these types.
The rest of the codebase catches BrokerError (or subclasses) — never raw
alpaca-py or httpx exceptions — so the SDK can be swapped without touching
business logic.

Retry policy summary (implemented in alpaca_client.py):
  - AlpacaConnectionError  → retry with exponential backoff
  - AlpacaRateLimitError   → wait retry_after seconds, then retry
  - AlpacaServerError      → retry with exponential backoff (5xx)
  - All 4xx except 429     → do NOT retry (client error, retrying won't help)
"""
from __future__ import annotations


class BrokerError(Exception):
    """Root of all broker-layer errors."""


# ── Connectivity ──────────────────────────────────────────────────────────────


class AlpacaConnectionError(BrokerError):
    """TCP reset, read timeout, WebSocket disconnect, or any network-level failure."""


class AlpacaRateLimitError(BrokerError):
    """HTTP 429 — Too Many Requests.

    Read ``retry_after`` (seconds) before scheduling the next attempt.
    ``backoff_on_429()`` on TokenBucketLimiter drains the bucket and sleeps.
    """

    def __init__(self, msg: str, retry_after: float = 1.0) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class AlpacaServerError(BrokerError):
    """HTTP 5xx — retriable with exponential backoff."""

    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


# ── Order errors ──────────────────────────────────────────────────────────────


class AlpacaOrderError(BrokerError):
    """Base for semantic order errors (order-specific 422 responses)."""

    def __init__(self, msg: str, alpaca_message: str = "") -> None:
        super().__init__(msg)
        self.alpaca_message = alpaca_message


class AlpacaOrderRejectedError(AlpacaOrderError):
    """Alpaca rejected the order for a business reason (PDT rule, market hours,
    asset not shortable, etc.). Do not retry — the condition must change first.
    """


class AlpacaDuplicateOrderError(AlpacaOrderError):
    """An order with this client_order_id already exists on Alpaca.

    Callers should query the existing order and return it rather than
    treating this as a hard failure — it signals at-least-once delivery.
    """


class AlpacaInsufficientFundsError(AlpacaOrderError):
    """Buying power insufficient for the requested order.

    Distinct from INSUFFICIENT_BUYING_POWER in ValidationResult: that check
    runs *before* sending to Alpaca; this exception fires when Alpaca itself
    rejects the order for this reason (e.g. buying power changed between
    validation and submission due to another fill).
    """


# ── Data / asset errors ───────────────────────────────────────────────────────


class AlpacaSymbolNotFoundError(BrokerError):
    """Ticker does not exist, is delisted, or is not available for trading."""

    def __init__(self, msg: str, symbol: str) -> None:
        super().__init__(msg)
        self.symbol = symbol


class AlpacaClockDriftError(BrokerError):
    """Local system clock deviates more than the configured threshold from
    Alpaca's server clock. Orders are blocked until the clock is re-synced.
    """

    def __init__(self, msg: str, drift_seconds: float) -> None:
        super().__init__(msg)
        self.drift_seconds = drift_seconds
