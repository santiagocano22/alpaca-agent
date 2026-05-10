"""Tests for src/broker/schemas.py and src/broker/exceptions.py.

Covers: OrderIntent validation, make_client_order_id uniqueness/format,
ValidationResult factory methods and approved↔code invariant,
RiskOverrides field constraints, and exception attributes.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.broker.exceptions import (
    AlpacaClockDriftError,
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
from src.broker.schemas import (
    OrderIntent,
    OrderSide,
    OrderType,
    RiskCheckCode,
    RiskOverrides,
    TimeInForce,
    ValidationResult,
    make_client_order_id,
)


# ── make_client_order_id ──────────────────────────────────────────────────────


class TestMakeClientOrderId:
    def test_length_within_alpaca_limit(self) -> None:
        oid = make_client_order_id(1, "QQQ")
        assert len(oid) <= 128

    def test_only_allowed_characters(self) -> None:
        import re
        oid = make_client_order_id(42, "TSLA")
        assert re.match(r"^[A-Za-z0-9_-]+$", oid), f"invalid chars in: {oid!r}"

    def test_symbol_uppercased(self) -> None:
        oid = make_client_order_id(1, "qqq")
        assert "QQQ" in oid

    def test_none_strategy_version_produces_valid_id(self) -> None:
        oid = make_client_order_id(None, "SPY")
        assert oid.startswith("0-SPY-")

    def test_uniqueness_1000_sequential(self) -> None:
        ids = {make_client_order_id(1, "QQQ") for _ in range(1_000)}
        assert len(ids) == 1_000, "collision among 1000 sequential IDs"

    def test_uniqueness_1000_concurrent(self) -> None:
        async def _generate() -> list[str]:
            return await asyncio.gather(
                *[asyncio.to_thread(make_client_order_id, 1, "QQQ") for _ in range(1_000)]
            )

        ids = asyncio.run(_generate())
        assert len(set(ids)) == 1_000, "collision among 1000 concurrent IDs"

    def test_dot_in_ticker_sanitized(self) -> None:
        """BRK.B has a dot; result must still pass Alpaca's charset requirement."""
        import re
        oid = make_client_order_id(1, "BRK.B")
        assert re.match(r"^[A-Za-z0-9_-]+$", oid), f"invalid chars in: {oid!r}"
        assert "BRK_B" in oid

    def test_dot_ticker_uniqueness_preserved(self) -> None:
        """Sanitization must not reduce entropy / cause collisions."""
        ids = {make_client_order_id(1, "BRK.B") for _ in range(100)}
        assert len(ids) == 100


# ── OrderIntent ───────────────────────────────────────────────────────────────


class TestOrderIntent:
    def _base(self, **kwargs) -> dict:
        return {
            "symbol": "QQQ",
            "side": OrderSide.BUY,
            "qty": 10.0,
            "client_order_id": make_client_order_id(1, "QQQ"),
            **kwargs,
        }

    def test_valid_market_order(self) -> None:
        intent = OrderIntent(**self._base())
        assert intent.order_type == OrderType.MARKET
        assert intent.is_closing is False

    def test_valid_limit_order(self) -> None:
        intent = OrderIntent(**self._base(order_type=OrderType.LIMIT, limit_price=450.0))
        assert intent.limit_price == 450.0

    def test_limit_order_without_price_raises(self) -> None:
        with pytest.raises(ValidationError, match="limit_price"):
            OrderIntent(**self._base(order_type=OrderType.LIMIT))

    def test_stop_limit_requires_both_prices(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(order_type=OrderType.STOP_LIMIT, limit_price=449.0))

    def test_stop_limit_with_both_prices(self) -> None:
        intent = OrderIntent(
            **self._base(order_type=OrderType.STOP_LIMIT, limit_price=449.0, stop_price=448.0)
        )
        assert intent.stop_price == 448.0

    def test_client_order_id_too_long_raises(self) -> None:
        with pytest.raises(ValidationError, match="128"):
            OrderIntent(**self._base(client_order_id="x" * 129))

    def test_client_order_id_invalid_chars_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"\[A-Za-z0-9"):
            OrderIntent(**self._base(client_order_id="bad id with spaces"))

    def test_client_order_id_with_at_sign_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(client_order_id="order@123"))

    def test_is_closing_defaults_false(self) -> None:
        assert OrderIntent(**self._base()).is_closing is False

    def test_is_closing_can_be_set_true(self) -> None:
        assert OrderIntent(**self._base(is_closing=True)).is_closing is True

    def test_qty_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(qty=0.0))

    def test_qty_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(qty=-5.0))

    def test_qty_fractional_allowed(self) -> None:
        intent = OrderIntent(**self._base(qty=0.5))
        assert intent.qty == 0.5

    def test_symbol_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(symbol=""))

    def test_symbol_with_spaces_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrderIntent(**self._base(symbol="QQ Q"))

    def test_symbol_uppercased(self) -> None:
        intent = OrderIntent(**self._base(symbol="spy"))
        assert intent.symbol == "SPY"


# ── ValidationResult ──────────────────────────────────────────────────────────


class TestValidationResult:
    def test_approve_factory(self) -> None:
        result = ValidationResult.approve("QQQ", estimated_order_value_usd=4500.0)
        assert result.approved is True
        assert result.code == RiskCheckCode.APPROVED
        assert result.estimated_order_value_usd == 4500.0

    def test_reject_factory(self) -> None:
        result = ValidationResult.reject(
            "QQQ",
            code=RiskCheckCode.BOT_PAUSED,
            reason="Bot is paused via /pause",
        )
        assert result.approved is False
        assert result.code == RiskCheckCode.BOT_PAUSED

    def test_checked_at_is_utc_aware(self) -> None:
        result = ValidationResult.approve("SPY")
        assert result.checked_at.tzinfo is not None
        assert result.checked_at.utcoffset().total_seconds() == 0.0

    def test_checked_at_is_recent(self) -> None:
        before = datetime.now(UTC)
        result = ValidationResult.approve("SPY")
        after = datetime.now(UTC)
        assert before <= result.checked_at <= after

    def test_approved_true_with_non_approved_code_raises(self) -> None:
        with pytest.raises(ValidationError):
            ValidationResult(
                approved=True,
                code=RiskCheckCode.BOT_PAUSED,
                reason="inconsistent",
                symbol="QQQ",
            )

    def test_approved_false_with_approved_code_raises(self) -> None:
        with pytest.raises(ValidationError):
            ValidationResult(
                approved=False,
                code=RiskCheckCode.APPROVED,
                reason="inconsistent",
                symbol="QQQ",
            )


# ── RiskOverrides ─────────────────────────────────────────────────────────────


class TestRiskOverrides:
    def test_empty_overrides_all_none(self) -> None:
        ro = RiskOverrides()
        assert ro.max_position_pct is None
        assert ro.max_total_exposure_pct is None

    def test_valid_overrides(self) -> None:
        ro = RiskOverrides(
            max_position_pct=5.0,
            max_total_exposure_pct=50.0,
            stop_loss_pct=2.0,
            max_concurrent_positions=5,
        )
        assert ro.max_position_pct == 5.0

    def test_pct_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(max_position_pct=101.0)

    def test_pct_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(max_position_pct=0.0)

    def test_stop_loss_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(stop_loss_pct=0.0)

    def test_stop_loss_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(stop_loss_pct=51.0)  # above 50%

    def test_stop_loss_at_maximum_allowed(self) -> None:
        ro = RiskOverrides(stop_loss_pct=50.0)
        assert ro.stop_loss_pct == 50.0

    def test_stop_loss_small_positive_allowed(self) -> None:
        ro = RiskOverrides(stop_loss_pct=0.01)
        assert ro.stop_loss_pct == 0.01

    def test_max_concurrent_positions_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(max_concurrent_positions=0)

    def test_max_concurrent_positions_above_limit_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskOverrides(max_concurrent_positions=51)


# ── Exception hierarchy ───────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_are_broker_errors(self) -> None:
        exceptions = [
            AlpacaConnectionError("x"),
            AlpacaRateLimitError("x", retry_after=5.0),
            AlpacaServerError("x", status_code=500),
            AlpacaOrderRejectedError("x", alpaca_message="PDT"),
            AlpacaDuplicateOrderError("x"),
            AlpacaInsufficientFundsError("x"),
            AlpacaSymbolNotFoundError("x", symbol="FAKE"),
            AlpacaClockDriftError("x", drift_seconds=10.0),
        ]
        for exc in exceptions:
            assert isinstance(exc, BrokerError), f"{type(exc)} not a BrokerError"

    def test_rate_limit_carries_retry_after(self) -> None:
        exc = AlpacaRateLimitError("too many requests", retry_after=42.5)
        assert exc.retry_after == 42.5

    def test_server_error_carries_status_code(self) -> None:
        exc = AlpacaServerError("internal error", status_code=503)
        assert exc.status_code == 503

    def test_symbol_not_found_carries_symbol(self) -> None:
        exc = AlpacaSymbolNotFoundError("not found", symbol="FAKE")
        assert exc.symbol == "FAKE"

    def test_clock_drift_carries_drift_seconds(self) -> None:
        exc = AlpacaClockDriftError("drift too high", drift_seconds=7.3)
        assert exc.drift_seconds == 7.3

    def test_order_errors_inherit_from_alpaca_order_error(self) -> None:
        for cls in (AlpacaOrderRejectedError, AlpacaDuplicateOrderError, AlpacaInsufficientFundsError):
            assert issubclass(cls, AlpacaOrderError)
            assert issubclass(cls, BrokerError)

    def test_order_error_carries_alpaca_message(self) -> None:
        exc = AlpacaOrderRejectedError("rejected", alpaca_message="PDT protection")
        assert exc.alpaca_message == "PDT protection"
