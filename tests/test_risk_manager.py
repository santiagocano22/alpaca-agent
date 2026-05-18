"""Tests for src/broker/risk_manager.py.

Coverage:
  - One pass + one fail per each of the 12 risk checks (with correct code).
  - Override more restrictive → applied.
  - Override less restrictive → ignored + WARNING logged.
  - Closing order that would exceed max_concurrent_positions: approved (exempt).
  - Integration: valid order traverses all 12 checks; get_account and
    get_positions each called exactly ONCE (checks 9-12 share a single fetch).
  - Short-circuit: BOT_PAUSED → Alpaca never contacted (call_count == 0).
  - Sell closing long → no buying-power check.
  - is_closing + market closed → approved (exempt from check 3).
  - Every ValidationResult persisted to AlertLog.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from src.broker.exceptions import AlpacaSymbolNotFoundError
from src.broker.risk_manager import (
    _resolve_overrides,
    _reset_clock_cache,
    validate_order,
)
from src.broker.schemas import (
    AccountSnapshot,
    AssetInfo,
    ClockSnapshot,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionSnapshot,
    RiskCheckCode,
    RiskOverrides,
    ValidationResult,
)
from src.storage.models import AlertLog, OrderAttempt
from src.strategy.schema import (
    ComparisonOp,
    Condition,
    EodPolicy,
    ExitRules,
    Horizon,
    IndicatorRef,
    IndicatorType,
    PositionSizing,
    RuleGroup,
    Session as StrategySession,
    Strategy,
    Timeframe,
)

# ── Helpers / factories ───────────────────────────────────────────────────────

_NOW = datetime(2025, 1, 8, 14, 30, 0, tzinfo=UTC)  # Wed 8 Jan 2025, 09:30 ET


def _make_strategy(
    *,
    max_position_pct: float = 10.0,
    max_total_exposure_pct: float = 50.0,
    max_concurrent_positions: int = 4,
    stop_loss_pct: float = 2.0,
    session: StrategySession = StrategySession.REGULAR,
    horizon: Horizon = Horizon.INTRADAY,
    timeframe: Timeframe = Timeframe.M5,
) -> Strategy:
    entry = RuleGroup(
        logic="AND",
        conditions=[
            Condition(
                left=IndicatorRef(type=IndicatorType.RSI, params={"period": 14}),
                op=ComparisonOp.LT,
                right=30.0,
            )
        ],
    )
    return Strategy(
        name="test",
        universe=["AAPL"],
        timeframe=timeframe,
        session=session,
        horizon=horizon,
        eod_policy=EodPolicy.CLOSE_ALL,
        entry_rules=entry,
        exit_rules=ExitRules(stop_loss_pct=stop_loss_pct),
        position_sizing=PositionSizing(
            max_position_pct=max_position_pct,
            max_total_exposure_pct=max_total_exposure_pct,
            max_concurrent_positions=max_concurrent_positions,
        ),
    )


def _make_intent(
    *,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 1.0,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    stop_price: float | None = None,
    is_closing: bool = False,
    estimated_price: float | None = 150.0,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        is_closing=is_closing,
        estimated_price=estimated_price,
        client_order_id="test-order-id-001",
    )


def _make_asset(*, tradeable: bool = True, status: str = "active") -> AssetInfo:
    return AssetInfo(
        symbol="AAPL",
        name="Apple Inc.",
        tradeable=tradeable,
        fractionable=True,
        shortable=True,
        easy_to_borrow=True,
        status=status,
    )


def _make_account(
    *,
    buying_power: float = 100_000.0,
    portfolio_value: float = 100_000.0,
) -> AccountSnapshot:
    return AccountSnapshot(
        buying_power=buying_power,
        cash=50_000.0,
        portfolio_value=portfolio_value,
        pattern_day_trader=False,
        trading_blocked=False,
        account_blocked=False,
        snapshot_at=_NOW,
    )


def _make_clock(*, drift_seconds: float = 0.0, is_open: bool = True) -> ClockSnapshot:
    ts = datetime(2025, 1, 8, 14, 30, 0, tzinfo=UTC)
    return ClockSnapshot(
        alpaca_timestamp=ts,
        local_timestamp=ts + timedelta(seconds=drift_seconds),
        drift_seconds=drift_seconds,
        is_open=is_open,
    )


def _make_position(*, symbol: str = "GOOG", market_value: float = 5_000.0) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        qty=10.0,
        side="long",
        market_value=market_value,
        avg_entry_price=500.0,
        unrealized_pl=100.0,
        unrealized_plpc=0.02,
        current_price=510.0,
    )


def _make_alpaca(
    *,
    clock: ClockSnapshot | None = None,
    account: AccountSnapshot | None = None,
    positions: list[PositionSnapshot] | None = None,
) -> MagicMock:
    """Return a mock AlpacaClient with all async methods pre-configured."""
    m = MagicMock()
    m.get_clock = AsyncMock(return_value=clock or _make_clock())
    m.get_account = AsyncMock(return_value=account or _make_account())
    m.get_positions = AsyncMock(return_value=positions if positions is not None else [])
    return m


def _make_asset_cache(*, asset: AssetInfo | None = None, raises: Exception | None = None) -> MagicMock:
    m = MagicMock()
    if raises is not None:
        m.get = AsyncMock(side_effect=raises)
    else:
        m.get = AsyncMock(return_value=asset or _make_asset())
    return m


# CalendarCache that considers _NOW as market-open (Wed 8 Jan 2025, 09:30 ET → 14:30 UTC)
def _open_calendar() -> dict:
    from datetime import date, time
    from src.utils.market_hours import MarketDay
    return {
        date(2025, 1, 8): MarketDay(date(2025, 1, 8), time(9, 30), time(16, 0)),
    }


def _closed_calendar() -> dict:
    # Return empty dict → no trading day found → market is closed
    return {}


# ── Autouse fixture: reset clock cache between every test ─────────────────────


@pytest.fixture(autouse=True)
def reset_clock_cache():
    _reset_clock_cache()
    yield
    _reset_clock_cache()


# ── Check 1: BOT_PAUSED ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check1_bot_paused_rejects(async_session):
    alpaca = _make_alpaca()
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=True,
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.BOT_PAUSED


@pytest.mark.asyncio
async def test_check1_bot_not_paused_does_not_short_circuit(async_session):
    """When bot is NOT paused, execution continues past check 1."""
    alpaca = _make_alpaca()
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.BOT_PAUSED


@pytest.mark.asyncio
async def test_check1_bot_paused_alpaca_not_called(async_session):
    """BOT_PAUSED short-circuits before any Alpaca call."""
    alpaca = _make_alpaca()
    await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=True,
        now=_NOW,
    )
    alpaca.get_clock.assert_not_called()
    alpaca.get_account.assert_not_called()
    alpaca.get_positions.assert_not_called()


# ── Check 2: INVALID_QUANTITY ─────────────────────────────────────────────────
# Note: OrderIntent validator rejects qty<=0 at construction; the check in
# risk_manager is an extra safety net.  We bypass Pydantic by patching qty.


@pytest.mark.asyncio
async def test_check2_invalid_quantity_zero(async_session):
    """qty=0 must be rejected with INVALID_QUANTITY."""
    intent = _make_intent()
    # Bypass pydantic guard by direct object mutation
    object.__setattr__(intent, "qty", 0.0)
    result = await validate_order(
        intent=intent,
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.INVALID_QUANTITY


@pytest.mark.asyncio
async def test_check2_valid_quantity_passes(async_session):
    alpaca = _make_alpaca()
    result = await validate_order(
        intent=_make_intent(qty=10.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.INVALID_QUANTITY


# ── Check 3: MARKET_CLOSED ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check3_market_closed_rejects(async_session):
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(session=StrategySession.REGULAR),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_closed_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.MARKET_CLOSED


@pytest.mark.asyncio
async def test_check3_market_open_passes(async_session):
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(session=StrategySession.REGULAR),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.MARKET_CLOSED


@pytest.mark.asyncio
async def test_check3_closing_order_exempt_when_market_closed(async_session):
    """is_closing=True must be approved even if market is closed."""
    result = await validate_order(
        intent=_make_intent(is_closing=True, side=OrderSide.SELL),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_closed_calendar(),
        now=_NOW,
    )
    # Should NOT be rejected for MARKET_CLOSED
    assert result.code != RiskCheckCode.MARKET_CLOSED


@pytest.mark.asyncio
async def test_check3_crypto_always_open(async_session):
    """CRYPTO_24_7 session ignores calendar."""
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(
            session=StrategySession.CRYPTO_24_7,
            horizon=Horizon.INTRADAY,
            timeframe=Timeframe.M5,
        ),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_closed_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.MARKET_CLOSED


# ── Check 4: SYMBOL_NOT_TRADEABLE ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check4_symbol_not_found_rejects(async_session):
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(raises=AlpacaSymbolNotFoundError("AAPL not found", symbol="AAPL")),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.SYMBOL_NOT_TRADEABLE


@pytest.mark.asyncio
async def test_check4_symbol_not_tradeable_flag(async_session):
    asset = _make_asset(tradeable=False)
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(asset=asset),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.SYMBOL_NOT_TRADEABLE


@pytest.mark.asyncio
async def test_check4_tradeable_symbol_passes(async_session):
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(asset=_make_asset(tradeable=True, status="active")),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.SYMBOL_NOT_TRADEABLE


# ── Check 5: CLOCK_DRIFT ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check5_large_drift_rejects(async_session):
    clock = _make_clock(drift_seconds=90.0)  # > 60s limit
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(clock=clock),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.CLOCK_DRIFT


@pytest.mark.asyncio
async def test_check5_small_drift_passes(async_session):
    clock = _make_clock(drift_seconds=5.0)
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(clock=clock),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.CLOCK_DRIFT


@pytest.mark.asyncio
async def test_check5_clock_timeout_does_not_reject(async_session):
    """When the clock call times out, drift is assumed 0 and check passes."""
    alpaca = _make_alpaca()
    alpaca.get_clock = AsyncMock(side_effect=asyncio.TimeoutError())
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.CLOCK_DRIFT


# ── Check 6: DUPLICATE_ORDER ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check6_duplicate_within_window_rejects(async_session):
    # Insert a recent non-terminal OrderAttempt
    recent = OrderAttempt(
        client_order_id="prev-order-001",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        order_type="market",
        status="pending",
        submitted_at=_NOW - timedelta(seconds=30),
    )
    async_session.add(recent)
    await async_session.flush()

    result = await validate_order(
        intent=_make_intent(symbol="AAPL", side=OrderSide.BUY),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.DUPLICATE_ORDER


@pytest.mark.asyncio
async def test_check6_terminal_status_not_duplicate(async_session):
    """A terminal (rejected/canceled) prior attempt should not trigger dup check."""
    old = OrderAttempt(
        client_order_id="prev-order-002",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        order_type="market",
        status="rejected",
        submitted_at=_NOW - timedelta(seconds=30),
    )
    async_session.add(old)
    await async_session.flush()

    result = await validate_order(
        intent=_make_intent(symbol="AAPL", side=OrderSide.BUY),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.DUPLICATE_ORDER


@pytest.mark.asyncio
async def test_check6_outside_window_not_duplicate(async_session):
    """An attempt older than 60s is outside the duplicate window."""
    old = OrderAttempt(
        client_order_id="prev-order-003",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        order_type="market",
        status="pending",
        submitted_at=_NOW - timedelta(seconds=120),
    )
    async_session.add(old)
    await async_session.flush()

    result = await validate_order(
        intent=_make_intent(symbol="AAPL", side=OrderSide.BUY),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.DUPLICATE_ORDER


# ── Check 7: PRICE_OUT_OF_RANGE ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check7_limit_price_too_low_rejects(async_session):
    # ref=100, limit=40 → 40 < 50% of 100
    intent = _make_intent(
        order_type=OrderType.LIMIT,
        limit_price=40.0,
        estimated_price=100.0,
    )
    result = await validate_order(
        intent=intent,
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=100.0,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.PRICE_OUT_OF_RANGE


@pytest.mark.asyncio
async def test_check7_limit_price_in_range_passes(async_session):
    # ref=100, limit=90 → within [50, 200]
    intent = _make_intent(
        order_type=OrderType.LIMIT,
        limit_price=90.0,
        estimated_price=100.0,
    )
    result = await validate_order(
        intent=intent,
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=100.0,
    )
    assert result.code != RiskCheckCode.PRICE_OUT_OF_RANGE


@pytest.mark.asyncio
async def test_check7_market_order_skips_price_check(async_session):
    """MARKET orders have no limit_price; check 7 must be skipped."""
    result = await validate_order(
        intent=_make_intent(order_type=OrderType.MARKET),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.PRICE_OUT_OF_RANGE


# ── Check 8: INVALID_STOP_LOSS ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check8_stop_loss_above_max_rejects(async_session):
    """stop_loss_pct > 50 is invalid (ExitRules caps at 50, but overrides could
    lower the effective value below minimum via a weird edge-case; here we use
    the exact boundary by crafting a strategy with stop_loss=2 and an override
    that would set it out of range — but since override must be ≤ strategy, the
    easiest test is a strategy with a valid stop_loss that the override tightens
    below the effective minimum (0.1 is already min, so let's test a strategy
    where the bounds logic would reject).

    Actually: the ExitRules pydantic validator accepts stop_loss_pct in (0.1, 50].
    The only way check 8 fires is via _resolve_overrides reducing the effective
    stop_loss_pct below _STOP_LOSS_MIN.  But overrides can only be MORE restrictive
    (smaller = lower stop loss = more restrictive).  So a stop_loss_pct override
    of 0.05 on a strategy with 2.0 is valid (it's more restrictive) and yields
    an effective SL of 0.05, which is < _STOP_LOSS_MIN (0.1) → check 8 fires.
    """
    overrides = RiskOverrides(stop_loss_pct=0.05)  # below 0.1 floor
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(stop_loss_pct=2.0),
        overrides=overrides,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.INVALID_STOP_LOSS


@pytest.mark.asyncio
async def test_check8_valid_stop_loss_passes(async_session):
    result = await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(stop_loss_pct=2.0),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.INVALID_STOP_LOSS


@pytest.mark.asyncio
async def test_check8_closing_order_exempt(async_session):
    """Closing orders bypass the stop-loss check entirely."""
    overrides = RiskOverrides(stop_loss_pct=0.05)
    result = await validate_order(
        intent=_make_intent(is_closing=True, side=OrderSide.SELL),
        strategy=_make_strategy(stop_loss_pct=2.0),
        overrides=overrides,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
    )
    assert result.code != RiskCheckCode.INVALID_STOP_LOSS


# ── Check 9: INSUFFICIENT_BUYING_POWER ───────────────────────────────────────


@pytest.mark.asyncio
async def test_check9_insufficient_buying_power_rejects(async_session):
    # order_value = 1 * 150 = 150, required = 150 * 1.01 = 151.5, bp = 100 → fail
    account = _make_account(buying_power=100.0, portfolio_value=100_000.0)
    result = await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.INSUFFICIENT_BUYING_POWER


@pytest.mark.asyncio
async def test_check9_sufficient_buying_power_passes(async_session):
    account = _make_account(buying_power=10_000.0, portfolio_value=100_000.0)
    result = await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.INSUFFICIENT_BUYING_POWER


@pytest.mark.asyncio
async def test_check9_sell_closing_long_exempt(async_session):
    """Sell that closes a long position must skip the buying-power check."""
    account = _make_account(buying_power=0.0)
    result = await validate_order(
        intent=_make_intent(side=OrderSide.SELL, is_closing=True),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.INSUFFICIENT_BUYING_POWER


# ── Check 10: POSITION_SIZE_EXCEEDED ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_check10_position_size_exceeded_rejects(async_session):
    # equity=10_000, max_position_pct=5% → max=500
    # order_value = 100 * 10 = 1_000 > 500 → fail
    account = _make_account(buying_power=100_000.0, portfolio_value=10_000.0)
    result = await validate_order(
        intent=_make_intent(qty=100.0, estimated_price=10.0),
        strategy=_make_strategy(
            max_position_pct=5.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=4,
        ),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=10.0,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.POSITION_SIZE_EXCEEDED


@pytest.mark.asyncio
async def test_check10_position_size_within_limit_passes(async_session):
    # equity=100_000, max_position_pct=10% → max=10_000
    # order_value = 1 * 150 = 150 ≤ 10_000 → pass
    account = _make_account(buying_power=100_000.0, portfolio_value=100_000.0)
    result = await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(max_position_pct=10.0),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.POSITION_SIZE_EXCEEDED


@pytest.mark.asyncio
async def test_check10_closing_order_exempt(async_session):
    """Closing orders are exempt from check 10."""
    account = _make_account(buying_power=100_000.0, portfolio_value=10_000.0)
    result = await validate_order(
        intent=_make_intent(qty=100.0, estimated_price=10.0, is_closing=True, side=OrderSide.SELL),
        strategy=_make_strategy(
            max_position_pct=5.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=4,
        ),
        overrides=None,
        alpaca=_make_alpaca(account=account),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=10.0,
    )
    assert result.code != RiskCheckCode.POSITION_SIZE_EXCEEDED


# ── Check 11: TOTAL_EXPOSURE_EXCEEDED ────────────────────────────────────────


@pytest.mark.asyncio
async def test_check11_total_exposure_exceeded_rejects(async_session):
    # equity=10_000, max_total_exposure=50% → max_exposure=5_000
    # existing_exposure=4_900, order_value=200 → post=5_100 > 5_000 → fail
    # PositionSizing coherence: 10% × 4 = 40 ≤ 50 ✓
    # order_value 200 ≤ max_position 1_000 ✓ (check 10 passes)
    account = _make_account(buying_power=100_000.0, portfolio_value=10_000.0)
    positions = [_make_position(market_value=4_900.0)]
    result = await validate_order(
        intent=_make_intent(qty=20.0, estimated_price=10.0),
        strategy=_make_strategy(
            max_position_pct=10.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=4,
        ),
        overrides=None,
        alpaca=_make_alpaca(account=account, positions=positions),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=10.0,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.TOTAL_EXPOSURE_EXCEEDED


@pytest.mark.asyncio
async def test_check11_total_exposure_within_limit_passes(async_session):
    # equity=100_000, max_total=50% → max=50_000
    # existing=5_000, order_value=150 → post=5_150 ≤ 50_000 → pass
    account = _make_account(buying_power=100_000.0, portfolio_value=100_000.0)
    positions = [_make_position(market_value=5_000.0)]
    result = await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(account=account, positions=positions),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.TOTAL_EXPOSURE_EXCEEDED


# ── Check 12: MAX_POSITIONS_REACHED ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_check12_max_positions_reached_rejects(async_session):
    # max_concurrent=2, currently 2 positions (GOOG, MSFT), new AAPL → 3 > 2 → fail
    positions = [
        _make_position(symbol="GOOG", market_value=5_000.0),
        _make_position(symbol="MSFT", market_value=5_000.0),
    ]
    result = await validate_order(
        intent=_make_intent(symbol="AAPL"),
        strategy=_make_strategy(
            max_position_pct=10.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=2,
        ),
        overrides=None,
        alpaca=_make_alpaca(positions=positions),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert not result.approved
    assert result.code == RiskCheckCode.MAX_POSITIONS_REACHED


@pytest.mark.asyncio
async def test_check12_increasing_existing_position_not_counted(async_session):
    """Adding more to a symbol already in positions doesn't increase count."""
    # 2 positions (GOOG, AAPL), buying more AAPL → count stays 2 ≤ max 2 → pass
    positions = [
        _make_position(symbol="GOOG", market_value=5_000.0),
        _make_position(symbol="AAPL", market_value=5_000.0),
    ]
    result = await validate_order(
        intent=_make_intent(symbol="AAPL"),
        strategy=_make_strategy(
            max_position_pct=10.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=2,
        ),
        overrides=None,
        alpaca=_make_alpaca(positions=positions),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.MAX_POSITIONS_REACHED


@pytest.mark.asyncio
async def test_check12_closing_order_exempt(async_session):
    """is_closing=True exempts check 12 (SELL to close position)."""
    # 2 positions, max=2, closing AAPL → approved
    positions = [
        _make_position(symbol="GOOG", market_value=5_000.0),
        _make_position(symbol="AAPL", market_value=5_000.0),
    ]
    result = await validate_order(
        intent=_make_intent(symbol="TSLA", is_closing=True, side=OrderSide.SELL),
        strategy=_make_strategy(
            max_position_pct=10.0,
            max_total_exposure_pct=50.0,
            max_concurrent_positions=2,
        ),
        overrides=None,
        alpaca=_make_alpaca(positions=positions),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.code != RiskCheckCode.MAX_POSITIONS_REACHED


# ── Override tests ────────────────────────────────────────────────────────────


def test_resolve_overrides_more_restrictive_applied():
    """A smaller override value (more restrictive) must be used."""
    strategy = _make_strategy(
        max_position_pct=10.0,
        max_total_exposure_pct=50.0,
        max_concurrent_positions=4,
        stop_loss_pct=3.0,
    )
    overrides = RiskOverrides(
        max_position_pct=5.0,          # 5 < 10 → more restrictive → applied
        max_total_exposure_pct=30.0,   # 30 < 50 → applied
        max_concurrent_positions=2,    # 2 < 4 → applied
        stop_loss_pct=1.5,             # 1.5 < 3.0 → applied
    )
    eff = _resolve_overrides(strategy, overrides)
    assert eff["max_position_pct"] == 5.0
    assert eff["max_total_exposure_pct"] == 30.0
    assert eff["max_concurrent_positions"] == 2.0
    assert eff["stop_loss_pct"] == 1.5


def test_resolve_overrides_less_restrictive_ignored():
    """A larger override value (less restrictive) must be discarded + warn."""
    strategy = _make_strategy(
        max_position_pct=5.0,
        max_total_exposure_pct=30.0,
        max_concurrent_positions=2,
        stop_loss_pct=2.0,
    )
    overrides = RiskOverrides(
        max_position_pct=20.0,          # 20 > 5 → less restrictive → ignored
        max_total_exposure_pct=None,
        max_concurrent_positions=None,
        stop_loss_pct=None,
    )
    # loguru doesn't propagate to Python's logging; patch at the call site.
    with patch("src.broker.risk_manager.logger") as mock_logger:
        eff = _resolve_overrides(strategy, overrides)

    assert eff["max_position_pct"] == 5.0  # strategy value kept
    # Verify that logger.warning was called with a message containing "LESS restrictive"
    warning_calls = mock_logger.warning.call_args_list
    assert any(
        "LESS restrictive" in str(args) or "less restrictive" in str(args).lower()
        for args in warning_calls
    ), f"Expected warning about less-restrictive override; got calls: {warning_calls}"


def test_resolve_overrides_none_returns_strategy_values():
    strategy = _make_strategy(max_position_pct=10.0, max_total_exposure_pct=50.0)
    eff = _resolve_overrides(strategy, None)
    assert eff["max_position_pct"] == 10.0
    assert eff["max_total_exposure_pct"] == 50.0


# ── Integration: valid order traverses all 12 checks ─────────────────────────


@pytest.mark.asyncio
async def test_integration_valid_order_approved(async_session):
    """A fully valid order must be approved (code=APPROVED)."""
    account = _make_account(buying_power=100_000.0, portfolio_value=100_000.0)
    result = await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(account=account, positions=[]),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert result.approved
    assert result.code == RiskCheckCode.APPROVED


@pytest.mark.asyncio
async def test_integration_single_account_and_positions_fetch(async_session):
    """get_account and get_positions must each be called exactly ONCE."""
    alpaca = _make_alpaca(
        account=_make_account(buying_power=100_000.0, portfolio_value=100_000.0),
        positions=[],
    )
    await validate_order(
        intent=_make_intent(qty=1.0, estimated_price=150.0),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    assert alpaca.get_account.call_count == 1
    assert alpaca.get_positions.call_count == 1


@pytest.mark.asyncio
async def test_integration_bot_paused_zero_alpaca_calls(async_session):
    """BOT_PAUSED must short-circuit before ANY Alpaca I/O."""
    alpaca = _make_alpaca()
    await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=alpaca,
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=True,
        now=_NOW,
    )
    assert alpaca.get_clock.call_count == 0
    assert alpaca.get_account.call_count == 0
    assert alpaca.get_positions.call_count == 0


# ── AlertLog persistence ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_result_persisted_to_alert_log(async_session):
    """Each call to validate_order (approved or rejected) must write one AlertLog row."""
    # First call: rejected (BOT_PAUSED)
    await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=True,
        now=_NOW,
    )
    # Second call: approved
    await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(
            account=_make_account(buying_power=100_000.0, portfolio_value=100_000.0),
            positions=[],
        ),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=False,
        calendar=_open_calendar(),
        now=_NOW,
        last_known_price=150.0,
    )
    rows = (await async_session.execute(select(AlertLog))).scalars().all()
    assert len(rows) == 2
    types = {r.type for r in rows}
    assert types == {"risk_check"}


@pytest.mark.asyncio
async def test_rejected_alert_log_contains_code(async_session):
    """Rejected result's AlertLog message must include the rejection code."""
    await validate_order(
        intent=_make_intent(),
        strategy=_make_strategy(),
        overrides=None,
        alpaca=_make_alpaca(),
        asset_cache=_make_asset_cache(),
        db_session=async_session,
        bot_paused=True,
        now=_NOW,
    )
    rows = (await async_session.execute(select(AlertLog))).scalars().all()
    assert len(rows) == 1
    assert "BOT_PAUSED" in rows[0].message
    assert "REJECTED" in rows[0].message
