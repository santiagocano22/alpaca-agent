"""Risk manager: validates OrderIntent against 12 ordered checks before submission.

Check order (cheap-first for fast short-circuit):
  1.  BOT_PAUSED           — in-memory flag, free
  2.  INVALID_QUANTITY     — qty > 0
  3.  MARKET_CLOSED        — market_hours (skipped if is_closing=True)
  4.  SYMBOL_NOT_TRADEABLE — asset_cache lookup (TTL 1h, not direct Alpaca)
  5.  CLOCK_DRIFT          — cached 2 min; 2-s timeout → assume drift=0, log warning
  6.  DUPLICATE_ORDER      — DB query: same symbol+side within 60 s
  7.  PRICE_OUT_OF_RANGE   — limit/stop_limit only: limit_price in [50%, 200%] of ref
  8.  INVALID_STOP_LOSS    — non-closing only: effective sl_pct in [0.1, 50]
  9.  INSUFFICIENT_BUYING_POWER — order_value × 1.01 ≤ account.buying_power
                                   (skipped for sell-closing-long)
  10. POSITION_SIZE_EXCEEDED    — order_value ≤ max_position_pct × portfolio_value
  11. TOTAL_EXPOSURE_EXCEEDED   — post-order exposure ≤ max_total_exposure_pct × portfolio_value
  12. MAX_POSITIONS_REACHED     — post-order position count ≤ max_concurrent_positions

Checks 9–12 share a SINGLE get_account() + get_positions() fetch (done once, after 1–8 pass).

Exemptions for is_closing=True: checks 3, 8, 10, 11, 12.
Exemption for sell-closing-long: check 9.

RiskOverrides apply only when MORE restrictive than the strategy value.
Less-restrictive overrides are ignored and logged at WARNING level.

Every ValidationResult (approved or rejected) is persisted to AlertLog as audit trail
via db_session.flush() — the caller owns the transaction / commit.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.broker.asset_cache import AssetCache
from src.broker.exceptions import AlpacaSymbolNotFoundError
from src.broker.schemas import (
    AccountSnapshot,
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
from src.strategy.schema import Session as StrategySession, Strategy
from src.utils.market_hours import CalendarCache, is_extended_hours_open, is_market_open

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_CLOCK_DRIFT_SECONDS: float = 60.0
_CLOCK_CACHE_TTL_SECONDS: float = 120.0
_CLOCK_FETCH_TIMEOUT_SECONDS: float = 2.0
_DUPLICATE_WINDOW_SECONDS: int = 60
_BUYING_POWER_BUFFER_PCT: float = 1.0          # 1 % safety margin on top of order value
_PRICE_RANGE_LO: float = 0.50                  # limit must be ≥ 50 % of ref price
_PRICE_RANGE_HI: float = 2.00                  # limit must be ≤ 200 % of ref price
_STOP_LOSS_MIN: float = 0.1
_STOP_LOSS_MAX: float = 50.0

# Statuses that mean the previous attempt definitively failed (not inflight)
_TERMINAL_FAILURE_STATUSES: frozenset[str] = frozenset(
    {"rejected", "canceled", "failed_to_submit"}
)

# ── Module-level clock cache ──────────────────────────────────────────────────
# Tuple of (ClockSnapshot, cached_at_utc).  Reset between tests via _reset_clock_cache().
_clock_cache: tuple[ClockSnapshot, datetime] | None = None


def _reset_clock_cache() -> None:
    """Invalidate the module-level clock cache. Call in test teardown."""
    global _clock_cache
    _clock_cache = None


# ── Override resolution ───────────────────────────────────────────────────────


def _resolve_overrides(strategy: Strategy, overrides: RiskOverrides | None) -> dict:
    """Return effective risk limits, applying overrides only when more restrictive.

    "More restrictive" = smaller value for all parameters (lower position cap,
    tighter stop loss, fewer concurrent positions).  Less-restrictive overrides
    are logged as warnings and discarded.
    """
    ps = strategy.position_sizing
    eff = {
        "max_position_pct": ps.max_position_pct,
        "max_total_exposure_pct": ps.max_total_exposure_pct,
        "max_concurrent_positions": float(ps.max_concurrent_positions),
        "stop_loss_pct": strategy.exit_rules.stop_loss_pct,
    }
    if overrides is None:
        return eff

    def _apply(key: str, override_val: float, strategy_val: float) -> float:
        if override_val > strategy_val:
            logger.warning(
                "Override {}={} is LESS restrictive than strategy value={}; ignoring",
                key,
                override_val,
                strategy_val,
            )
            return strategy_val
        return override_val

    if overrides.max_position_pct is not None:
        eff["max_position_pct"] = _apply(
            "max_position_pct", overrides.max_position_pct, ps.max_position_pct
        )
    if overrides.max_total_exposure_pct is not None:
        eff["max_total_exposure_pct"] = _apply(
            "max_total_exposure_pct",
            overrides.max_total_exposure_pct,
            ps.max_total_exposure_pct,
        )
    if overrides.max_concurrent_positions is not None:
        eff["max_concurrent_positions"] = _apply(
            "max_concurrent_positions",
            float(overrides.max_concurrent_positions),
            float(ps.max_concurrent_positions),
        )
    if overrides.stop_loss_pct is not None:
        eff["stop_loss_pct"] = _apply(
            "stop_loss_pct", overrides.stop_loss_pct, strategy.exit_rules.stop_loss_pct
        )

    return eff


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _get_clock_cached(alpaca) -> ClockSnapshot | None:
    """Fetch clock snapshot with 2-min TTL and 2-s timeout.

    On timeout or error: returns None (caller treats as drift=0, does not reject).
    """
    global _clock_cache
    now = datetime.now(UTC)
    if _clock_cache is not None:
        snap, cached_at = _clock_cache
        if (now - cached_at).total_seconds() < _CLOCK_CACHE_TTL_SECONDS:
            return snap
    try:
        snap = await asyncio.wait_for(
            alpaca.get_clock(), timeout=_CLOCK_FETCH_TIMEOUT_SECONDS
        )
        _clock_cache = (snap, now)
        return snap
    except asyncio.TimeoutError:
        logger.warning(
            "Clock check timed out after {}s; assuming drift=0 (safe, will not block orders)",
            _CLOCK_FETCH_TIMEOUT_SECONDS,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Clock check error: {}; assuming drift=0 (safe)", exc)
        return None


def _estimate_order_value(
    intent: OrderIntent, last_known_price: float | None
) -> float | None:
    """Best-effort order value in USD. Returns None if no price info is available."""
    price = intent.estimated_price if intent.estimated_price is not None else last_known_price
    if price is None or price <= 0:
        return None
    return price * intent.qty


async def _persist_log(session: AsyncSession, result: ValidationResult) -> None:
    """Write one AlertLog row to the current session (caller commits)."""
    entry = AlertLog(
        type="risk_check",
        message=(
            f"{'APPROVED' if result.approved else 'REJECTED'} "
            f"[{result.symbol}] {result.code.value}: {result.reason}"
        ),
        telegram_sent=False,
    )
    session.add(entry)
    await session.flush()


# ── Public API ────────────────────────────────────────────────────────────────


async def validate_order(
    intent: OrderIntent,
    strategy: Strategy,
    overrides: RiskOverrides | None,
    alpaca,
    asset_cache: AssetCache,
    db_session: AsyncSession,
    bot_paused: bool,
    last_known_price: float | None = None,
    calendar: CalendarCache | None = None,
    now: datetime | None = None,
) -> ValidationResult:
    """Validate an OrderIntent through all 12 risk checks.

    Returns a ValidationResult (approved or rejected) and persists it to AlertLog.
    The function never raises — all errors are converted to rejections.

    Args:
        intent:           The proposed order.
        strategy:         Active strategy (contains position sizing and exit rules).
        overrides:        Active /setlimit overrides (may be None).
        alpaca:           AlpacaClient for account/positions/clock lookups.
        asset_cache:      AssetCache for symbol tradability checks.
        db_session:       Open AsyncSession; flush() is called but not commit().
        bot_paused:       Global pause flag (set by /pause Telegram command).
        last_known_price: Last price seen by the engine (for checks 7, 9, 10, 11).
        calendar:         CalendarCache for check 3; if None, check 3 is skipped.
        now:              Reference time (defaults to datetime.now(UTC)).
    """
    _now = now if now is not None else datetime.now(UTC)
    eff = _resolve_overrides(strategy, overrides)
    order_value = _estimate_order_value(intent, last_known_price)

    # ── Local helpers (closures over the local variables above) ───────────────

    async def _reject(code: RiskCheckCode, reason: str) -> ValidationResult:
        result = ValidationResult.reject(
            symbol=intent.symbol,
            code=code,
            reason=reason,
            estimated_order_value_usd=order_value,
        )
        await _persist_log(db_session, result)
        return result

    async def _approve() -> ValidationResult:
        result = ValidationResult.approve(
            symbol=intent.symbol,
            estimated_order_value_usd=order_value,
        )
        await _persist_log(db_session, result)
        return result

    # ── 1. BOT_PAUSED ─────────────────────────────────────────────────────────
    if bot_paused:
        logger.info("Check 1 FAIL BOT_PAUSED: {}", intent.symbol)
        return await _reject(RiskCheckCode.BOT_PAUSED, "Bot is paused; no new orders accepted")

    # ── 2. INVALID_QUANTITY ────────────────────────────────────────────────────
    if intent.qty <= 0:
        return await _reject(
            RiskCheckCode.INVALID_QUANTITY,
            f"qty must be positive, got {intent.qty}",
        )

    # ── 3. MARKET_CLOSED ──────────────────────────────────────────────────────
    # Closing orders bypass this check (allowed outside regular hours so they can
    # be queued as PendingAction and executed at next open).
    if not intent.is_closing and calendar is not None:
        sess = strategy.session
        if sess == StrategySession.CRYPTO_24_7:
            market_open = True
        elif sess == StrategySession.EXTENDED:
            market_open = is_extended_hours_open(_now, calendar)
        else:  # REGULAR
            market_open = is_market_open(_now, calendar)

        if not market_open:
            return await _reject(
                RiskCheckCode.MARKET_CLOSED,
                f"Market is closed for session={sess.value}",
            )

    # ── 4. SYMBOL_NOT_TRADEABLE ────────────────────────────────────────────────
    try:
        asset = await asset_cache.get(intent.symbol)
    except AlpacaSymbolNotFoundError:
        return await _reject(
            RiskCheckCode.SYMBOL_NOT_TRADEABLE,
            f"Symbol {intent.symbol} not found on Alpaca",
        )
    except Exception as exc:  # noqa: BLE001
        return await _reject(
            RiskCheckCode.SYMBOL_NOT_TRADEABLE,
            f"Failed to fetch asset info for {intent.symbol}: {exc}",
        )

    if not asset.tradeable or asset.status != "active":
        return await _reject(
            RiskCheckCode.SYMBOL_NOT_TRADEABLE,
            f"{intent.symbol} is not tradeable (tradeable={asset.tradeable}, status={asset.status!r})",
        )

    # ── 5. CLOCK_DRIFT ─────────────────────────────────────────────────────────
    clock = await _get_clock_cached(alpaca)
    if clock is not None:
        drift = abs(clock.drift_seconds)
        if drift > _MAX_CLOCK_DRIFT_SECONDS:
            return await _reject(
                RiskCheckCode.CLOCK_DRIFT,
                f"Local clock drift {drift:.1f}s exceeds max {_MAX_CLOCK_DRIFT_SECONDS}s",
            )

    # ── 6. DUPLICATE_ORDER ─────────────────────────────────────────────────────
    cutoff = _now - timedelta(seconds=_DUPLICATE_WINDOW_SECONDS)
    dup_stmt = (
        select(OrderAttempt)
        .where(
            OrderAttempt.symbol == intent.symbol.upper(),
            OrderAttempt.side == intent.side.value,
            OrderAttempt.submitted_at >= cutoff,
            OrderAttempt.status.notin_(list(_TERMINAL_FAILURE_STATUSES)),
        )
        .limit(1)
    )
    dup_result = await db_session.execute(dup_stmt)
    if dup_result.scalars().first() is not None:
        return await _reject(
            RiskCheckCode.DUPLICATE_ORDER,
            f"Duplicate order: {intent.symbol} {intent.side.value} already submitted within {_DUPLICATE_WINDOW_SECONDS}s",
        )

    # ── 7. PRICE_OUT_OF_RANGE ──────────────────────────────────────────────────
    if intent.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and intent.limit_price is not None:
        ref_price = last_known_price if last_known_price is not None else intent.estimated_price
        if ref_price is not None and ref_price > 0:
            lo = ref_price * _PRICE_RANGE_LO
            hi = ref_price * _PRICE_RANGE_HI
            if not (lo <= intent.limit_price <= hi):
                return await _reject(
                    RiskCheckCode.PRICE_OUT_OF_RANGE,
                    f"Limit price {intent.limit_price:.4f} is outside "
                    f"[{lo:.4f}, {hi:.4f}] (50%–200% of ref {ref_price:.4f})",
                )

    # ── 8. INVALID_STOP_LOSS ───────────────────────────────────────────────────
    if not intent.is_closing:
        sl = eff["stop_loss_pct"]
        if not (_STOP_LOSS_MIN <= sl <= _STOP_LOSS_MAX):
            return await _reject(
                RiskCheckCode.INVALID_STOP_LOSS,
                f"Effective stop_loss_pct {sl} is outside valid range "
                f"[{_STOP_LOSS_MIN}, {_STOP_LOSS_MAX}]",
            )

    # ── Fetch account + positions ONCE for checks 9–12 ────────────────────────
    account: AccountSnapshot = await alpaca.get_account()
    positions: list[PositionSnapshot] = await alpaca.get_positions()
    equity = account.portfolio_value

    # ── 9. INSUFFICIENT_BUYING_POWER ──────────────────────────────────────────
    # A sell that closes a long position returns shares and receives cash —
    # no buying power is consumed; skip the check.
    is_sell_closing = intent.side == OrderSide.SELL and intent.is_closing
    if not is_sell_closing and order_value is not None:
        required_bp = order_value * (1 + _BUYING_POWER_BUFFER_PCT / 100)
        if required_bp > account.buying_power:
            return await _reject(
                RiskCheckCode.INSUFFICIENT_BUYING_POWER,
                f"Need {required_bp:.2f} buying power (incl. {_BUYING_POWER_BUFFER_PCT}% buffer), "
                f"have {account.buying_power:.2f}",
            )

    # Checks 10–12 are exempt for closing orders.
    if intent.is_closing:
        return await _approve()

    # ── 10. POSITION_SIZE_EXCEEDED ─────────────────────────────────────────────
    if order_value is not None and equity > 0:
        max_pos_value = eff["max_position_pct"] / 100 * equity
        if order_value > max_pos_value:
            return await _reject(
                RiskCheckCode.POSITION_SIZE_EXCEEDED,
                f"Order value {order_value:.2f} > max position "
                f"{max_pos_value:.2f} ({eff['max_position_pct']}% × equity {equity:.2f})",
            )

    # ── 11. TOTAL_EXPOSURE_EXCEEDED ────────────────────────────────────────────
    if order_value is not None and equity > 0:
        current_exposure = sum(abs(p.market_value) for p in positions)
        post_exposure = current_exposure + order_value
        max_exposure = eff["max_total_exposure_pct"] / 100 * equity
        if post_exposure > max_exposure:
            return await _reject(
                RiskCheckCode.TOTAL_EXPOSURE_EXCEEDED,
                f"Post-order exposure {post_exposure:.2f} > max "
                f"{max_exposure:.2f} ({eff['max_total_exposure_pct']}% × equity {equity:.2f})",
            )

    # ── 12. MAX_POSITIONS_REACHED ──────────────────────────────────────────────
    existing_symbols = {p.symbol.upper() for p in positions}
    post_count = len(existing_symbols)
    if intent.symbol.upper() not in existing_symbols:
        post_count += 1
    max_pos = int(eff["max_concurrent_positions"])
    if post_count > max_pos:
        return await _reject(
            RiskCheckCode.MAX_POSITIONS_REACHED,
            f"Would have {post_count} open positions, max is {max_pos}",
        )

    return await _approve()
