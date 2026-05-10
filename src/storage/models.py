from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp for use as SQLAlchemy column defaults."""
    return datetime.now(UTC)


# ── Trading core ──────────────────────────────────────────────────────────────


class OrderAttempt(Base):
    """Every order submission attempt, persisted BEFORE calling Alpaca.

    Lifecycle: pending → submitted → accepted → filled | partially_filled
                                              → rejected | canceled | failed_to_submit

    ``client_order_id`` is the idempotency key. If the bot crashes between
    persisting this row and receiving Alpaca's confirmation, the reconciler
    can query Alpaca by client_order_id on restart.

    ``trades`` is updated from the trade_updates WebSocket stream (the source
    of truth for fills), not from submit_order's return value.
    """

    __tablename__ = "order_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    alpaca_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)  # populated after Alpaca accepts
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)          # "buy" | "sell"
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    order_type: Mapped[str] = mapped_column(String(15), nullable=False)   # "market" | "limit" | "stop_limit"
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    rule_trigger: Mapped[str | None] = mapped_column(String(500), nullable=True)
    strategy_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="order_attempt", lazy="select")

    __table_args__ = (
        Index("ix_order_attempts_symbol_submitted_at", "symbol", "submitted_at"),
        Index("ix_order_attempts_alpaca_order_id", "alpaca_order_id"),
    )


class Trade(Base):
    """A confirmed fill (full or partial) received from the trade_updates stream.

    One OrderAttempt may produce 0 (pending/rejected), 1 (full fill), or N
    (partial fills) Trade rows. P&L is computed here at fill time.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Idempotency: submitted to Alpaca as client_order_id
    client_order_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    order_attempt_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("order_attempts.id"), nullable=True
    )
    order_attempt: Mapped["OrderAttempt | None"] = relationship("OrderAttempt", back_populates="trades")
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)          # "buy" | "sell"
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    filled_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_type: Mapped[str] = mapped_column(String(15), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)       # pending/filled/cancelled/rejected
    rule_trigger: Mapped[str | None] = mapped_column(String(500), nullable=True)
    strategy_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)       # realized P&L when a sell closes a position

    __table_args__ = (
        Index("ix_trades_symbol_created_at", "symbol", "created_at"),
        Index("ix_trades_order_attempt_id", "order_attempt_id"),
    )


# ── Strategy management ───────────────────────────────────────────────────────


class StrategyVersion(Base):
    """Immutable history of every strategy ever activated.

    Only one row may have is_active=True at a time, enforced by the partial
    unique index below. Never delete or overwrite past versions — use
    is_active=False + deactivated_at to preserve rollback capability.
    """

    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_input: Mapped[str] = mapped_column(String, nullable=False)
    parsed_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Exactly one active strategy at a time (SQLite partial unique index)
    __table_args__ = (
        Index(
            "ix_strategy_versions_one_active",
            "is_active",
            unique=True,
            sqlite_where=text("is_active = 1"),
        ),
    )


# ── Risk management ───────────────────────────────────────────────────────────


class RiskOverride(Base):
    """A single risk parameter override set via /setlimit Telegram command.

    The active override for a given param_name is the most recent row with
    is_active=True for that param_name.  Historical rows are kept for audit.

    Overrides are always MORE RESTRICTIVE than the strategy defaults.
    The risk_manager enforces this: it takes min(strategy_value, override_value)
    and logs a warning if an override would have been less restrictive.
    """

    __tablename__ = "risk_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    param_name: Mapped[str] = mapped_column(String(50), nullable=False)
    # "max_position_pct" | "max_total_exposure_pct" | "stop_loss_pct" | "max_concurrent_positions"
    param_value: Mapped[float] = mapped_column(Float, nullable=False)
    set_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    set_by: Mapped[str] = mapped_column(String(100), nullable=False)   # "telegram:<chat_id>" | "system"
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_risk_overrides_param_active", "param_name", "is_active"),
    )


# ── Logging / queuing ─────────────────────────────────────────────────────────


class AlertLog(Base):
    """Audit trail of every Telegram message sent and every notable event."""

    __tablename__ = "alert_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)   # trade/status/error/daily_summary/risk_check
    message: Mapped[str] = mapped_column(String, nullable=False)
    telegram_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class PendingStrategy(Base):
    """A strategy parsed by the LLM awaiting /confirm or /cancel.

    Has a TTL (expires_at). After expiry the bot rejects /confirm and asks
    the user to re-send /strategy.
    """

    __tablename__ = "pending_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_input: Mapped[str] = mapped_column(String, nullable=False)
    parsed_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    chat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PendingAction(Base):
    """Out-of-hours actions (e.g. close_all) queued for execution at next open."""

    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)         # "close_all"
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending/executed/cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    action_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
