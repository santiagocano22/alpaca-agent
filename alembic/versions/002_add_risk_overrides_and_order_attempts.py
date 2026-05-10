"""add risk_overrides, order_attempts; add order_attempt_id to trades

Revision ID: 002
Revises: 001
Create Date: 2026-05-09 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── order_attempts (must exist before trades references it) ───────────────
    op.create_table(
        "order_attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_order_id", sa.String(length=128), nullable=False),
        sa.Column("alpaca_order_id", sa.String(length=50), nullable=True),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("order_type", sa.String(length=15), nullable=False),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("rule_trigger", sa.String(length=500), nullable=True),
        sa.Column("strategy_version_id", sa.Integer(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_order_id"),
    )
    op.create_index(
        "ix_order_attempts_symbol_submitted_at",
        "order_attempts",
        ["symbol", "submitted_at"],
    )
    op.create_index(
        "ix_order_attempts_alpaca_order_id",
        "order_attempts",
        ["alpaca_order_id"],
    )

    # ── trades: add order_attempt_id column + index ──────────────────────────
    op.add_column(
        "trades",
        sa.Column("order_attempt_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_trades_order_attempt_id",
        "trades",
        ["order_attempt_id"],
    )

    # ── risk_overrides ────────────────────────────────────────────────────────
    op.create_table(
        "risk_overrides",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("param_name", sa.String(length=50), nullable=False),
        sa.Column("param_value", sa.Float(), nullable=False),
        sa.Column("set_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("set_by", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_risk_overrides_param_active",
        "risk_overrides",
        ["param_name", "is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_risk_overrides_param_active", table_name="risk_overrides")
    op.drop_table("risk_overrides")
    op.drop_index("ix_trades_order_attempt_id", table_name="trades")
    op.drop_column("trades", "order_attempt_id")
    op.drop_index("ix_order_attempts_alpaca_order_id", table_name="order_attempts")
    op.drop_index("ix_order_attempts_symbol_submitted_at", table_name="order_attempts")
    op.drop_table("order_attempts")
