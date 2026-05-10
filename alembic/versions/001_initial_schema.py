"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-09 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("raw_input", sa.String(), nullable=False),
        sa.Column("parsed_config", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.Column("deactivated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_strategy_versions_one_active",
        "strategy_versions",
        ["is_active"],
        unique=True,
        sqlite_where=sa.text("is_active = 1"),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_order_id", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("filled_price", sa.Float(), nullable=True),
        sa.Column("order_type", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("rule_trigger", sa.String(length=500), nullable=True),
        sa.Column("strategy_version_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("filled_at", sa.DateTime(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_order_id"),
    )
    op.create_index("ix_trades_symbol_created_at", "trades", ["symbol", "created_at"])

    op.create_table(
        "alert_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("telegram_sent", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "pending_strategies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_input", sa.String(), nullable=False),
        sa.Column("parsed_config", sa.JSON(), nullable=False),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("executed_at", sa.DateTime(), nullable=True),
        sa.Column("action_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("pending_actions")
    op.drop_table("pending_strategies")
    op.drop_table("alert_logs")
    op.drop_index("ix_trades_symbol_created_at", table_name="trades")
    op.drop_table("trades")
    op.drop_index("ix_strategy_versions_one_active", table_name="strategy_versions")
    op.drop_table("strategy_versions")
