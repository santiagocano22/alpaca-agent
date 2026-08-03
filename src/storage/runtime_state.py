"""Persistence helpers for operational state that must survive restarts."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import RuntimeState

_RUNTIME_STATE_ID = 1


async def load_runtime_state(session: AsyncSession) -> RuntimeState:
    row = await session.get(RuntimeState, _RUNTIME_STATE_ID)
    if row is None:
        row = RuntimeState(
            id=_RUNTIME_STATE_ID,
            bot_paused=False,
            position_highs={},
            updated_at=datetime.now(UTC),
        )
        session.add(row)
        await session.commit()
    return row


async def persist_bot_paused(session: AsyncSession, paused: bool) -> None:
    row = await load_runtime_state(session)
    row.bot_paused = paused
    row.updated_at = datetime.now(UTC)
    await session.commit()


async def persist_position_highs(
    session: AsyncSession,
    position_highs: dict[str, float],
) -> None:
    row = await load_runtime_state(session)
    row.position_highs = {
        symbol.upper(): float(high) for symbol, high in position_highs.items()
    }
    row.updated_at = datetime.now(UTC)
    await session.commit()
