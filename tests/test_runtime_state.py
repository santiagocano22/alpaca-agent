from __future__ import annotations

import pytest

from src.storage.runtime_state import (
    load_runtime_state,
    persist_bot_paused,
    persist_position_highs,
)


@pytest.mark.asyncio
async def test_runtime_state_survives_separate_sessions(async_session) -> None:
    await persist_bot_paused(async_session, True)
    await persist_position_highs(async_session, {"spy": 612.5, "QQQ": 555.25})

    row = await load_runtime_state(async_session)

    assert row.bot_paused is True
    assert row.position_highs == {"SPY": 612.5, "QQQ": 555.25}


@pytest.mark.asyncio
async def test_runtime_state_defaults_are_safe_and_empty(async_session) -> None:
    row = await load_runtime_state(async_session)

    assert row.bot_paused is False
    assert row.position_highs == {}
