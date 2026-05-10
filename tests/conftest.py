from __future__ import annotations

import warnings
from collections.abc import AsyncGenerator
from datetime import date, time

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.storage.models import Base
from src.utils.market_hours import CalendarCache, MarketDay


def pytest_configure(config) -> None:
    # alpaca-py's stream module imports websockets.legacy which emits a
    # DeprecationWarning.  We can't control that third-party dependency, so we
    # silence it specifically here.  This hook runs after -W flags are applied
    # and inserts the ignore filter at position 0 (highest priority), so it
    # takes precedence over any -W error::DeprecationWarning CLI flag.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"websockets\..*",
    )


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory SQLite session. Tables are created fresh for every test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def sample_calendar() -> CalendarCache:
    """Regular trading week Mon 6 Jan – Fri 10 Jan 2025, plus Mon 13 Jan.
    Weekend (11–12 Jan) is intentionally absent.
    """
    return {
        date(2025, 1, 6): MarketDay(date(2025, 1, 6), time(9, 30), time(16, 0)),
        date(2025, 1, 7): MarketDay(date(2025, 1, 7), time(9, 30), time(16, 0)),
        date(2025, 1, 8): MarketDay(date(2025, 1, 8), time(9, 30), time(16, 0)),
        date(2025, 1, 9): MarketDay(date(2025, 1, 9), time(9, 30), time(16, 0)),
        date(2025, 1, 10): MarketDay(date(2025, 1, 10), time(9, 30), time(16, 0)),
        date(2025, 1, 13): MarketDay(date(2025, 1, 13), time(9, 30), time(16, 0)),
    }
