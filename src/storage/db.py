from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str) -> AsyncEngine:
    """Create the async engine and session factory. Call once at startup."""
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("DB engine not initialized. Call init_engine() first.")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager that yields a session with automatic commit/rollback.

    Usage:
        async with get_session() as session:
            session.add(obj)
        # commits on clean exit, rolls back on exception

    SQLAlchemy's AsyncSession does not auto-commit — the explicit commit/rollback
    here ensures every caller gets correct transactional semantics without
    having to remember to commit manually.
    """
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. Call init_engine() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables_for_testing() -> None:
    """Create all tables directly. For tests only — production uses Alembic migrations."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
