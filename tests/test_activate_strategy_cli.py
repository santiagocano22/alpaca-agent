from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.storage.models import Base, StrategyVersion
from src.strategy.activate_cli import activate_strategy


@pytest.mark.asyncio
async def test_activate_strategy_preserves_history_and_exact_json(tmp_path: Path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    source = Path("strategies/etf_breakout_20_recommended.json")
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    async with factory() as session:
        session.add(
            StrategyVersion(
                name="old",
                raw_input="old",
                parsed_config=json.loads(source.read_text(encoding="utf-8")),
                is_active=True,
            )
        )
        await session.commit()

    activated = await activate_strategy(strategy_path, factory)

    async with factory() as session:
        rows=(await session.execute(select(StrategyVersion).order_by(StrategyVersion.id))).scalars().all()
    assert [row.is_active for row in rows] == [False, True]
    assert rows[-1].parsed_config == activated.model_dump(mode="json")
    assert rows[-1].raw_input == "exact-json:strategy.json"
    await engine.dispose()
