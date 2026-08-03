"""Activate an exact validated strategy JSON in the configured database."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings
from src.storage.models import StrategyVersion
from src.strategy.schema import Strategy


async def activate_strategy(
    strategy_path: Path,
    session_factory: async_sessionmaker[AsyncSession],
) -> Strategy:
    strategy = Strategy.model_validate(json.loads(strategy_path.read_text(encoding="utf-8")))
    now = datetime.now(UTC)
    async with session_factory() as session:
        await session.execute(
            update(StrategyVersion)
            .where(StrategyVersion.is_active.is_(True))
            .values(is_active=False, deactivated_at=now)
        )
        session.add(
            StrategyVersion(
                name=strategy.name,
                raw_input=f"exact-json:{strategy_path.name}",
                parsed_config=strategy.model_dump(mode="json"),
                is_active=True,
                activated_at=now,
            )
        )
        await session.commit()
    return strategy


async def _run(path: Path) -> None:
    settings = get_settings()
    if settings.is_live:
        raise SystemExit(
            "Refusing strategy activation against a live Alpaca URL; use paper first"
        )
    engine = create_async_engine(settings.database_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        strategy = await activate_strategy(path, factory)
        async with factory() as session:
            active = (
                await session.execute(
                    select(StrategyVersion).where(StrategyVersion.is_active.is_(True))
                )
            ).scalars().one()
        print(
            f"Activated {strategy.name!r} from {path} "
            f"(strategy_version_id={active.id}, paper_only=True)"
        )
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.strategy.resolve()))


if __name__ == "__main__":
    main()
