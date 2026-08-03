"""Run a historical strategy simulation from the command line.

Example:
    python -m src.backtest_cli \
      --strategy strategies/etf_pullback_trend_filtered.json \
      --days 30 --send-telegram
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient

from src.backtest import BacktestConfig, format_backtest_report, run_backtest
from src.broker.alpaca_client import AlpacaClient
from src.broker.rate_limiter import TokenBucketLimiter
from src.config import get_settings
from src.strategy.schema import Strategy
from src.utils.market_hours import ET


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simula una estrategia con datos de Alpaca")
    parser.add_argument("--strategy", type=Path, required=True, help="Archivo JSON validado")
    parser.add_argument("--days", type=int, default=30, help="Días calendario (default: 30)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Envía el reporte al chat autorizado configurado en .env",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not 5 <= args.days <= 730:
        raise SystemExit("--days debe estar entre 5 y 730")
    strategy = Strategy.model_validate(json.loads(args.strategy.read_text(encoding="utf-8")))
    settings = get_settings()
    data_client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_api_secret,
    )
    alpaca = AlpacaClient(
        trading_client=None,
        data_client=data_client,
        settings=settings,
        rate_limiter=TokenBucketLimiter(rate=200 / 60, burst=20),
    )
    end_date = datetime.now(ET).date() - timedelta(days=1)
    result = await run_backtest(
        strategy,
        alpaca,
        BacktestConfig(
            start_date=end_date - timedelta(days=args.days - 1),
            end_date=end_date,
            initial_cash=args.initial_cash,
            slippage_bps=args.slippage_bps,
        ),
    )
    report = format_backtest_report(result)
    print(report)
    if args.send_telegram:
        from telegram import Bot

        async with Bot(settings.telegram_bot_token) as bot:
            await bot.send_message(
                chat_id=settings.telegram_authorized_chat_id,
                text=report,
            )


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
