"""Deterministic historical simulation for validated daily strategies.

The backtester never calls an order endpoint. It downloads daily bars through
``AlpacaClient.get_bars`` and replays them chronologically through the same
``StrategyEngine`` used by the live agent. Signals are generated at a day's
close and filled at the next available session open to avoid look-ahead bias.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd

from src.broker.schemas import BarData, PositionSnapshot
from src.strategy.engine import StrategyEngine
from src.strategy.exceptions import InsufficientHistoryError
from src.strategy.schema import Strategy, Timeframe
from src.utils.market_hours import ET


class BacktestError(RuntimeError):
    """Raised when a simulation cannot produce a meaningful result."""


@dataclass(frozen=True)
class BacktestConfig:
    start_date: date
    end_date: date
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0
    warmup_calendar_days: int = 800

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.slippage_bps <= 100:
            raise ValueError("slippage_bps must be between 0 and 100")


@dataclass(frozen=True)
class BacktestFill:
    trade_date: date
    symbol: str
    side: str
    qty: float
    price: float
    reason: str
    pnl: float | None = None


@dataclass
class _Position:
    symbol: str
    qty: float
    entry_price: float
    entry_date: date
    peak_price: float


@dataclass(frozen=True)
class OpenPositionResult:
    symbol: str
    qty: float
    entry_price: float
    last_price: float
    unrealized_pnl: float


@dataclass(frozen=True)
class ConditionStat:
    label: str
    passed: int
    evaluated: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.evaluated if self.evaluated else 0.0


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    start_date: date
    end_date: date
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    benchmark_return_pct: float | None
    entry_signals: int
    fills: tuple[BacktestFill, ...]
    open_positions: tuple[OpenPositionResult, ...]
    condition_stats: tuple[ConditionStat, ...]
    rejection_counts: dict[str, int]
    data_errors: dict[str, str]
    trading_days: int

    @property
    def entries(self) -> int:
        return sum(fill.side == "BUY" for fill in self.fills)

    @property
    def exits(self) -> int:
        return sum(fill.side == "SELL" for fill in self.fills)

    @property
    def realized_pnl(self) -> float:
        return sum(fill.pnl or 0.0 for fill in self.fills if fill.side == "SELL")

    @property
    def unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.open_positions)

    @property
    def wins(self) -> int:
        return sum(
            fill.side == "SELL" and fill.pnl is not None and fill.pnl > 0
            for fill in self.fills
        )

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.exits if self.exits else None


def _bars_frame(bars: list[BarData]) -> tuple[pd.DataFrame, dict[date, int]]:
    if not bars:
        return pd.DataFrame(), {}
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    frame = pd.DataFrame(
        [
            {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in ordered
        ],
        index=pd.DatetimeIndex([bar.timestamp for bar in ordered]),
    )
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(UTC)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    by_date: dict[date, int] = {}
    for offset, timestamp in enumerate(frame.index):
        by_date[timestamp.astimezone(ET).date()] = offset
    return frame, by_date


def _mark_to_market(
    cash: float,
    positions: dict[str, _Position],
    marks: dict[str, float],
) -> float:
    return cash + sum(
        position.qty * marks.get(symbol, position.entry_price)
        for symbol, position in positions.items()
    )


def _position_snapshot(position: _Position, current_price: float) -> PositionSnapshot:
    pnl = (current_price - position.entry_price) * position.qty
    market_value = current_price * position.qty
    return PositionSnapshot(
        symbol=position.symbol,
        qty=position.qty,
        side="long",
        market_value=market_value,
        avg_entry_price=position.entry_price,
        unrealized_pl=pnl,
        unrealized_plpc=(current_price / position.entry_price - 1.0),
        current_price=current_price,
    )


async def run_backtest(
    strategy: Strategy,
    alpaca,
    config: BacktestConfig,
) -> BacktestResult:
    """Fetch and replay historical bars without accessing trading endpoints."""
    if strategy.timeframe != Timeframe.D1:
        raise BacktestError("El backtest histórico actualmente soporta estrategias 1D")

    from alpaca.data.enums import Adjustment
    from alpaca.data.timeframe import TimeFrame

    fetch_start = datetime.combine(
        config.start_date - timedelta(days=config.warmup_calendar_days),
        time.min,
        tzinfo=ET,
    ).astimezone(UTC)
    fetch_end = datetime.combine(
        config.end_date + timedelta(days=1),
        time.min,
        tzinfo=ET,
    ).astimezone(UTC)

    async def fetch(symbol: str) -> tuple[str, list[BarData] | Exception]:
        try:
            bars = await alpaca.get_bars(
                symbol,
                start=fetch_start,
                end=fetch_end,
                timeframe=TimeFrame.Day,
                adjustment=Adjustment.ALL,
            )
            return symbol, bars
        except Exception as exc:  # noqa: BLE001
            return symbol, exc

    fetched = await asyncio.gather(*(fetch(symbol) for symbol in strategy.universe))
    frames: dict[str, pd.DataFrame] = {}
    positions_by_date: dict[str, dict[date, int]] = {}
    data_errors: dict[str, str] = {}
    for symbol, payload in fetched:
        if isinstance(payload, Exception):
            data_errors[symbol] = str(payload)
            continue
        frame, date_offsets = _bars_frame(payload)
        if frame.empty:
            data_errors[symbol] = "sin barras históricas"
            continue
        frames[symbol] = frame
        positions_by_date[symbol] = date_offsets

    if not frames:
        raise BacktestError("Alpaca no devolvió datos históricos para ningún símbolo")

    engine = StrategyEngine(strategy)
    required = engine.required_lookback_bars()
    evaluation_dates = sorted(
        {
            trading_date
            for symbol, offsets in positions_by_date.items()
            for trading_date, offset in offsets.items()
            if config.start_date <= trading_date <= config.end_date
            and offset + 1 >= required
        }
    )
    if not evaluation_dates:
        raise BacktestError(
            f"Histórico insuficiente: se requieren {required} barras previas por símbolo"
        )

    cash = config.initial_cash
    positions: dict[str, _Position] = {}
    marks: dict[str, float] = {}
    pending_entries: dict[str, str] = {}
    pending_exits: dict[str, str] = {}
    fills: list[BacktestFill] = []
    equity_curve: list[float] = []
    rejection_counts: Counter[str] = Counter()
    condition_counts: dict[str, list[int]] = {}
    entry_signals = 0
    slippage = config.slippage_bps / 10_000.0

    for trading_date in evaluation_dates:
        todays_rows: dict[str, pd.Series] = {}
        for symbol, frame in frames.items():
            offset = positions_by_date[symbol].get(trading_date)
            if offset is not None:
                todays_rows[symbol] = frame.iloc[offset]

        # Signals from yesterday fill at today's open: exits first, then entries.
        for symbol in list(pending_exits):
            row = todays_rows.get(symbol)
            position = positions.get(symbol)
            if row is None or position is None:
                continue
            fill_price = float(row["open"]) * (1.0 - slippage)
            proceeds = position.qty * fill_price
            pnl = (fill_price - position.entry_price) * position.qty
            cash += proceeds
            fills.append(
                BacktestFill(
                    trade_date=trading_date,
                    symbol=symbol,
                    side="SELL",
                    qty=position.qty,
                    price=fill_price,
                    reason=pending_exits.pop(symbol),
                    pnl=pnl,
                )
            )
            positions.pop(symbol)

        for symbol, row in todays_rows.items():
            marks[symbol] = float(row["open"])

        for symbol in list(pending_entries):
            row = todays_rows.get(symbol)
            if row is None:
                continue
            reason = pending_entries.pop(symbol)
            if symbol in positions:
                rejection_counts["posición ya abierta"] += 1
                continue
            if len(positions) >= strategy.position_sizing.max_concurrent_positions:
                rejection_counts["máximo de posiciones"] += 1
                continue

            equity = _mark_to_market(cash, positions, marks)
            target_value = equity * strategy.position_sizing.max_position_pct / 100.0
            exposure = sum(
                position.qty * marks.get(held, position.entry_price)
                for held, position in positions.items()
            )
            max_exposure = equity * strategy.position_sizing.max_total_exposure_pct / 100.0
            if exposure + target_value > max_exposure + 0.01:
                rejection_counts["máxima exposición"] += 1
                continue

            fill_price = float(row["open"]) * (1.0 + slippage)
            qty = target_value / fill_price
            cost = qty * fill_price
            if cost * 1.01 > cash:
                rejection_counts["capital insuficiente"] += 1
                continue
            cash -= cost
            positions[symbol] = _Position(
                symbol=symbol,
                qty=qty,
                entry_price=fill_price,
                entry_date=trading_date,
                peak_price=max(fill_price, float(row["high"])),
            )
            fills.append(
                BacktestFill(
                    trade_date=trading_date,
                    symbol=symbol,
                    side="BUY",
                    qty=qty,
                    price=fill_price,
                    reason=reason,
                )
            )

        for symbol, row in todays_rows.items():
            marks[symbol] = float(row["close"])
            position = positions.get(symbol)
            if position is not None:
                position.peak_price = max(position.peak_price, float(row["high"]))

            offset = positions_by_date[symbol][trading_date]
            bars = frames[symbol].iloc[: offset + 1]
            try:
                outcomes = engine.entry_condition_outcomes(bars)
            except InsufficientHistoryError:
                continue
            for label, passed in outcomes:
                counts = condition_counts.setdefault(label, [0, 0])
                counts[0] += int(passed)
                counts[1] += 1

            if position is not None and symbol not in pending_exits:
                signal = engine.evaluate_exit(
                    symbol,
                    bars,
                    _position_snapshot(position, float(row["close"])),
                    high_since_entry=position.peak_price,
                )
                if signal is not None:
                    pending_exits[symbol] = signal.reason
                continue

            signal = engine.evaluate_entry(symbol, bars)
            if signal is not None:
                entry_signals += 1
                if symbol not in positions and symbol not in pending_entries:
                    pending_entries[symbol] = signal.reason

        equity_curve.append(_mark_to_market(cash, positions, marks))

    final_equity = _mark_to_market(cash, positions, marks)
    peak = config.initial_cash
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)

    open_positions = tuple(
        OpenPositionResult(
            symbol=symbol,
            qty=position.qty,
            entry_price=position.entry_price,
            last_price=marks.get(symbol, position.entry_price),
            unrealized_pnl=(marks.get(symbol, position.entry_price) - position.entry_price)
            * position.qty,
        )
        for symbol, position in sorted(positions.items())
    )
    condition_stats = tuple(
        ConditionStat(label=label, passed=counts[0], evaluated=counts[1])
        for label, counts in condition_counts.items()
    )

    benchmark_return: float | None = None
    benchmark = frames.get("SPY")
    benchmark_offsets = positions_by_date.get("SPY", {})
    if benchmark is not None:
        available = [d for d in evaluation_dates if d in benchmark_offsets]
        if available:
            first = benchmark.iloc[benchmark_offsets[available[0]]]
            last = benchmark.iloc[benchmark_offsets[available[-1]]]
            benchmark_return = (float(last["close"]) / float(first["open"]) - 1.0) * 100

    return BacktestResult(
        strategy_name=strategy.name,
        start_date=evaluation_dates[0],
        end_date=evaluation_dates[-1],
        initial_equity=config.initial_cash,
        final_equity=final_equity,
        total_return_pct=(final_equity / config.initial_cash - 1.0) * 100,
        max_drawdown_pct=max_drawdown * 100,
        benchmark_return_pct=benchmark_return,
        entry_signals=entry_signals,
        fills=tuple(fills),
        open_positions=open_positions,
        condition_stats=condition_stats,
        rejection_counts=dict(rejection_counts),
        data_errors=data_errors,
        trading_days=len(evaluation_dates),
    )


def format_backtest_report(result: BacktestResult) -> str:
    """Render a compact plain-text report safe for Telegram's message limit."""
    benchmark = (
        f"{result.benchmark_return_pct:+.2f}%"
        if result.benchmark_return_pct is not None
        else "N/A"
    )
    win_rate = f"{result.win_rate:.0%}" if result.win_rate is not None else "N/A"
    lines = [
        f"🧪 Backtest: {result.strategy_name}",
        f"Periodo: {result.start_date} → {result.end_date} ({result.trading_days} ruedas)",
        "",
        f"Capital inicial: ${result.initial_equity:,.2f}",
        f"Capital final: ${result.final_equity:,.2f}",
        f"Retorno: {result.total_return_pct:+.2f}% | SPY: {benchmark}",
        f"Drawdown máximo: {result.max_drawdown_pct:.2f}%",
        f"P&L realizado: ${result.realized_pnl:+,.2f}",
        f"P&L abierto: ${result.unrealized_pnl:+,.2f}",
        "",
        f"Señales de entrada: {result.entry_signals}",
        f"Entradas: {result.entries} | Salidas: {result.exits} | Win rate: {win_rate}",
        f"Posiciones abiertas al final: {len(result.open_positions)}",
        "",
        "Condiciones de entrada cumplidas:",
    ]
    for stat in result.condition_stats:
        lines.append(
            f"• {stat.label}: {stat.passed}/{stat.evaluated} ({stat.pass_rate:.1%})"
        )
    if result.rejection_counts:
        lines.extend(["", "Señales bloqueadas:"])
        lines.extend(
            f"• {reason}: {count}"
            for reason, count in sorted(result.rejection_counts.items())
        )
    if result.fills:
        lines.extend(["", "Últimas ejecuciones simuladas:"])
        for fill in result.fills[-10:]:
            pnl = f" | P&L ${fill.pnl:+,.2f}" if fill.pnl is not None else ""
            lines.append(
                f"• {fill.trade_date} {fill.side} {fill.symbol} "
                f"{fill.qty:.4g} @ ${fill.price:.2f}{pnl}"
            )
    if result.data_errors:
        lines.extend(["", "Errores de datos:"])
        lines.extend(f"• {symbol}: {error}" for symbol, error in result.data_errors.items())
    lines.extend(
        [
            "",
            "Supuestos: señal al cierre, ejecución en la próxima apertura, "
            "slippage 5 bps, acciones fraccionarias y sin impuestos.",
        ]
    )
    report = "\n".join(lines)
    return report if len(report) <= 4000 else report[:3990] + "…"
