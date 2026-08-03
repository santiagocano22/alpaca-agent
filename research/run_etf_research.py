"""Reproducible ETF strategy research using Alpaca daily bars.

This module is deliberately separate from the live trading loop.  It never
constructs a TradingClient and therefore cannot submit orders.  Signals use a
completed daily bar and simulated fills occur at the next available open.

Usage:
    .venv/bin/python -m research.run_etf_research --download
    .venv/bin/python -m research.run_etf_research --run
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

UNIVERSE = ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLI", "XLV", "XLY", "XLE")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "research" / "data"
RESULTS_DIR = ROOT / "research" / "results"
DOWNLOAD_START = datetime(2018, 1, 1, tzinfo=UTC)
DOWNLOAD_END = datetime(2026, 8, 1, tzinfo=UTC)

SPLITS = {
    "train": (date(2020, 1, 2), date(2022, 12, 30)),
    "validation": (date(2023, 1, 3), date(2024, 6, 28)),
    "test": (date(2024, 7, 1), date(2026, 7, 31)),
    "full": (date(2020, 1, 2), date(2026, 7, 31)),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    signal: str
    inverse: str
    stop_loss_pct: float
    trailing_stop_pct: float
    position_pct: float = 15.0
    max_exposure_pct: float = 75.0
    max_positions: int = 5
    deployable: bool = True


CANDIDATES = (
    Candidate(
        "baseline_rsi_cross_1", "pullback", "baseline_cross_1", "below_ema50_cross",
        6.0, 8.0, 12.0, 48.0, 4,
    ),
    Candidate("rsi_cross_3", "pullback", "baseline_cross_3", "below_ema50_cross", 6.0, 8.0),
    Candidate("rsi_cross_5", "pullback", "baseline_cross_5", "below_ema50_cross", 6.0, 8.0),
    Candidate("rsi_zone_40_55", "pullback", "rsi_zone", "below_ema50_cross", 6.0, 8.0),
    Candidate("pullback_below_ema20", "pullback", "pullback_below", "below_ema50_cross", 7.0, 9.0),
    Candidate("pullback_reclaim_ema20", "pullback", "pullback_reclaim", "below_ema50_cross", 7.0, 9.0),
    Candidate("breakout_20", "breakout", "breakout_20", "below_ema20_cross", 7.0, 10.0),
    Candidate("breakout_50", "breakout", "breakout_50", "below_ema20_cross", 7.0, 10.0),
    Candidate("relative_momentum_126", "relative_momentum", "relative_126", "relative_exit", 8.0, 12.0, deployable=False),
    Candidate("mean_reversion_rsi_35", "mean_reversion", "meanrev_rsi", "rsi_55", 6.0, 8.0),
    Candidate("mean_reversion_bbands", "mean_reversion", "meanrev_bbands", "price_above_ema20", 6.0, 8.0),
    Candidate("simple_trend", "trend", "simple_trend", "below_ema50_cross", 7.0, 10.0),
)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_date: date
    peak: float
    entry_value: float


@dataclass(frozen=True)
class Fill:
    trade_date: date
    symbol: str
    side: str
    price: float
    qty: float
    value: float
    pnl: float | None
    return_pct: float | None
    reason: str


@dataclass
class Simulation:
    candidate: Candidate
    split: str
    slippage_bps: float
    equity: pd.Series
    gross_exposure: pd.Series
    position_count: pd.Series
    fills: list[Fill]
    entry_signals: int
    rejection_counts: dict[str, int]
    benchmark: pd.Series


def download_bars() -> None:
    """Download adjusted SIP bars; no trading client or order endpoint exists."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from src.config import get_settings

    settings = get_settings()
    client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_api_secret)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in UNIVERSE:
        # Alpaca's multi-symbol endpoint applies its page limit across the
        # whole request and can yield an incomplete symbol mapping.  One
        # request per ETF makes completeness independently verifiable.
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=DOWNLOAD_START,
            end=DOWNLOAD_END,
            limit=10_000,
            adjustment=Adjustment.ALL,
            feed=DataFeed.SIP,
        )
        bar_set = client.get_stock_bars(request)
        rows = bar_set.data.get(symbol, [])
        if not rows:
            raise RuntimeError(f"Alpaca returned no bars for {symbol}")
        frame = pd.DataFrame(
            {
                "timestamp": [row.timestamp for row in rows],
                "open": [float(row.open) for row in rows],
                "high": [float(row.high) for row in rows],
                "low": [float(row.low) for row in rows],
                "close": [float(row.close) for row in rows],
                "volume": [float(row.volume) for row in rows],
            }
        )
        frame.to_csv(DATA_DIR / f"{symbol}.csv", index=False)
        print(symbol, len(frame), frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1])


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi14"] = (100.0 - 100.0 / (1.0 + rs)).where(avg_loss != 0.0, 100.0)
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["sma200"] = close.rolling(200, min_periods=200).mean()
    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=1)
    out["bb_lower"] = middle - 2.0 * std
    out["prior_high20"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["prior_high50"] = out["high"].shift(1).rolling(50, min_periods=50).max()
    out["mom126"] = close / close.shift(126) - 1.0
    out["cross_rsi40"] = (out["rsi14"].shift(1) <= 40) & (out["rsi14"] > 40)
    out["cross_above_ema20"] = (close.shift(1) <= out["ema20"].shift(1)) & (close > out["ema20"])
    out["cross_below_ema20"] = (close.shift(1) >= out["ema20"].shift(1)) & (close < out["ema20"])
    out["cross_below_ema50"] = (close.shift(1) >= out["ema50"].shift(1)) & (close < out["ema50"])
    return out


def load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing = [symbol for symbol in UNIVERSE if not (DATA_DIR / f"{symbol}.csv").exists()]
    if missing:
        raise RuntimeError(f"Missing cached bars for: {', '.join(missing)}; run with --download")
    for symbol in UNIVERSE:
        frame = pd.read_csv(DATA_DIR / f"{symbol}.csv", parse_dates=["timestamp"])
        index = pd.DatetimeIndex(frame.pop("timestamp"), name="timestamp")
        if index.tz is None:
            index = index.tz_localize(UTC)
        frame.index = index
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frames[symbol] = _features(frame)
    return frames


def _rolling_recent(signal: pd.Series, bars: int) -> pd.Series:
    return signal.astype(float).rolling(bars, min_periods=1).max().astype(bool)


def build_signals(
    frames: dict[str, pd.DataFrame], candidate: Candidate
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[pd.Timestamp, list[str]]]:
    entries: dict[str, pd.Series] = {}
    exits: dict[str, pd.Series] = {}
    priority: dict[pd.Timestamp, list[str]] = {}

    common_dates = sorted(set().union(*(set(frame.index) for frame in frames.values())))
    momentum = pd.DataFrame({s: frames[s]["mom126"] for s in UNIVERSE}).reindex(common_dates)
    ranks = momentum.rank(axis=1, ascending=False, method="first")

    for symbol, frame in frames.items():
        close = frame["close"]
        trend = (close > frame["sma200"]) & (frame["ema20"] > frame["ema50"])
        full_trend = trend & (close > frame["ema20"])
        cross = frame["cross_rsi40"]
        if candidate.signal.startswith("baseline_cross_"):
            lookback = int(candidate.signal.rsplit("_", 1)[1])
            entry = full_trend & _rolling_recent(cross, lookback)
        elif candidate.signal == "rsi_zone":
            entry = full_trend & frame["rsi14"].between(40, 55, inclusive="both")
        elif candidate.signal == "pullback_below":
            entry = trend & (close <= frame["ema20"]) & (frame["rsi14"] >= 35)
        elif candidate.signal == "pullback_reclaim":
            recent_reclaim = _rolling_recent(frame["cross_above_ema20"], 3)
            entry = trend & (close > frame["ema20"]) & recent_reclaim & (frame["rsi14"] <= 60)
        elif candidate.signal.startswith("breakout_"):
            lookback = int(candidate.signal.rsplit("_", 1)[1])
            prior_high = frame["high"].shift(1).rolling(lookback, min_periods=lookback).max()
            entry = (close > frame["sma200"]) & (close > prior_high)
        elif candidate.signal == "meanrev_rsi":
            entry = (close > frame["sma200"]) & (frame["rsi14"] <= 35)
        elif candidate.signal == "meanrev_bbands":
            entry = (close > frame["sma200"]) & (close < frame["bb_lower"])
        elif candidate.signal == "simple_trend":
            entry = trend
        elif candidate.signal == "relative_126":
            aligned_rank = ranks[symbol].reindex(frame.index)
            month_end = pd.Series(False, index=frame.index)
            month_end.loc[frame.groupby(frame.index.strftime("%Y-%m")).tail(1).index] = True
            entry = month_end & (aligned_rank <= 4) & (close > frame["sma200"])
        else:  # pragma: no cover - candidate table is closed
            raise ValueError(candidate.signal)

        if candidate.inverse == "below_ema50_cross":
            inverse = frame["cross_below_ema50"]
        elif candidate.inverse == "below_ema20_cross":
            inverse = frame["cross_below_ema20"]
        elif candidate.inverse == "rsi_55":
            inverse = frame["rsi14"] >= 55
        elif candidate.inverse == "price_above_ema20":
            inverse = close > frame["ema20"]
        elif candidate.inverse == "relative_exit":
            aligned_rank = ranks[symbol].reindex(frame.index)
            month_end = pd.Series(False, index=frame.index)
            month_end.loc[frame.groupby(frame.index.strftime("%Y-%m")).tail(1).index] = True
            inverse = month_end & ((aligned_rank > 4) | (close <= frame["sma200"]))
        else:  # pragma: no cover
            raise ValueError(candidate.inverse)

        entries[symbol] = entry.fillna(False).astype(bool)
        exits[symbol] = inverse.fillna(False).astype(bool)

    if candidate.signal == "relative_126":
        for timestamp in common_dates:
            row = ranks.loc[timestamp]
            priority[timestamp] = [s for s in row.sort_values().index if pd.notna(row[s])]
    return entries, exits, priority


def _date_index(frame: pd.DataFrame) -> dict[date, int]:
    return {timestamp.date(): offset for offset, timestamp in enumerate(frame.index)}


def simulate(
    frames: dict[str, pd.DataFrame], candidate: Candidate, split: str, slippage_bps: float
) -> Simulation:
    start, end = SPLITS[split]
    entries, inverse_exits, priorities = build_signals(frames, candidate)
    offsets = {symbol: _date_index(frame) for symbol, frame in frames.items()}
    dates = sorted(
        d for d in set().union(*(set(mapping) for mapping in offsets.values())) if start <= d <= end
    )
    cash = 100_000.0
    positions: dict[str, Position] = {}
    marks: dict[str, float] = {}
    pending_entries: dict[str, str] = {}
    pending_exits: dict[str, str] = {}
    fills: list[Fill] = []
    equity_values: list[float] = []
    exposure_values: list[float] = []
    count_values: list[int] = []
    rejection_counts: dict[str, int] = {}
    entry_signals = 0
    slip = slippage_bps / 10_000.0

    def mark_equity() -> float:
        return cash + sum(p.qty * marks.get(s, p.entry_price) for s, p in positions.items())

    for trading_date in dates:
        rows: dict[str, pd.Series] = {}
        timestamps: dict[str, pd.Timestamp] = {}
        for symbol, frame in frames.items():
            offset = offsets[symbol].get(trading_date)
            if offset is not None:
                rows[symbol] = frame.iloc[offset]
                timestamps[symbol] = frame.index[offset]

        for symbol in list(pending_exits):
            if symbol not in rows or symbol not in positions:
                continue
            position = positions.pop(symbol)
            price = float(rows[symbol]["open"]) * (1.0 - slip)
            value = position.qty * price
            pnl = value - position.entry_value
            cash += value
            fills.append(Fill(trading_date, symbol, "SELL", price, position.qty, value, pnl,
                              pnl / position.entry_value * 100.0, pending_exits.pop(symbol)))

        for symbol, row in rows.items():
            marks[symbol] = float(row["open"])

        ordered = list(pending_entries)
        if candidate.signal == "relative_126" and timestamps:
            timestamp = next(iter(timestamps.values()))
            rank_order = priorities.get(timestamp, [])
            ordered.sort(key=lambda symbol: rank_order.index(symbol) if symbol in rank_order else 999)

        for symbol in ordered:
            if symbol not in rows:
                continue
            reason = pending_entries.pop(symbol)
            if symbol in positions:
                rejection_counts["already_open"] = rejection_counts.get("already_open", 0) + 1
                continue
            if len(positions) >= candidate.max_positions:
                rejection_counts["max_positions"] = rejection_counts.get("max_positions", 0) + 1
                continue
            equity = mark_equity()
            target = equity * candidate.position_pct / 100.0
            exposure = sum(p.qty * marks.get(s, p.entry_price) for s, p in positions.items())
            if exposure + target > equity * candidate.max_exposure_pct / 100.0 + 0.01:
                rejection_counts["max_exposure"] = rejection_counts.get("max_exposure", 0) + 1
                continue
            price = float(rows[symbol]["open"]) * (1.0 + slip)
            qty = target / price
            value = qty * price
            if value > cash + 0.01:
                rejection_counts["cash"] = rejection_counts.get("cash", 0) + 1
                continue
            cash -= value
            positions[symbol] = Position(
                symbol, qty, price, trading_date, max(price, float(rows[symbol]["high"])), value
            )
            fills.append(Fill(trading_date, symbol, "BUY", price, qty, value, None, None, reason))

        for symbol, row in rows.items():
            marks[symbol] = float(row["close"])
            position = positions.get(symbol)
            timestamp = timestamps[symbol]
            if position is not None:
                position.peak = max(position.peak, float(row["high"]))
                close = float(row["close"])
                if close <= position.entry_price * (1.0 - candidate.stop_loss_pct / 100.0):
                    pending_exits.setdefault(symbol, "stop_loss")
                elif close <= position.peak * (1.0 - candidate.trailing_stop_pct / 100.0):
                    pending_exits.setdefault(symbol, "trailing_stop")
                elif bool(inverse_exits[symbol].loc[timestamp]):
                    pending_exits.setdefault(symbol, "inverse_signal")
                continue
            if bool(entries[symbol].loc[timestamp]):
                entry_signals += 1
                pending_entries.setdefault(symbol, candidate.signal)

        equity = mark_equity()
        gross = sum(p.qty * marks.get(s, p.entry_price) for s, p in positions.items())
        equity_values.append(equity)
        exposure_values.append(gross / equity if equity > 0 else 0.0)
        count_values.append(len(positions))

    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    equity = pd.Series(equity_values, index=index, name="equity")
    exposure = pd.Series(exposure_values, index=index, name="gross_exposure")
    counts = pd.Series(count_values, index=index, name="positions")
    spy = frames["SPY"]
    spy_slice = spy[(spy.index.date >= start) & (spy.index.date <= end)]
    benchmark = spy_slice["close"] / float(spy_slice["open"].iloc[0]) * 100_000.0
    benchmark.index = pd.DatetimeIndex(benchmark.index.date)
    benchmark = benchmark.reindex(index).ffill()
    return Simulation(candidate, split, slippage_bps, equity, exposure, counts, fills,
                      entry_signals, rejection_counts, benchmark)


def metrics(sim: Simulation) -> dict[str, object]:
    equity = sim.equity
    returns = equity.pct_change().fillna(equity.iloc[0] / 100_000.0 - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.2425, 1 / 252)
    total_return = equity.iloc[-1] / 100_000.0 - 1.0
    cagr = (equity.iloc[-1] / 100_000.0) ** (1.0 / years) - 1.0
    vol = returns.std(ddof=1) * math.sqrt(252)
    sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(252) if returns.std(ddof=1) else np.nan
    downside = math.sqrt(float((returns.clip(upper=0.0) ** 2).mean()))
    sortino = returns.mean() / downside * math.sqrt(252) if downside else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    completed = [fill for fill in sim.fills if fill.side == "SELL" and fill.return_pct is not None]
    trade_returns = np.array([float(fill.return_pct) for fill in completed], dtype=float)
    pnls = np.array([float(fill.pnl) for fill in completed], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    buys = [fill for fill in sim.fills if fill.side == "BUY"]
    buy_months = {fill.trade_date.strftime("%Y-%m") for fill in buys}
    months = pd.period_range(equity.index[0], equity.index[-1], freq="M")
    turnover = sum(fill.value for fill in sim.fills) / float(equity.mean()) / years
    benchmark_return = sim.benchmark.iloc[-1] / 100_000.0 - 1.0
    result: dict[str, object] = {
        "candidate": sim.candidate.name,
        "family": sim.candidate.family,
        "deployable": sim.candidate.deployable,
        "split": sim.split,
        "slippage_bps": sim.slippage_bps,
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "volatility_pct": vol * 100,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown_pct": max_dd * 100,
        "calmar": cagr / abs(max_dd) if max_dd else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else np.nan,
        "expectancy_trade_pct": float(trade_returns.mean()) if len(trade_returns) else np.nan,
        "win_rate_pct": float((trade_returns > 0).mean() * 100) if len(trade_returns) else np.nan,
        "avg_gain_pct": float(trade_returns[trade_returns > 0].mean()) if np.any(trade_returns > 0) else np.nan,
        "avg_loss_pct": float(trade_returns[trade_returns < 0].mean()) if np.any(trade_returns < 0) else np.nan,
        "best_trade_pct": float(trade_returns.max()) if len(trade_returns) else np.nan,
        "worst_trade_pct": float(trade_returns.min()) if len(trade_returns) else np.nan,
        "entries": len(buys),
        "exits": len(completed),
        "entries_per_month": len(buys) / len(months),
        "zero_entry_months": len(months) - len(buy_months),
        "time_in_market_pct": float((sim.position_count > 0).mean() * 100),
        "average_gross_exposure_pct": float(sim.gross_exposure.mean() * 100),
        "annual_turnover_x": turnover,
        "entry_signals": sim.entry_signals,
        "benchmark_return_pct": benchmark_return * 100,
        "final_equity": float(equity.iloc[-1]),
    }
    return result


def _score(row: pd.Series) -> float:
    activity_penalty = abs(float(row["entries_per_month"]) - 4.0) * 0.15
    dd_penalty = max(0.0, abs(float(row["max_drawdown_pct"])) - 25.0) * 0.1
    return (
        float(row["sharpe"]) + 0.35 * float(row["calmar"])
        + min(float(row["entries_per_month"]), 2.0) * 0.15
        - activity_penalty - dd_penalty
    )


def bootstrap_trade_sequence(sim: Simulation, samples: int = 10_000, seed: int = 586) -> dict[str, float]:
    # A trade return applies only to its target portfolio weight.  Multiplying
    # by that weight avoids the severe overstatement from compounding every
    # position-level return as if it represented the whole portfolio.
    weight = sim.candidate.position_pct / 100.0
    completed = [fill.return_pct / 100.0 * weight for fill in sim.fills if fill.return_pct is not None]
    if not completed:
        return {}
    rng = np.random.default_rng(seed)
    values = np.asarray(completed)
    terminal: list[float] = []
    max_dd: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(values, size=len(values), replace=True)
        path = np.cumprod(1.0 + sampled)
        terminal.append(float(path[-1] - 1.0))
        peaks = np.maximum.accumulate(np.r_[1.0, path])
        dd = np.r_[1.0, path] / peaks - 1.0
        max_dd.append(float(dd.min()))
    return {
        "samples": samples,
        "terminal_return_p05_pct": float(np.quantile(terminal, 0.05) * 100),
        "terminal_return_median_pct": float(np.median(terminal) * 100),
        "terminal_return_p95_pct": float(np.quantile(terminal, 0.95) * 100),
        "max_drawdown_p50_pct": float(np.median(max_dd) * 100),
        "max_drawdown_p05_worst_pct": float(np.quantile(max_dd, 0.05) * 100),
    }


def symbol_metrics(sim: Simulation) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in UNIVERSE:
        trades = [f for f in sim.fills if f.side == "SELL" and f.symbol == symbol]
        returns = [float(f.return_pct) for f in trades if f.return_pct is not None]
        pnls = [float(f.pnl) for f in trades if f.pnl is not None]
        rows.append({
            "symbol": symbol,
            "trades": len(trades),
            "mean_trade_pct": float(np.mean(returns)) if returns else np.nan,
            "total_realized_pnl": float(np.sum(pnls)) if pnls else 0.0,
            "win_rate_pct": float(np.mean(np.asarray(returns) > 0) * 100) if returns else np.nan,
        })
    return rows


def baseline_last_month_diagnostics(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    start, end = date(2026, 7, 1), date(2026, 7, 31)
    counts = {
        "price_above_sma200": 0,
        "ema20_above_ema50": 0,
        "rsi14_crosses_above_40_one_bar": 0,
        "price_above_ema20": 0,
        "complete_entry": 0,
    }
    evaluated = 0
    per_symbol: dict[str, int] = {}
    for symbol, frame in frames.items():
        subset = frame[(frame.index.date >= start) & (frame.index.date <= end)]
        conditions = {
            "price_above_sma200": subset["close"] > subset["sma200"],
            "ema20_above_ema50": subset["ema20"] > subset["ema50"],
            "rsi14_crosses_above_40_one_bar": subset["cross_rsi40"],
            "price_above_ema20": subset["close"] > subset["ema20"],
        }
        complete = pd.concat(conditions, axis=1).all(axis=1)
        for name, values in conditions.items():
            counts[name] += int(values.sum())
        counts["complete_entry"] += int(complete.sum())
        per_symbol[symbol] = int(complete.sum())
        evaluated += len(subset)
    return {
        "period": f"{start}..{end}",
        "symbol_days_evaluated": evaluated,
        "condition_pass_counts": counts,
        "complete_entries_by_symbol": per_symbol,
    }


def regime_metrics(sim: Simulation, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    returns = sim.equity.pct_change().fillna(sim.equity.iloc[0] / 100_000.0 - 1.0)
    spy = frames["SPY"].copy()
    spy.index = pd.DatetimeIndex(spy.index.date)
    bull = (spy["close"] > spy["sma200"]).reindex(returns.index).ffill().fillna(False)
    rows: list[dict[str, object]] = []
    for label, mask in (("SPY_above_SMA200", bull), ("SPY_at_or_below_SMA200", ~bull)):
        selected = returns[mask]
        rows.append({
            "regime": label,
            "days": len(selected),
            "cumulative_return_pct": (float((1.0 + selected).prod()) - 1.0) * 100.0,
            "annualized_volatility_pct": float(selected.std(ddof=1) * math.sqrt(252) * 100.0),
            "mean_daily_return_bps": float(selected.mean() * 10_000.0),
        })
    return rows


def run_research() -> None:
    frames = load_frames()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    simulations: dict[tuple[str, str, float], Simulation] = {}
    for candidate in CANDIDATES:
        for split in SPLITS:
            sim = simulate(frames, candidate, split, 5.0)
            simulations[(candidate.name, split, 5.0)] = sim
            metric_rows.append(metrics(sim))
        for slip in (10.0, 20.0):
            sim = simulate(frames, candidate, "full", slip)
            simulations[(candidate.name, "full", slip)] = sim
            metric_rows.append(metrics(sim))

    table = pd.DataFrame(metric_rows)
    table.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)
    full = table[(table["split"] == "full") & (table["slippage_bps"] == 5.0)].copy()
    validation = table[(table["split"] == "validation") & (table["slippage_bps"] == 5.0)].set_index("candidate")
    train = table[(table["split"] == "train") & (table["slippage_bps"] == 5.0)].set_index("candidate")
    full["selection_score"] = [
        (_score(train.loc[name]) + 2.0 * _score(validation.loc[name])) / 3.0
        for name in full["candidate"]
    ]
    full.sort_values("selection_score", ascending=False).to_csv(
        RESULTS_DIR / "candidate_comparison.csv", index=False
    )

    # Anchored walk-forward selection: only preceding observations choose the
    # candidate applied to the next untouched calendar year.
    wf_rows: list[dict[str, object]] = []
    deployable = [c for c in CANDIDATES if c.deployable]
    original_splits = dict(SPLITS)
    try:
        for year in (2022, 2023, 2024, 2025, 2026):
            SPLITS["wf_train"] = (date(2020, 1, 2), date(year - 1, 12, 31))
            SPLITS["wf_oos"] = (date(year, 1, 1), date(year, 12, 31) if year < 2026 else date(2026, 7, 31))
            scored: list[tuple[float, Candidate]] = []
            for candidate in deployable:
                row = pd.Series(metrics(simulate(frames, candidate, "wf_train", 5.0)))
                scored.append((_score(row), candidate))
            _, winner = max(scored, key=lambda item: item[0])
            oos = metrics(simulate(frames, winner, "wf_oos", 5.0))
            wf_rows.append({"oos_year": year, "selected_on_prior_data": winner.name, **oos})
    finally:
        SPLITS.clear()
        SPLITS.update(original_splits)
    pd.DataFrame(wf_rows).to_csv(RESULTS_DIR / "walk_forward.csv", index=False)

    best_name = str(full.sort_values("selection_score", ascending=False).iloc[0]["candidate"])
    best_sim = simulations[(best_name, "full", 5.0)]
    pd.DataFrame(symbol_metrics(best_sim)).to_csv(RESULTS_DIR / "recommended_by_symbol.csv", index=False)
    pd.DataFrame(regime_metrics(best_sim, frames)).to_csv(
        RESULTS_DIR / "recommended_by_regime.csv", index=False
    )
    by_year: list[dict[str, object]] = []
    for year in range(2020, 2027):
        SPLITS["year"] = (date(year, 1, 1), date(year, 12, 31) if year < 2026 else date(2026, 7, 31))
        by_year.append(metrics(simulate(frames, best_sim.candidate, "year", 5.0)))
        by_year[-1]["year"] = year
    SPLITS.pop("year", None)
    pd.DataFrame(by_year).to_csv(RESULTS_DIR / "recommended_by_year.csv", index=False)

    diagnostics = baseline_last_month_diagnostics(frames)
    with (RESULTS_DIR / "baseline_last_month.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    # Local robustness grid around the selected 20-day breakout.  Every
    # combination is evaluated without changing position sizing or exits.
    sensitivity_rows: list[dict[str, object]] = []
    if best_name.startswith("breakout_"):
        for lookback in (15, 20, 25):
            for stop in (6.0, 7.0, 8.0):
                for trail in (9.0, 10.0, 11.0):
                    candidate = Candidate(
                        f"breakout_{lookback}_sl{stop:g}_tr{trail:g}",
                        "breakout_sensitivity",
                        f"breakout_{lookback}",
                        "below_ema20_cross",
                        stop,
                        trail,
                    )
                    for split in ("train", "validation", "test", "full"):
                        row = metrics(simulate(frames, candidate, split, 5.0))
                        row.update({"lookback": lookback, "stop": stop, "trail": trail})
                        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(RESULTS_DIR / "sensitivity_grid.csv", index=False)
    with (RESULTS_DIR / "bootstrap.json").open("w", encoding="utf-8") as handle:
        json.dump(bootstrap_trade_sequence(best_sim), handle, indent=2)

    fills = pd.DataFrame([asdict(fill) for fill in best_sim.fills])
    fills.to_csv(RESULTS_DIR / "recommended_fills.csv", index=False)
    print(full.sort_values("selection_score", ascending=False)[
        ["candidate", "selection_score", "cagr_pct", "max_drawdown_pct", "sharpe", "entries_per_month", "time_in_market_pct"]
    ].to_string(index=False))
    print("selection_by_train_validation", best_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.download and not args.run:
        raise SystemExit("Choose --download and/or --run")
    if args.download:
        download_bars()
    if args.run:
        run_research()
