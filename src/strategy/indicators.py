"""Technical indicator implementations over pandas DataFrames.

All functions are pure: they read from ``df`` and return a new Series or
DataFrame without modifying the input.  NaN values in the output signal that
the indicator has not yet warmed up; callers must handle them explicitly.

DataFrame column contract:
    open, high, low, close, volume  (float)
    index: pd.DatetimeIndex (required only by ``vwap``)

Reference implementations used for ground-truth test values:
    RSI  — Wilder (1978) smoothed via EWM alpha=1/period
    MACD — Appel (1979) standard triple EWM
    ATR  — Wilder (1978) EWM average of True Range
    VWAP — cumulative within each calendar day
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Uses EWM with alpha=1/period (com=period-1) and min_periods=period so
    the first ``period`` rows are NaN while the indicator warms up.
    Returns a Series named "rsi".
    """
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    # Avoid 0/0; avg_loss=0 means pure gains → RS=∞ → RSI=100 (handled by
    # float division: x/0.0=inf, 1+inf=inf, 100/inf=0, 100-0=100)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss was 0 → RSI should be 100 (all gains, no losses)
    result = result.where(avg_loss != 0.0, 100.0)
    # avg_gain was 0 (all losses covered by division giving 0, so result=0) ✓
    return result.rename("rsi")


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Exponential Moving Average using standard span formula (alpha=2/(period+1)).

    Returns a Series named "ema_{period}".
    """
    return df[column].ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Simple Moving Average.  First ``period-1`` values are NaN.

    Returns a Series named "sma_{period}".
    """
    return df[column].rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD indicator (Appel 1979).

    Returns a DataFrame with columns: ``macd``, ``signal``, ``histogram``.
    All use EWM with adjust=False (standard charting convention).
    """
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=df.index,
    )


def bbands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns a DataFrame with columns: ``upper``, ``middle``, ``lower``.
    Uses sample std (ddof=1) and rolling window.  First ``period-1`` values NaN.
    """
    middle = df["close"].rolling(window=period, min_periods=period).mean()
    std = df["close"].rolling(window=period, min_periods=period).std(ddof=1)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower}, index=df.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder).

    True Range = max(H-L, |H-C_prev|, |L-C_prev|).
    Smoothed via EWM alpha=1/period.  Returns a Series named "atr_{period}".
    """
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(com=period - 1, adjust=False).mean().rename(f"atr_{period}")


def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday Volume-Weighted Average Price, resets at midnight each day.

    Requires a tz-aware DatetimeIndex.  Returns a Series named "vwap".
    If the index is not a DatetimeIndex, treats all rows as one session.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = tp * df["volume"]

    result = pd.Series(np.nan, index=df.index, name="vwap")

    if isinstance(df.index, pd.DatetimeIndex):
        dates = np.array(df.index.date)
    else:
        dates = np.zeros(len(df), dtype=int)

    for d in pd.unique(dates):
        mask = dates == d
        cum_tpv = tpv.iloc[mask].cumsum()
        cum_vol = df["volume"].iloc[mask].cumsum()
        result.iloc[mask] = (cum_tpv / cum_vol).values

    return result


def volume_avg(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average volume over ``period`` bars.

    Returns a Series named "volume_avg_{period}".
    """
    return (
        df["volume"]
        .rolling(window=period, min_periods=period)
        .mean()
        .rename(f"volume_avg_{period}")
    )


def breakout(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling N-bar high and low breakout levels.

    Returns a DataFrame with columns: ``high_n``, ``low_n``.
    First ``lookback-1`` values are NaN.
    """
    high_n = df["high"].rolling(window=lookback, min_periods=lookback).max()
    low_n = df["low"].rolling(window=lookback, min_periods=lookback).min()
    return pd.DataFrame({"high_n": high_n, "low_n": low_n}, index=df.index)


def price(df: pd.DataFrame, ohlc: str = "close") -> pd.Series:
    """Return the requested OHLC price column directly.

    ``ohlc`` must be one of: "open", "high", "low", "close".
    Returns a Series named ``ohlc``.
    """
    if ohlc not in ("open", "high", "low", "close"):
        raise ValueError(f"ohlc must be one of open/high/low/close, got {ohlc!r}")
    return df[ohlc].rename(ohlc)
