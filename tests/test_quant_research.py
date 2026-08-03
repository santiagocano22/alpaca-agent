from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from research.run_etf_research import Candidate, UNIVERSE, _features, simulate


def _synthetic_frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2019-01-02", "2020-01-10", freq="B", tz=UTC)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in UNIVERSE:
        frame = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000.0},
            index=index,
        )
        if symbol == "SPY":
            frame.loc["2020-01-02", ["open", "high", "low", "close"]] = [100, 106, 99, 105]
            frame.loc["2020-01-03", ["open", "high", "low", "close"]] = [110, 111, 107, 108]
            frame.loc["2020-01-06", ["open", "high", "low", "close"]] = [108, 109, 89, 90]
            frame.loc["2020-01-07", ["open", "high", "low", "close"]] = [80, 83, 79, 82]
        frames[symbol] = _features(frame)
    return frames


def test_research_simulator_uses_next_open_and_preserves_opening_gaps() -> None:
    candidate = Candidate(
        "test_breakout",
        "breakout",
        "breakout_20",
        "below_ema20_cross",
        7.0,
        10.0,
        position_pct=10.0,
        max_exposure_pct=10.0,
        max_positions=1,
    )

    result = simulate(_synthetic_frames(), candidate, "train", slippage_bps=5.0)

    spy_fills = [fill for fill in result.fills if fill.symbol == "SPY"]
    assert [(fill.side, fill.trade_date) for fill in spy_fills] == [
        ("BUY", pd.Timestamp("2020-01-03").date()),
        ("SELL", pd.Timestamp("2020-01-07").date()),
    ]
    assert spy_fills[0].price == pytest.approx(110.0 * 1.0005)
    assert spy_fills[1].price == pytest.approx(80.0 * 0.9995)
    assert spy_fills[1].reason == "stop_loss"


def test_research_simulator_never_exceeds_configured_exposure() -> None:
    candidate = Candidate(
        "test_breakout",
        "breakout",
        "breakout_20",
        "below_ema20_cross",
        7.0,
        10.0,
        position_pct=10.0,
        max_exposure_pct=10.0,
        max_positions=1,
    )

    result = simulate(_synthetic_frames(), candidate, "train", slippage_bps=0.0)

    assert result.position_count.max() <= 1
    assert float(result.gross_exposure.max()) <= 0.11
