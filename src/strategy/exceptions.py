"""Strategy-layer exception hierarchy."""
from __future__ import annotations


class StrategyError(Exception):
    """Root of all strategy-layer errors."""


class InsufficientHistoryError(StrategyError):
    """The bars DataFrame has fewer rows than the engine's required lookback.

    Attributes:
        required: Minimum number of bars the engine needs for warm indicators.
        got:      Number of bars actually provided.
    """

    def __init__(self, msg: str, *, required: int, got: int) -> None:
        super().__init__(msg)
        self.required = required
        self.got = got
