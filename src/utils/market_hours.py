"""Market hours utilities.

All public functions are pure: they accept a timezone-aware `now` datetime and
a CalendarCache dict built from Alpaca's /v2/calendar endpoint. No side effects.
The scheduler is responsible for keeping the cache fresh (refresh daily at 00:05 ET).

Rules:
- All times stored in MarketDay are in America/New_York (ET).
- All public functions raise ValueError if given a naive (no tzinfo) datetime.
  Never assume system local time: a silent TZ bug in a trading bot can cost real money.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# date → MarketDay mapping built from Alpaca's calendar API
CalendarCache = dict[date, "MarketDay"]


@dataclass
class MarketDay:
    """Schedule for a single trading day (all times in ET)."""

    date: date
    open: time   # regular session open
    close: time  # regular session close
    # Extended-hours defaults match Alpaca's standard pre/post market window
    session_open: time = field(default_factory=lambda: time(4, 0))
    session_close: time = field(default_factory=lambda: time(20, 0))


class MarketState(enum.Enum):
    IDLE = "IDLE"      # outside trading hours, no evaluation
    WARMUP = "WARMUP"  # N minutes before open, pre-heating indicators
    ACTIVE = "ACTIVE"  # regular session open, evaluating rules


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ensure_aware(dt: datetime) -> datetime:
    """Validate that `dt` is timezone-aware and convert to ET.

    Raises ValueError for naive datetimes instead of silently assuming system
    local time — a silent TZ assumption in a trading bot is a liability.
    """
    if dt.tzinfo is None:
        raise ValueError(
            f"market_hours requires a timezone-aware datetime, got naive: {dt!r}. "
            "Use datetime.now(UTC) or datetime(..., tzinfo=ZoneInfo('America/New_York'))."
        )
    return dt.astimezone(ET)


# ── Core predicates ───────────────────────────────────────────────────────────


def is_market_open(now: datetime, calendar: CalendarCache) -> bool:
    """True if the regular session is currently open [open, close)."""
    now_et = _ensure_aware(now)
    market_day = calendar.get(now_et.date())
    if market_day is None:
        return False
    open_dt = datetime.combine(now_et.date(), market_day.open, tzinfo=ET)
    close_dt = datetime.combine(now_et.date(), market_day.close, tzinfo=ET)
    return open_dt <= now_et < close_dt


def is_extended_hours_open(now: datetime, calendar: CalendarCache) -> bool:
    """True if the extended session (pre/post market) is currently open [session_open, session_close)."""
    now_et = _ensure_aware(now)
    market_day = calendar.get(now_et.date())
    if market_day is None:
        return False
    s_open = datetime.combine(now_et.date(), market_day.session_open, tzinfo=ET)
    s_close = datetime.combine(now_et.date(), market_day.session_close, tzinfo=ET)
    return s_open <= now_et < s_close


def today_is_holiday(now: datetime, calendar: CalendarCache) -> bool:
    """True if today is a weekday that is absent from the calendar (i.e. a market holiday).

    Weekends return False — they are expected non-trading days, not holidays.
    """
    now_et = _ensure_aware(now)
    today = now_et.date()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return today not in calendar


# ── Look-ahead helpers ────────────────────────────────────────────────────────


def next_open(now: datetime, calendar: CalendarCache) -> Optional[datetime]:
    """Next regular session open strictly after `now` (ET-aware datetime).

    If the market is currently open, returns the *next* trading day's open, not today's.
    Returns None when the calendar has no data far enough ahead (> 60 days).
    """
    now_et = _ensure_aware(now)
    today = now_et.date()
    for i in range(60):
        d = today + timedelta(days=i)
        market_day = calendar.get(d)
        if market_day is None:
            continue
        open_dt = datetime.combine(d, market_day.open, tzinfo=ET)
        if open_dt > now_et:
            return open_dt
    return None


def next_close(now: datetime, calendar: CalendarCache) -> Optional[datetime]:
    """Next regular session close strictly after `now` (ET-aware datetime).

    Returns None when no data is available far enough ahead (> 60 days).
    """
    now_et = _ensure_aware(now)
    today = now_et.date()
    for i in range(60):
        d = today + timedelta(days=i)
        market_day = calendar.get(d)
        if market_day is None:
            continue
        close_dt = datetime.combine(d, market_day.close, tzinfo=ET)
        if close_dt > now_et:
            return close_dt
    return None


def minutes_to_close(now: datetime, calendar: CalendarCache) -> Optional[float]:
    """Minutes until today's regular session close.

    Returns a positive float while the current ET calendar day still has an
    upcoming close (including WARMUP, before the session opens).
    Returns None once today's close has passed — calling this in IDLE state
    (after 16:00 ET) has no useful meaning; callers should check first.
    """
    now_et = _ensure_aware(now)
    nxt = next_close(now, calendar)
    if nxt is None:
        return None
    # Only meaningful when the next close is still today (ET date)
    if nxt.date() != now_et.date():
        return None
    return (nxt - now).total_seconds() / 60


# ── State machine ─────────────────────────────────────────────────────────────


def get_market_state(
    now: datetime,
    calendar: CalendarCache,
    warmup_minutes: int = 5,
) -> MarketState:
    """Return the current bot state (IDLE / WARMUP / ACTIVE).

    WARMUP is entered when the next open is within `warmup_minutes` of `now`.
    ACTIVE is entered at the exact open and exits at the exact close.
    IDLE covers all other times (nights, weekends, holidays, outside warmup window).
    """
    _ensure_aware(now)  # validate once; internal functions re-validate but that's fine
    if is_market_open(now, calendar):
        return MarketState.ACTIVE

    nxt_open = next_open(now, calendar)
    if nxt_open is not None:
        delta_minutes = (nxt_open - now).total_seconds() / 60
        if 0 < delta_minutes <= warmup_minutes:
            return MarketState.WARMUP

    return MarketState.IDLE
