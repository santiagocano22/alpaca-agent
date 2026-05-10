"""Tests for src/utils/market_hours.py.

All functions are pure (take `now` + `calendar`) so tests inject deterministic
data — no mocking of datetime.now() needed.

DST dates used (hardcoded, not relative to today):
  Spring forward 2026: second Sunday = March 8, 2026
    - Friday Mar 6: EST (UTC-5) → 9:30 ET = 14:30 UTC
    - Monday Mar 9: EDT (UTC-4) → 9:30 ET = 13:30 UTC
  Fall back 2026: first Sunday = November 1, 2026
    - Friday Oct 30: EDT (UTC-4) → 9:30 ET = 13:30 UTC
    - Monday Nov 2: EST (UTC-5) → 9:30 ET = 14:30 UTC
"""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.utils.market_hours import (
    CalendarCache,
    MarketDay,
    MarketState,
    get_market_state,
    is_extended_hours_open,
    is_market_open,
    minutes_to_close,
    next_close,
    next_open,
    today_is_holiday,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

_NAIVE = datetime(2026, 5, 11, 9, 30)  # no tzinfo — used by naive-input tests


def et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ── is_market_open ────────────────────────────────────────────────────────────


class TestIsMarketOpen:
    def test_open_mid_session(self, sample_calendar: CalendarCache) -> None:
        assert is_market_open(et(2025, 1, 6, 12, 0), sample_calendar) is True

    def test_open_at_exact_open_boundary(self, sample_calendar: CalendarCache) -> None:
        # [open, close) — open is inclusive
        assert is_market_open(et(2025, 1, 6, 9, 30), sample_calendar) is True

    def test_closed_at_exact_close_boundary(self, sample_calendar: CalendarCache) -> None:
        # close is exclusive
        assert is_market_open(et(2025, 1, 6, 16, 0), sample_calendar) is False

    def test_closed_before_open(self, sample_calendar: CalendarCache) -> None:
        assert is_market_open(et(2025, 1, 6, 9, 29), sample_calendar) is False

    def test_closed_after_close(self, sample_calendar: CalendarCache) -> None:
        assert is_market_open(et(2025, 1, 6, 17, 0), sample_calendar) is False

    def test_closed_on_saturday(self, sample_calendar: CalendarCache) -> None:
        assert is_market_open(et(2025, 1, 11, 12, 0), sample_calendar) is False

    def test_closed_on_sunday(self, sample_calendar: CalendarCache) -> None:
        assert is_market_open(et(2025, 1, 12, 12, 0), sample_calendar) is False

    def test_closed_on_holiday(self, sample_calendar: CalendarCache) -> None:
        # 1 Jan 2025 is New Year's — not in sample_calendar
        assert is_market_open(et(2025, 1, 1, 12, 0), sample_calendar) is False

    def test_utc_input_at_open(self, sample_calendar: CalendarCache) -> None:
        # Jan is EST (UTC-5): 14:30 UTC = 9:30 ET (market just opened)
        assert is_market_open(utc(2025, 1, 6, 14, 30), sample_calendar) is True

    def test_utc_input_before_open(self, sample_calendar: CalendarCache) -> None:
        # 14:29 UTC = 9:29 ET — one minute before open
        assert is_market_open(utc(2025, 1, 6, 14, 29), sample_calendar) is False

    def test_early_close_thanksgiving_eve(self) -> None:
        calendar: CalendarCache = {
            date(2025, 11, 26): MarketDay(date(2025, 11, 26), time(9, 30), time(13, 0)),
        }
        assert is_market_open(et(2025, 11, 26, 12, 59), calendar) is True
        assert is_market_open(et(2025, 11, 26, 13, 0), calendar) is False

    def test_empty_calendar(self) -> None:
        assert is_market_open(et(2025, 1, 6, 12, 0), {}) is False


# ── next_open ─────────────────────────────────────────────────────────────────


class TestNextOpen:
    def test_before_todays_open_returns_today(self, sample_calendar: CalendarCache) -> None:
        result = next_open(et(2025, 1, 6, 8, 0), sample_calendar)
        assert result == et(2025, 1, 6, 9, 30)

    def test_during_session_returns_next_trading_day(self, sample_calendar: CalendarCache) -> None:
        result = next_open(et(2025, 1, 6, 12, 0), sample_calendar)
        assert result == et(2025, 1, 7, 9, 30)

    def test_after_close_returns_next_trading_day(self, sample_calendar: CalendarCache) -> None:
        result = next_open(et(2025, 1, 6, 17, 0), sample_calendar)
        assert result == et(2025, 1, 7, 9, 30)

    def test_friday_after_close_skips_weekend_to_monday(
        self, sample_calendar: CalendarCache
    ) -> None:
        result = next_open(et(2025, 1, 10, 17, 0), sample_calendar)
        assert result == et(2025, 1, 13, 9, 30)

    def test_saturday_skips_to_monday(self, sample_calendar: CalendarCache) -> None:
        result = next_open(et(2025, 1, 11, 10, 0), sample_calendar)
        assert result == et(2025, 1, 13, 9, 30)

    def test_empty_calendar_returns_none(self) -> None:
        assert next_open(et(2025, 1, 6, 12, 0), {}) is None


# ── next_close ────────────────────────────────────────────────────────────────


class TestNextClose:
    def test_during_session_returns_todays_close(self, sample_calendar: CalendarCache) -> None:
        result = next_close(et(2025, 1, 6, 12, 0), sample_calendar)
        assert result == et(2025, 1, 6, 16, 0)

    def test_before_open_returns_todays_close(self, sample_calendar: CalendarCache) -> None:
        result = next_close(et(2025, 1, 6, 8, 0), sample_calendar)
        assert result == et(2025, 1, 6, 16, 0)

    def test_after_close_returns_next_day_close(self, sample_calendar: CalendarCache) -> None:
        result = next_close(et(2025, 1, 6, 17, 0), sample_calendar)
        assert result == et(2025, 1, 7, 16, 0)

    def test_friday_after_close_skips_to_monday_close(
        self, sample_calendar: CalendarCache
    ) -> None:
        result = next_close(et(2025, 1, 10, 17, 0), sample_calendar)
        assert result == et(2025, 1, 13, 16, 0)

    def test_empty_calendar_returns_none(self) -> None:
        assert next_close(et(2025, 1, 6, 12, 0), {}) is None


# ── minutes_to_close ──────────────────────────────────────────────────────────


class TestMinutesToClose:
    def test_exactly_five_minutes_before_close(self, sample_calendar: CalendarCache) -> None:
        result = minutes_to_close(et(2025, 1, 6, 15, 55), sample_calendar)
        assert result == pytest.approx(5.0)

    def test_one_hour_before_close(self, sample_calendar: CalendarCache) -> None:
        result = minutes_to_close(et(2025, 1, 6, 15, 0), sample_calendar)
        assert result == pytest.approx(60.0)

    def test_empty_calendar_returns_none(self) -> None:
        assert minutes_to_close(et(2025, 1, 6, 12, 0), {}) is None


class TestMinutesToCloseSemantics:
    """Verify the contract: positive during WARMUP/ACTIVE, None after close (IDLE)."""

    def test_warmup_returns_positive_minutes_to_todays_close(
        self, sample_calendar: CalendarCache
    ) -> None:
        # 9:25 ET = WARMUP (5 min before open), 6h35m = 395 min until close
        result = minutes_to_close(et(2025, 1, 6, 9, 25), sample_calendar)
        assert result == pytest.approx(395.0)

    def test_active_returns_minutes_to_todays_close(
        self, sample_calendar: CalendarCache
    ) -> None:
        # 9:35 ET = ACTIVE, 6h25m = 385 min until close
        result = minutes_to_close(et(2025, 1, 6, 9, 35), sample_calendar)
        assert result == pytest.approx(385.0)

    def test_idle_after_close_returns_none(self, sample_calendar: CalendarCache) -> None:
        # 16:01 ET = after today's session close — semantically None
        result = minutes_to_close(et(2025, 1, 6, 16, 1), sample_calendar)
        assert result is None

    def test_utc_during_session(self, sample_calendar: CalendarCache) -> None:
        # Jan is EST (UTC-5): 20:00 UTC = 15:00 ET — during session, 60 min to close
        result = minutes_to_close(utc(2025, 1, 6, 20, 0), sample_calendar)
        assert result == pytest.approx(60.0)

    def test_utc_after_close_returns_none(self, sample_calendar: CalendarCache) -> None:
        # 21:01 UTC = 16:01 ET — after close, next close is tomorrow → None
        result = minutes_to_close(utc(2025, 1, 6, 21, 1), sample_calendar)
        assert result is None

    def test_before_open_on_trading_day_is_positive(
        self, sample_calendar: CalendarCache
    ) -> None:
        # 8:00 ET = well before open; today's close is at 16:00 → 480 min
        result = minutes_to_close(et(2025, 1, 6, 8, 0), sample_calendar)
        assert result == pytest.approx(480.0)


# ── get_market_state ──────────────────────────────────────────────────────────


class TestGetMarketState:
    def test_active_mid_session(self, sample_calendar: CalendarCache) -> None:
        assert get_market_state(et(2025, 1, 6, 12, 0), sample_calendar) == MarketState.ACTIVE

    def test_active_at_exact_open(self, sample_calendar: CalendarCache) -> None:
        assert get_market_state(et(2025, 1, 6, 9, 30), sample_calendar) == MarketState.ACTIVE

    def test_warmup_three_minutes_before(self, sample_calendar: CalendarCache) -> None:
        assert (
            get_market_state(et(2025, 1, 6, 9, 27), sample_calendar, warmup_minutes=5)
            == MarketState.WARMUP
        )

    def test_warmup_exactly_at_warmup_boundary(self, sample_calendar: CalendarCache) -> None:
        # 9:25 ET = exactly 5 min before 9:30 → boundary is inclusive (0 < delta <= warmup)
        assert (
            get_market_state(et(2025, 1, 6, 9, 25), sample_calendar, warmup_minutes=5)
            == MarketState.WARMUP
        )

    def test_idle_one_minute_outside_warmup(self, sample_calendar: CalendarCache) -> None:
        # 9:24 ET = 6 min before open — just outside the 5-min warmup window
        assert (
            get_market_state(et(2025, 1, 6, 9, 24), sample_calendar, warmup_minutes=5)
            == MarketState.IDLE
        )

    def test_idle_early_morning(self, sample_calendar: CalendarCache) -> None:
        assert get_market_state(et(2025, 1, 6, 7, 0), sample_calendar) == MarketState.IDLE

    def test_idle_after_close(self, sample_calendar: CalendarCache) -> None:
        assert get_market_state(et(2025, 1, 6, 17, 0), sample_calendar) == MarketState.IDLE

    def test_idle_on_saturday(self, sample_calendar: CalendarCache) -> None:
        assert get_market_state(et(2025, 1, 11, 12, 0), sample_calendar) == MarketState.IDLE

    def test_idle_empty_calendar(self) -> None:
        assert get_market_state(et(2025, 1, 6, 9, 29), {}) == MarketState.IDLE


# ── is_extended_hours_open ────────────────────────────────────────────────────


class TestExtendedHoursOpen:
    def test_open_pre_market(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 6, 5, 0), sample_calendar) is True

    def test_open_during_regular_session(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 6, 12, 0), sample_calendar) is True

    def test_open_post_market(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 6, 17, 0), sample_calendar) is True

    def test_closed_before_session_open(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 6, 3, 59), sample_calendar) is False

    def test_closed_at_session_close_boundary(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 6, 20, 0), sample_calendar) is False

    def test_closed_on_weekend(self, sample_calendar: CalendarCache) -> None:
        assert is_extended_hours_open(et(2025, 1, 11, 10, 0), sample_calendar) is False


# ── today_is_holiday ──────────────────────────────────────────────────────────


class TestTodayIsHoliday:
    def test_weekday_absent_from_calendar_is_holiday(self) -> None:
        # 1 Jan 2025 is Wednesday — not in any calendar
        assert today_is_holiday(et(2025, 1, 1, 12, 0), {}) is True

    def test_saturday_is_not_a_holiday(self) -> None:
        # 4 Jan 2025 is Saturday — weekend, not a holiday
        assert today_is_holiday(et(2025, 1, 4, 12, 0), {}) is False

    def test_sunday_is_not_a_holiday(self) -> None:
        assert today_is_holiday(et(2025, 1, 5, 12, 0), {}) is False

    def test_normal_trading_day_is_not_a_holiday(self, sample_calendar: CalendarCache) -> None:
        assert today_is_holiday(et(2025, 1, 6, 12, 0), sample_calendar) is False

    def test_trading_day_absent_from_empty_calendar_is_holiday(self) -> None:
        assert today_is_holiday(et(2025, 1, 6, 12, 0), {}) is True


# ── NEW: Naive datetime rejection ─────────────────────────────────────────────


class TestNaiveInputRejected:
    """Every public function must raise ValueError for naive datetimes.

    A naive datetime silently treated as local time would be a catastrophic
    and hard-to-debug bug in a trading context.
    """

    def test_is_market_open_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            is_market_open(_NAIVE, sample_calendar)

    def test_is_extended_hours_open_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            is_extended_hours_open(_NAIVE, sample_calendar)

    def test_today_is_holiday_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            today_is_holiday(_NAIVE, sample_calendar)

    def test_next_open_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            next_open(_NAIVE, sample_calendar)

    def test_next_close_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            next_close(_NAIVE, sample_calendar)

    def test_minutes_to_close_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            minutes_to_close(_NAIVE, sample_calendar)

    def test_get_market_state_rejects_naive(self, sample_calendar: CalendarCache) -> None:
        with pytest.raises(ValueError, match="naive"):
            get_market_state(_NAIVE, sample_calendar)


# ── NEW: DST spring forward (March 8, 2026 — second Sunday of March) ─────────


@pytest.fixture
def spring_forward_calendar_2026() -> CalendarCache:
    """Friday Mar 6 (EST/UTC-5) and Monday Mar 9 (EDT/UTC-4) 2026."""
    return {
        date(2026, 3, 6): MarketDay(date(2026, 3, 6), time(9, 30), time(16, 0)),
        date(2026, 3, 9): MarketDay(date(2026, 3, 9), time(9, 30), time(16, 0)),
    }


class TestDSTSpringForward:
    def test_friday_before_dst_uses_est_offset(
        self, spring_forward_calendar_2026: CalendarCache
    ) -> None:
        # Mar 6 is EST: UTC offset = -05:00 → 9:30 ET = 14:30 UTC
        assert et(2026, 3, 6, 9, 30).utcoffset().total_seconds() == -5 * 3600

    def test_monday_after_dst_uses_edt_offset(
        self, spring_forward_calendar_2026: CalendarCache
    ) -> None:
        # Mar 9 is EDT: UTC offset = -04:00 → 9:30 ET = 13:30 UTC
        assert et(2026, 3, 9, 9, 30).utcoffset().total_seconds() == -4 * 3600

    def test_friday_open_at_correct_utc_time(
        self, spring_forward_calendar_2026: CalendarCache
    ) -> None:
        # 14:30 UTC = 9:30 EST: open
        assert is_market_open(utc(2026, 3, 6, 14, 30), spring_forward_calendar_2026) is True
        # 14:29 UTC = 9:29 EST: not yet open
        assert is_market_open(utc(2026, 3, 6, 14, 29), spring_forward_calendar_2026) is False

    def test_monday_open_at_new_utc_time(
        self, spring_forward_calendar_2026: CalendarCache
    ) -> None:
        # 13:30 UTC = 9:30 EDT: open (one hour earlier in UTC than before DST)
        assert is_market_open(utc(2026, 3, 9, 13, 30), spring_forward_calendar_2026) is True
        # 13:29 UTC = 9:29 EDT: not yet open
        assert is_market_open(utc(2026, 3, 9, 13, 29), spring_forward_calendar_2026) is False
        # Friday's old UTC open (14:30) is now 10:30 EDT — still in session
        assert is_market_open(utc(2026, 3, 9, 14, 30), spring_forward_calendar_2026) is True

    def test_next_open_from_friday_close_returns_monday_et_time(
        self, spring_forward_calendar_2026: CalendarCache
    ) -> None:
        # After Friday close, next open is Monday 9:30 EDT
        result = next_open(et(2026, 3, 6, 17, 0), spring_forward_calendar_2026)
        assert result is not None
        assert result.date() == date(2026, 3, 9)
        # The time must be 9:30 ET on Monday (EDT), not 9:30 EST
        assert result == datetime(2026, 3, 9, 9, 30, tzinfo=ET)
        assert result.utcoffset().total_seconds() == -4 * 3600


# ── NEW: DST fall back (November 1, 2026 — first Sunday of November) ─────────


@pytest.fixture
def fall_back_calendar_2026() -> CalendarCache:
    """Friday Oct 30 (EDT/UTC-4) and Monday Nov 2 (EST/UTC-5) 2026."""
    return {
        date(2026, 10, 30): MarketDay(date(2026, 10, 30), time(9, 30), time(16, 0)),
        date(2026, 11, 2): MarketDay(date(2026, 11, 2), time(9, 30), time(16, 0)),
    }


class TestDSTFallBack:
    def test_friday_before_fallback_uses_edt_offset(
        self, fall_back_calendar_2026: CalendarCache
    ) -> None:
        # Oct 30 is EDT: UTC offset = -04:00 → 9:30 ET = 13:30 UTC
        assert et(2026, 10, 30, 9, 30).utcoffset().total_seconds() == -4 * 3600

    def test_monday_after_fallback_uses_est_offset(
        self, fall_back_calendar_2026: CalendarCache
    ) -> None:
        # Nov 2 is EST: UTC offset = -05:00 → 9:30 ET = 14:30 UTC
        assert et(2026, 11, 2, 9, 30).utcoffset().total_seconds() == -5 * 3600

    def test_friday_open_at_edt_utc_time(
        self, fall_back_calendar_2026: CalendarCache
    ) -> None:
        # 13:30 UTC = 9:30 EDT: open
        assert is_market_open(utc(2026, 10, 30, 13, 30), fall_back_calendar_2026) is True
        assert is_market_open(utc(2026, 10, 30, 13, 29), fall_back_calendar_2026) is False

    def test_monday_open_at_est_utc_time(
        self, fall_back_calendar_2026: CalendarCache
    ) -> None:
        # 14:30 UTC = 9:30 EST: open (one hour later in UTC than before fallback)
        assert is_market_open(utc(2026, 11, 2, 14, 30), fall_back_calendar_2026) is True
        assert is_market_open(utc(2026, 11, 2, 14, 29), fall_back_calendar_2026) is False
        # Friday's UTC open (13:30) is now 8:30 EST on Monday — before open
        assert is_market_open(utc(2026, 11, 2, 13, 30), fall_back_calendar_2026) is False

    def test_ambiguous_hour_no_crash_and_returns_idle(
        self, fall_back_calendar_2026: CalendarCache
    ) -> None:
        # Nov 1, 1:30 AM ET is ambiguous: fold=0 is EDT (-04), fold=1 is EST (-05).
        # Market is closed at 1:30 AM regardless; bot must not crash or loop.
        amb_edt = datetime(2026, 11, 1, 1, 30, tzinfo=ET, fold=0)
        amb_est = datetime(2026, 11, 1, 1, 30, tzinfo=ET, fold=1)
        assert is_market_open(amb_edt, fall_back_calendar_2026) is False
        assert is_market_open(amb_est, fall_back_calendar_2026) is False
        assert get_market_state(amb_edt, fall_back_calendar_2026) == MarketState.IDLE
        assert get_market_state(amb_est, fall_back_calendar_2026) == MarketState.IDLE
        # Different UTC representations of the same "clock time"
        assert amb_edt.utcoffset().total_seconds() == -4 * 3600
        assert amb_est.utcoffset().total_seconds() == -5 * 3600


# ── NEW: Holiday skip — explicit next_open tests ──────────────────────────────


class TestHolidaySkip:
    def test_july_4_friday_2025_skips_to_monday(self) -> None:
        # July 4, 2025 is a Friday (Independence Day) — absent from calendar
        calendar: CalendarCache = {
            date(2025, 7, 3): MarketDay(date(2025, 7, 3), time(9, 30), time(16, 0)),  # Thu
            date(2025, 7, 7): MarketDay(date(2025, 7, 7), time(9, 30), time(16, 0)),  # Mon
        }
        july4_noon = datetime(2025, 7, 4, 12, 0, tzinfo=ET)
        assert is_market_open(july4_noon, calendar) is False
        assert next_open(july4_noon, calendar) == datetime(2025, 7, 7, 9, 30, tzinfo=ET)

    def test_good_friday_2026_skips_to_monday(self) -> None:
        # Good Friday 2026 = April 3 (Friday) — absent from calendar
        calendar: CalendarCache = {
            date(2026, 4, 2): MarketDay(date(2026, 4, 2), time(9, 30), time(16, 0)),  # Thu
            date(2026, 4, 6): MarketDay(date(2026, 4, 6), time(9, 30), time(16, 0)),  # Mon
        }
        good_friday = datetime(2026, 4, 3, 12, 0, tzinfo=ET)
        assert is_market_open(good_friday, calendar) is False
        assert next_open(good_friday, calendar) == datetime(2026, 4, 6, 9, 30, tzinfo=ET)

    def test_two_consecutive_holidays_skip_to_correct_day(self) -> None:
        # Christmas 2025 (Thu Dec 25) + Boxing Day (Fri Dec 26) both absent
        # → next open is Mon Dec 29
        calendar: CalendarCache = {
            date(2025, 12, 24): MarketDay(date(2025, 12, 24), time(9, 30), time(13, 0)),  # early close
            date(2025, 12, 29): MarketDay(date(2025, 12, 29), time(9, 30), time(16, 0)),  # Mon
        }
        xmas = datetime(2025, 12, 25, 12, 0, tzinfo=ET)
        assert is_market_open(xmas, calendar) is False
        result = next_open(xmas, calendar)
        assert result == datetime(2025, 12, 29, 9, 30, tzinfo=ET)

    def test_christmas_eve_early_close_then_holiday(self) -> None:
        # Dec 24 closes early at 13:00; Dec 25 is holiday
        calendar: CalendarCache = {
            date(2025, 12, 24): MarketDay(date(2025, 12, 24), time(9, 30), time(13, 0)),
            date(2025, 12, 29): MarketDay(date(2025, 12, 29), time(9, 30), time(16, 0)),
        }
        after_early_close = datetime(2025, 12, 24, 14, 0, tzinfo=ET)
        assert is_market_open(after_early_close, calendar) is False
        result = next_open(after_early_close, calendar)
        assert result == datetime(2025, 12, 29, 9, 30, tzinfo=ET)
