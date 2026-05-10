"""Tests for src/config.py — focuses on the LIVE_TRADING_CONFIRMED safety guard."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum required env vars for Settings() to instantiate."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1234:token")
    monkeypatch.setenv("TELEGRAM_AUTHORIZED_CHAT_ID", "99999")


class TestLiveTradingGuard:
    def test_paper_url_without_confirmation_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_env(monkeypatch)
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "false")

        from src.config import Settings

        s = Settings()
        assert s.is_live is False

    def test_live_url_without_confirmation_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_env(monkeypatch)
        monkeypatch.setenv("ALPACA_BASE_URL", "https://live-api.alpaca.markets")
        monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "false")

        from src.config import Settings

        with pytest.raises(ValidationError, match="LIVE_TRADING_CONFIRMED"):
            Settings()

    def test_live_url_with_confirmation_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_env(monkeypatch)
        monkeypatch.setenv("ALPACA_BASE_URL", "https://live-api.alpaca.markets")
        monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")

        from src.config import Settings

        s = Settings()
        assert s.is_live is True
        assert s.live_trading_confirmed is True

    def test_is_live_property_false_for_paper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_env(monkeypatch)
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "false")

        from src.config import Settings

        assert Settings().is_live is False

    def test_dry_run_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_env(monkeypatch)
        from src.config import Settings

        assert Settings().dry_run is False

    def test_defaults_match_paper_trading_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_env(monkeypatch)
        from src.config import Settings

        s = Settings()
        assert s.alpaca_base_url == "https://paper-api.alpaca.markets/v2"
        assert s.eod_close_minutes_before == 5
        assert s.warmup_minutes_before_open == 5
        assert s.pending_strategy_ttl_seconds == 600
