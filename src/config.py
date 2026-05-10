from __future__ import annotations

from zoneinfo import ZoneInfo

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LIVE_HOST = "live-api.alpaca.markets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Alpaca
    alpaca_api_key: str
    alpaca_api_secret: str
    # /v2 is explicit because alpaca-py's TradingClient uses this URL verbatim
    # when passed via constructor parameter (unlike its own internal default,
    # which appends the version path automatically). Verified against the paper
    # API on 2026-05-09.
    # TODO: confirm whether the live URL also requires /v2 before going live.
    #       Candidate: https://api.alpaca.markets/v2 — not yet verified.
    alpaca_base_url: str = "https://paper-api.alpaca.markets/v2"

    # Anthropic
    anthropic_api_key: str

    # Telegram
    telegram_bot_token: str
    telegram_authorized_chat_id: int

    # Safety
    live_trading_confirmed: bool = False
    dry_run: bool = False

    # Bot behaviour
    eod_close_minutes_before: int = 5
    warmup_minutes_before_open: int = 5
    pending_strategy_ttl_seconds: int = 600
    healthcheck_interval_minutes: int = 15

    # Database
    database_url: str = "sqlite+aiosqlite:///./trading_bot.db"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo("America/New_York")

    @property
    def is_live(self) -> bool:
        return _LIVE_HOST in self.alpaca_base_url

    @model_validator(mode="after")
    def _guard_live_trading(self) -> Settings:
        if self.is_live and not self.live_trading_confirmed:
            raise ValueError(
                f"Live trading URL detected ({self.alpaca_base_url}) but "
                "LIVE_TRADING_CONFIRMED is not 'true'. "
                "Set LIVE_TRADING_CONFIRMED=true in your .env file to proceed. "
                "WARNING: This will use real money."
            )
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
