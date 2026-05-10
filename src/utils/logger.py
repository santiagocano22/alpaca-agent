from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "logs") -> None:
    logger.remove()

    # Force all record timestamps to UTC so logs are consistent regardless of the
    # host machine's local timezone (critical when running on a VM in any region).
    logger.configure(
        patcher=lambda record: record.update({"time": record["time"].astimezone(UTC)})
    )

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}Z</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    )
    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS}Z | {level: <8} | {name}:{line} — {message}"
    )

    logger.add(sys.stdout, level=log_level, format=fmt_console, colorize=True)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.add(
        f"{log_dir}/trading_bot.log",
        level=log_level,
        format=fmt_file,
        rotation="00:00",
        retention="30 days",
        compression="gz",
    )

    # Machine-parseable structured log.
    # Each line is a JSON object. Key fields for parsing / alerting:
    #   .record.time.repr        → ISO 8601 UTC timestamp (e.g. "2026-05-09T14:30:00+00:00")
    #   .record.time.timestamp   → Unix epoch float
    #   .record.level.name       → "INFO" / "WARNING" / "ERROR" / ...
    #   .record.name             → module path (e.g. "src.broker.alpaca_client")
    #   .record.message          → the log message string
    #   .record.extra            → structured context dict, populated via logger.bind(key=val)
    #                              e.g. logger.bind(symbol="QQQ", order_id="abc").info("filled")
    logger.add(
        f"{log_dir}/trading_bot.json",
        level=log_level,
        serialize=True,
        rotation="100 MB",
        retention="30 days",
        compression="gz",
    )
