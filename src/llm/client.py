"""Anthropic LLM client with retry and usage logging.

Design:
- ``LLMClient.complete()`` is the single public method.  All callers (strategy_parser,
  summarizer) use it by injecting an LLMClient instance, which can be replaced with a
  mock in tests without touching the Anthropic SDK.
- Transient errors (5xx, 529, 429) are retried up to ``max_retries`` times with
  exponential back-off.  4xx errors (except 429) are permanent and are re-raised
  immediately.
- A ``_sleeper`` callable is injected for tests to replace ``asyncio.sleep`` so
  tests run in < 1 ms even when simulating retry delays.
- Token usage (input + output) is logged at DEBUG level after every successful call.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import anthropic
from loguru import logger

# HTTP status codes that are worth retrying.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 529})


class LLMError(Exception):
    """Raised when all retries are exhausted or a permanent error occurs."""


class LLMClient:
    """Thin async wrapper around anthropic.AsyncAnthropic.

    Args:
        api_key:      Anthropic API key.
        max_retries:  Maximum number of retry attempts for transient errors.
                      Set to 0 to disable retries (useful in tests that expect
                      errors on the first call).
        _sleeper:     Async callable used for back-off delay.  Defaults to
                      ``asyncio.sleep``.  Inject a no-op coroutine in tests.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_retries: int = 3,
        _sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._max_retries = max_retries
        self._sleeper = _sleeper or asyncio.sleep

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:
        """Send a single-turn request and return the assistant's text content.

        Retries on transient errors with exponential back-off starting at 1 s
        (1 s, 2 s, 4 s, …).

        Args:
            model:      Anthropic model identifier, e.g. ``"claude-sonnet-4-5"``.
            system:     System prompt (the instruction / persona).
            user:       User message (the input to process).
            max_tokens: Maximum tokens in the response.

        Returns:
            The text of the first content block.

        Raises:
            LLMError: When all retries are exhausted or a non-retryable API
                      error is encountered.
        """
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self._max_retries:
            try:
                message = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                usage = message.usage
                logger.debug(
                    "LLM call model={} input_tokens={} output_tokens={}",
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                )
                # Return the text of the first content block.
                for block in message.content:
                    if hasattr(block, "text"):
                        return block.text
                raise LLMError("Anthropic response contained no text block")

            except anthropic.APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS_CODES:
                    last_exc = exc
                    delay = 2.0 ** attempt  # 1 s, 2 s, 4 s, …
                    logger.warning(
                        "LLM transient error status={} attempt={}/{} retry_in={:.1f}s",
                        exc.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    await self._sleeper(delay)
                    attempt += 1
                    continue
                # Permanent 4xx (except 429 which is in _RETRYABLE) → fail fast.
                raise LLMError(f"Anthropic permanent error: {exc}") from exc

            except anthropic.APIConnectionError as exc:
                last_exc = exc
                delay = 2.0 ** attempt
                logger.warning(
                    "LLM connection error attempt={}/{} retry_in={:.1f}s: {}",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                await self._sleeper(delay)
                attempt += 1
                continue

            except LLMError:
                raise

            except Exception as exc:
                raise LLMError(f"Unexpected error calling Anthropic: {exc}") from exc

        raise LLMError(
            f"LLM call failed after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc
