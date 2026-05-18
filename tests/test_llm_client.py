"""Tests for src/llm/client.py.

The Anthropic SDK is fully mocked — no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from src.llm.client import LLMClient, LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_message(text: str, input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
    """Build a mock anthropic.Message with a text content block."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


def _no_sleep(_: float):
    """Synchronous no-op sleeper; wrapped as a coroutine below."""
    pass


async def _async_no_sleep(_: float) -> None:  # noqa: RUF029
    return None


def _make_client(*, max_retries: int = 3) -> LLMClient:
    return LLMClient("fake-key", max_retries=max_retries, _sleeper=_async_no_sleep)


# ── Successful call ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_returns_text():
    client = _make_client()
    expected = '{"foo": "bar"}'
    client._client.messages.create = AsyncMock(return_value=_make_message(expected))

    result = await client.complete(model="m", system="s", user="u", max_tokens=100)

    assert result == expected


@pytest.mark.asyncio
async def test_complete_logs_token_usage(capfd):
    """Token usage logged at DEBUG; no assertion on exact log output,
    just verify no exception is raised."""
    client = _make_client()
    client._client.messages.create = AsyncMock(
        return_value=_make_message("ok", input_tokens=5, output_tokens=15)
    )
    result = await client.complete(model="m", system="s", user="u", max_tokens=50)
    assert result == "ok"


# ── Retry on transient errors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_500_then_succeeds():
    """First call raises 500, second call succeeds."""
    client = _make_client(max_retries=2)

    error_response = MagicMock()
    error_response.status_code = 500
    transient = anthropic.APIStatusError(
        "Internal Server Error",
        response=error_response,
        body=None,
    )

    success = _make_message("recovered")
    client._client.messages.create = AsyncMock(
        side_effect=[transient, success]
    )

    result = await client.complete(model="m", system="s", user="u", max_tokens=50)
    assert result == "recovered"
    assert client._client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_429_then_succeeds():
    """429 (rate limit) is retryable."""
    client = _make_client(max_retries=2)

    error_response = MagicMock()
    error_response.status_code = 429
    rate_limit = anthropic.RateLimitError(
        "Rate limit exceeded",
        response=error_response,
        body=None,
    )

    success = _make_message("ok after rate limit")
    client._client.messages.create = AsyncMock(side_effect=[rate_limit, success])

    result = await client.complete(model="m", system="s", user="u", max_tokens=50)
    assert result == "ok after rate limit"


@pytest.mark.asyncio
async def test_exhausted_retries_raises_llm_error():
    """After max_retries+1 failures, LLMError is raised."""
    client = _make_client(max_retries=2)

    error_response = MagicMock()
    error_response.status_code = 503
    err = anthropic.APIStatusError("Service Unavailable", response=error_response, body=None)

    client._client.messages.create = AsyncMock(side_effect=err)

    with pytest.raises(LLMError):
        await client.complete(model="m", system="s", user="u", max_tokens=50)

    # Called max_retries+1 = 3 times
    assert client._client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_400_raises_immediately():
    """4xx (except 429) raises LLMError without retrying."""
    client = _make_client(max_retries=3)

    error_response = MagicMock()
    error_response.status_code = 400
    bad_request = anthropic.BadRequestError(
        "Bad request",
        response=error_response,
        body=None,
    )

    client._client.messages.create = AsyncMock(side_effect=bad_request)

    with pytest.raises(LLMError, match="permanent"):
        await client.complete(model="m", system="s", user="u", max_tokens=50)

    # Should NOT retry — called exactly once
    assert client._client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_connection_error():
    """APIConnectionError triggers retry."""
    client = _make_client(max_retries=1)

    conn_err = anthropic.APIConnectionError(request=MagicMock())
    success = _make_message("recovered from conn error")
    client._client.messages.create = AsyncMock(side_effect=[conn_err, success])

    result = await client.complete(model="m", system="s", user="u", max_tokens=50)
    assert result == "recovered from conn error"


# ── Empty content block ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_text_block_raises_llm_error():
    """A message with no text content blocks raises LLMError."""
    client = _make_client(max_retries=0)

    # Content block without 'text' attribute
    block = MagicMock(spec=[])  # no attributes
    msg = MagicMock()
    msg.content = [block]
    msg.usage = MagicMock(input_tokens=1, output_tokens=0)
    client._client.messages.create = AsyncMock(return_value=msg)

    with pytest.raises(LLMError, match="no text block"):
        await client.complete(model="m", system="s", user="u", max_tokens=50)
