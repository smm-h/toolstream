"""Tests for _request_with_retry() retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from toolstream._direct import _backoff, _request_with_retry


def _ok_response() -> httpx.Response:
    """Build a successful JSON response."""
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}]},
        request=httpx.Request("POST", "https://test/"),
    )


def _error_response(status: int, headers: dict | None = None) -> httpx.Response:
    """Build an error response that raises on raise_for_status()."""
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://test/"),
    )


# --- _backoff ---


def test_backoff_increases():
    b0 = _backoff(0)
    # 2^0 = 1, + jitter [0,1) -> [1.0, 2.0)
    assert 1.0 <= b0 < 2.0
    # 2^3 = 8, + jitter -> [8.0, 9.0)
    b3 = _backoff(3)
    assert 8.0 <= b3 < 9.0


def test_backoff_capped_at_16():
    b10 = _backoff(10)
    # 2^10 = 1024, capped to 16, + jitter -> [16.0, 17.0)
    assert 16.0 <= b10 < 17.0


# --- retry on ReadTimeout ---


@pytest.mark.asyncio
async def test_retry_on_read_timeout():
    """Retries on ReadTimeout and succeeds on 2nd attempt."""
    mock_post = AsyncMock(
        side_effect=[
            httpx.ReadTimeout("read timed out"),
            _ok_response(),
        ],
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = mock_post

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _request_with_retry(
            client, "https://test/", json={}, headers={},
        )

    assert result == {"choices": [{"message": {"content": "ok"}}]}
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


# --- retry on ConnectError ---


@pytest.mark.asyncio
async def test_retry_on_connect_error():
    """Retries on ConnectError and succeeds on 2nd attempt."""
    mock_post = AsyncMock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            _ok_response(),
        ],
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = mock_post

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _request_with_retry(
            client, "https://test/", json={}, headers={},
        )

    assert result == {"choices": [{"message": {"content": "ok"}}]}
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


# --- retry on 429 with Retry-After ---


@pytest.mark.asyncio
async def test_retry_on_429_with_retry_after():
    """Respects Retry-After header on 429, retries, and succeeds."""
    error_resp = _error_response(429, headers={"retry-after": "2"})

    def raise_then_ok():
        calls = []

        async def side_effect(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.HTTPStatusError(
                    "rate limited",
                    request=error_resp.request,
                    response=error_resp,
                )
            return _ok_response()

        return side_effect

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=raise_then_ok())

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _request_with_retry(
            client, "https://test/", json={}, headers={},
        )

    assert result == {"choices": [{"message": {"content": "ok"}}]}
    assert client.post.call_count == 2
    # Should sleep for 2.0 seconds (from Retry-After header)
    mock_sleep.assert_called_once()
    delay = mock_sleep.call_args[0][0]
    assert delay == 2.0


# --- retry on 503 ---


@pytest.mark.asyncio
async def test_retry_on_503():
    """Retries on 503 with backoff and succeeds."""
    error_resp = _error_response(503)

    def raise_then_ok():
        calls = []

        async def side_effect(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.HTTPStatusError(
                    "service unavailable",
                    request=error_resp.request,
                    response=error_resp,
                )
            return _ok_response()

        return side_effect

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=raise_then_ok())

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _request_with_retry(
            client, "https://test/", json={}, headers={},
        )

    assert result == {"choices": [{"message": {"content": "ok"}}]}
    assert client.post.call_count == 2
    mock_sleep.assert_called_once()
    # Backoff for attempt 0: [1.0, 2.0)
    delay = mock_sleep.call_args[0][0]
    assert 1.0 <= delay < 2.0


# --- no retry on 400 ---


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """400 errors raise immediately without retrying."""
    error_resp = _error_response(400)

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "bad request",
            request=error_resp.request,
            response=error_resp,
        ),
    )

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _request_with_retry(
                client, "https://test/", json={}, headers={},
            )

    assert exc_info.value.response.status_code == 400
    assert client.post.call_count == 1
    mock_sleep.assert_not_called()


# --- max retries exhausted ---


@pytest.mark.asyncio
async def test_max_retries_exhausted():
    """Raises after all retry attempts are exhausted."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(
        side_effect=httpx.ReadTimeout("read timed out"),
    )

    with patch("toolstream._direct.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(httpx.ReadTimeout):
            await _request_with_retry(
                client, "https://test/", json={}, headers={},
            )

    # _MAX_RETRIES = 3, so 4 total attempts (0, 1, 2, 3)
    assert client.post.call_count == 4
    # Sleeps between attempts: 3 sleeps (after attempts 0, 1, 2)
    assert mock_sleep.call_count == 3
