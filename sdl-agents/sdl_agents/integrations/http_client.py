"""HTTP client with retries (reference mcp_common spirit)."""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.logging_utils import get_logger

logger = get_logger("http")


def _format_http_exc(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__} (no message)"


def _should_retry(exc: Exception, attempt: int, retries: int) -> bool:
    if attempt >= retries:
        return False
    if isinstance(exc, httpx.TimeoutException):
        return isinstance(exc, httpx.ConnectTimeout)
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def request_with_retry(
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    connect_timeout: float | None = None,
    retries: int = 2,
) -> httpx.Response:
    connect = connect_timeout if connect_timeout is not None else min(10.0, timeout)
    timeout_cfg = httpx.Timeout(connect=connect, read=timeout, write=timeout, pool=connect)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                response = await client.request(method, url, json=json, headers=headers)
                response.raise_for_status()
                return response
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning(
                "HTTP %s %s attempt %s failed: %s",
                method,
                url,
                attempt + 1,
                _format_http_exc(exc),
            )
            if not _should_retry(exc, attempt, retries):
                break
    assert last_exc is not None
    raise last_exc
