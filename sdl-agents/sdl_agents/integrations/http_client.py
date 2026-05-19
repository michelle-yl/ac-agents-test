"""HTTP client with retries (reference mcp_common spirit)."""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.logging_utils import get_logger

logger = get_logger("http")


async def request_with_retry(
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, json=json, headers=headers)
                response.raise_for_status()
                return response
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning("HTTP %s %s attempt %s failed: %s", method, url, attempt + 1, exc)
    assert last_exc is not None
    raise last_exc
