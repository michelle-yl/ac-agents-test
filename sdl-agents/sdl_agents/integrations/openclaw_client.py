"""OpenClaw gateway client (mock or live).

Live mode uses POST /v1/chat/completions per:
https://docs.openclaw.ai/gateway/openai-http-api
"""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.caveman import instruction_suffix
from sdl_agents.config import (
    OPENCLAW_GATEWAY_TOKEN,
    OPENCLAW_MODEL,
    is_live_integration,
    openclaw_base_urls,
)
from sdl_agents.logging_utils import get_logger

logger = get_logger("openclaw")


def _chat_completions_url(base: str) -> str:
    return f"{base.rstrip('/')}/v1/chat/completions"


def _models_url(base: str) -> str:
    return f"{base.rstrip('/')}/v1/models"


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OPENCLAW_GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {OPENCLAW_GATEWAY_TOKEN}"
    return headers


def _procedure_message(query: str, context: str) -> str:
    body = f"{query}\n\nContext:\n{context}" if context.strip() else query
    return body + instruction_suffix()


def _format_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__} (no message)"


def _parse_chat_completion(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenClaw response missing choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError("OpenClaw response missing message.content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = "\n".join(p for p in parts if p)
    return {
        "text": str(content),
        "skill": data.get("skill") or OPENCLAW_MODEL,
        "ok": True,
    }


async def query_procedures(query: str, context: str = "") -> dict[str, Any]:
    """Ask OpenClaw for procedure guidance."""
    if not is_live_integration():
        return {
            "text": (
                "Mock procedure guidance (SDL_INTEGRATION_MODE=mock). "
                "1) Prepare plate layout. 2) Perform serial dilution. 3) Verify volumes. "
                f"Query: {query[:100]}"
            ),
            "skill": "mock-lab-procedures",
            "ok": True,
        }

    bases = openclaw_base_urls()
    payload = {
        "model": OPENCLAW_MODEL,
        "messages": [{"role": "user", "content": _procedure_message(query, context)}],
    }

    last_err = "no base URLs configured"
    async with httpx.AsyncClient(timeout=30.0) as client:
        for base in bases:
            url = _chat_completions_url(base)
            try:
                response = await client.post(url, headers=_headers(), json=payload)
                if response.status_code == 404:
                    last_err = (
                        "OpenClaw /v1/chat/completions returned 404. Enable "
                        "gateway.http.endpoints.chatCompletions.enabled in "
                        "openclaw.json and restart the gateway container."
                    )
                    break
                response.raise_for_status()
                return _parse_chat_completion(response.json())
            except httpx.ConnectError as exc:
                last_err = _format_error(exc)
                continue
            except Exception as exc:
                last_err = _format_error(exc)
                break

    if "ConnectError" in last_err:
        last_err = (
            f"{last_err}. OpenClaw gateway is not reachable (tried {bases}). "
            "Start the gateway container (e.g. docker compose up -d openclaw-gateway "
            "in your OpenClaw project) and confirm port 18789 is listening."
        )
    logger.warning("OpenClaw unavailable: %s", last_err)
    return {
        "text": "",
        "skill": None,
        "ok": False,
        "error": last_err,
    }


async def health_check() -> bool:
    if not is_live_integration():
        return True
    async with httpx.AsyncClient(timeout=5.0) as client:
        for base in openclaw_base_urls():
            url = _models_url(base)
            try:
                r = await client.get(url, headers=_headers())
                if r.status_code == 200:
                    return True
            except httpx.ConnectError:
                continue
            except httpx.HTTPError:
                return False
    return False
