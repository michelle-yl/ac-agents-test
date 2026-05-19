"""OpenClaw gateway client (mock or live)."""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.caveman import instruction_suffix
from sdl_agents.config import OPENCLAW_BASE_URL, is_live_integration
from sdl_agents.logging_utils import get_logger

logger = get_logger("openclaw")


def _procedure_message(query: str, context: str) -> str:
    body = f"{query}\n\nContext:\n{context}" if context.strip() else query
    return body + instruction_suffix()


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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OPENCLAW_BASE_URL}/api/chat",
                json={"message": _procedure_message(query, context)},
            )
            response.raise_for_status()
            data = response.json()
            return {
                "text": data.get("response") or data.get("text", ""),
                "skill": data.get("skill", "openclaw"),
                "ok": True,
            }
    except Exception as exc:
        logger.warning("OpenClaw unavailable: %s", exc)
        return {
            "text": "",
            "skill": None,
            "ok": False,
            "error": str(exc),
        }


async def health_check() -> bool:
    if not is_live_integration():
        return True
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OPENCLAW_BASE_URL}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
