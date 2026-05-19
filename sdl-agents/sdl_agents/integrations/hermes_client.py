"""Nous Hermes Agent client (mock or live)."""

from __future__ import annotations

from typing import Any

from sdl_agents.config import HERMES_API_KEY, HERMES_BASE_URL, is_live_integration
from sdl_agents.integrations.http_client import request_with_retry
from sdl_agents.logging_utils import get_logger

logger = get_logger("hermes")


async def run_task(
    prompt: str,
    context: str = "",
    *,
    timeout: float = 60.0,
    task_type: str = "general",
) -> dict[str, Any]:
    """Run a Hermes task; returns {text, sources, session_id}."""
    if not is_live_integration():
        return _mock_response(prompt, context, task_type)

    headers = {}
    if HERMES_API_KEY:
        headers["Authorization"] = f"Bearer {HERMES_API_KEY}"

    payload = {
        "prompt": prompt,
        "context": context,
        "task_type": task_type,
    }
    try:
        response = await request_with_retry(
            "POST",
            f"{HERMES_BASE_URL}/v1/task",
            json=payload,
            headers=headers or None,
            timeout=timeout,
        )
        data = response.json()
        return {
            "text": data.get("text") or data.get("response", ""),
            "sources": data.get("sources", []),
            "session_id": data.get("session_id"),
        }
    except Exception as exc:
        logger.error("Hermes live call failed: %s", exc)
        raise


def _mock_response(prompt: str, context: str, task_type: str) -> dict[str, Any]:
    if task_type == "academic":
        text = (
            "Mock literature summary (integration mode=mock). "
            "Enable SDL_INTEGRATION_MODE=live and HERMES_BASE_URL for real search. "
            f"Query excerpt: {prompt[:120]}"
        )
        sources = [{"title": "Mock et al. (2024)", "doi": "10.0000/mock"}]
    else:
        text = (
            "Mock safety synthesis. decision=needs_review; risk_level=medium. "
            f"Context length: {len(context)} chars."
        )
        sources = [{"file": "corpus/safety/mock_bsl2.md", "chunk_id": "0"}]

    return {"text": text, "sources": sources, "session_id": "mock-session"}
