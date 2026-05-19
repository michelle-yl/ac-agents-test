"""Nous Hermes Agent client — OpenAI-compatible Chat Completions (mock or live).

Live mode uses POST /v1/chat/completions per:
https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server
"""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.caveman import merge_system
from sdl_agents.config import (
    HERMES_API_KEY,
    HERMES_BASE_URL,
    HERMES_MODEL,
    REPO_ROOT,
    hermes_http_origin,
    is_live_integration,
)
from sdl_agents.integrations.http_client import request_with_retry
from sdl_agents.logging_utils import get_logger

logger = get_logger("hermes")

# #region agent log
def _agent_dbg(message: str, data: dict[str, Any], hypothesis_id: str) -> None:
    import json
    import time

    try:
        line = (
            json.dumps(
                {
                    "sessionId": "cb4de1",
                    "hypothesisId": hypothesis_id,
                    "location": "hermes_client.py",
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                },
                default=str,
            )
            + "\n"
        )
        (REPO_ROOT / "debug-cb4de1.log").open("a", encoding="utf-8").write(line)
    except OSError:
        pass


# #endregion

_SYSTEM_BY_TASK: dict[str, str] = {
    "academic": (
        "You are an academic literature research assistant. "
        "Summarize peer-reviewed work with citations (DOI/PMID) when available. "
        "Do not fabricate papers."
    ),
    "safety": (
        "You are a laboratory safety specialist. "
        "Answer about biosafety, MSDS, PPE, and OSHA. "
        "When assessing risk, include decision (approved|needs_review|blocked) "
        "and risk_level (low|medium|high|critical) in your reply."
    ),
    "general": "You are a helpful laboratory assistant.",
}


def _chat_completions_url() -> str:
    return f"{HERMES_BASE_URL}/chat/completions"


def _build_messages(prompt: str, context: str, task_type: str) -> list[dict[str, str]]:
    system = merge_system(_SYSTEM_BY_TASK.get(task_type, _SYSTEM_BY_TASK["general"]))
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    user_content = prompt.strip()
    if context.strip():
        user_content = f"{user_content}\n\nContext:\n{context.strip()}"
    messages.append({"role": "user", "content": user_content})
    return messages


def _parse_chat_completion(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Hermes response missing choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError("Hermes response missing message.content")
    text = content if isinstance(content, str) else str(content)
    return {
        "text": text,
        "sources": [],
        "session_id": data.get("id"),
    }


async def run_task(
    prompt: str,
    context: str = "",
    *,
    timeout: float = 120.0,
    task_type: str = "general",
) -> dict[str, Any]:
    """Call Hermes via OpenAI Chat Completions; returns {text, sources, session_id}."""
    if not is_live_integration():
        return _mock_response(prompt, context, task_type)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if HERMES_API_KEY.strip():
        headers["Authorization"] = f"Bearer {HERMES_API_KEY.strip()}"
    # #region agent log
    _agent_dbg(
        "hermes_run_task_pre_post",
        {
            "task_type": task_type,
            "has_authorization_header": "Authorization" in headers,
            "api_key_length": len(HERMES_API_KEY.strip()),
            "chat_url": _chat_completions_url(),
        },
        "H1",
    )
    # #endregion
    if not HERMES_API_KEY.strip():
        raise RuntimeError(
            "Live Hermes requires HERMES_API_KEY. Set it in sdl-agents/.env to the same value as "
            "API_SERVER_KEY in ~/.hermes/.env (see Hermes API Server docs). "
            "Or set SDL_INTEGRATION_MODE=mock to skip live Hermes."
        )
    payload: dict[str, Any] = {
        "model": HERMES_MODEL,
        "messages": _build_messages(prompt, context, task_type),
        "stream": False,
    }
    try:
        response = await request_with_retry(
            "POST",
            _chat_completions_url(),
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        data = response.json()
        return _parse_chat_completion(data)
    except httpx.HTTPStatusError as exc:
        # #region agent log
        _agent_dbg(
            "hermes_http_status_error",
            {"status_code": exc.response.status_code, "url": str(exc.request.url)},
            "H1",
        )
        # #endregion
        if exc.response.status_code == 401:
            raise RuntimeError(
                "Hermes API 401 Unauthorized: HERMES_API_KEY must match API_SERVER_KEY "
                "from the Hermes gateway ~/.hermes/.env. See "
                "https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server"
            ) from exc
        logger.error("Hermes live call failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Hermes live call failed: %s", exc)
        raise


async def health_check() -> bool:
    """GET /health or /v1/health on the gateway origin (non-live returns True)."""
    if not is_live_integration():
        return True
    origin = hermes_http_origin(HERMES_BASE_URL)
    for path in ("/health", "/v1/health"):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{origin}{path}")
                if r.status_code == 200:
                    return True
        except httpx.HTTPError:
            continue
    return False


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
