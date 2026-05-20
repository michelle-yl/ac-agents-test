"""Nous Hermes Agent client — OpenAI-compatible Chat Completions (mock or live).

Live mode uses POST /v1/chat/completions per:
https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from sdl_agents.caveman import merge_system
from sdl_agents.config import (
    HERMES_API_KEY,
    HERMES_CONNECT_TIMEOUT,
    HERMES_MODEL,
    HERMES_READ_TIMEOUT,
    hermes_http_origin,
    hermes_openai_base_urls,
    is_live_integration,
)
from sdl_agents.integrations.http_client import request_with_retry
from sdl_agents.logging_utils import get_logger

logger = get_logger("hermes")

# Hermes gateway handles one agent run at a time; parallel calls starve and hit read timeouts.
_HERMES_GATE_LOCK = asyncio.Lock()
_DEBUG_LOG = Path(__file__).resolve().parents[3] / "debug-69efbc.log"

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
    "procedures": (
        "You are an experimental laboratory procedures specialist. "
        "Answer about pipetting, dilutions, plate layouts, liquid handling, and volume calculations. "
        "Give clear step-by-step guidance when describing protocols."
    ),
    "general": "You are a helpful laboratory assistant.",
}


def _chat_completions_url(openai_base: str) -> str:
    return f"{openai_base.rstrip('/')}/chat/completions"


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


def _format_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__} (no message)"


def _agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "69efbc",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # endregion


async def run_task(
    prompt: str,
    context: str = "",
    *,
    timeout: float | None = None,
    task_type: str = "general",
) -> dict[str, Any]:
    """Call Hermes via OpenAI Chat Completions; returns {text, sources, session_id}."""
    if not is_live_integration():
        return _mock_response(prompt, context, task_type)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if HERMES_API_KEY.strip():
        headers["Authorization"] = f"Bearer {HERMES_API_KEY.strip()}"
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

    read_timeout = HERMES_READ_TIMEOUT if timeout is None else timeout
    bases = hermes_openai_base_urls()
    last_err = "no Hermes base URLs configured"
    async with _HERMES_GATE_LOCK:
        _agent_debug_log(
            hypothesis_id="H1",
            location="hermes_client.py:run_task",
            message="hermes_run_start",
            data={"task_type": task_type, "read_timeout": read_timeout, "bases": bases},
        )
        for openai_base in bases:
            url = _chat_completions_url(openai_base)
            t0 = time.time()
            try:
                response = await request_with_retry(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=read_timeout,
                    connect_timeout=HERMES_CONNECT_TIMEOUT,
                    retries=1,
                )
                elapsed = round(time.time() - t0, 2)
                _agent_debug_log(
                    hypothesis_id="H1",
                    location="hermes_client.py:run_task",
                    message="hermes_run_ok",
                    data={"url": url, "elapsed_s": elapsed, "task_type": task_type},
                    run_id="post-fix",
                )
                return _parse_chat_completion(response.json())
            except httpx.ConnectError as exc:
                last_err = _format_error(exc)
                _agent_debug_log(
                    hypothesis_id="H2",
                    location="hermes_client.py:run_task",
                    message="hermes_connect_error",
                    data={"url": url, "error": last_err},
                )
                continue
            except httpx.TimeoutException as exc:
                last_err = _format_error(exc)
                _agent_debug_log(
                    hypothesis_id="H3",
                    location="hermes_client.py:run_task",
                    message="hermes_timeout",
                    data={"url": url, "error": last_err, "read_timeout": read_timeout},
                )
                raise RuntimeError(
                    f"Hermes timed out after {read_timeout}s ({last_err}). "
                    "The gateway may be busy with another agent run; calls are serialized. "
                    "Increase HERMES_READ_TIMEOUT or retry."
                ) from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    raise RuntimeError(
                        "Hermes API 401 Unauthorized: HERMES_API_KEY must match API_SERVER_KEY "
                        "from the Hermes gateway ~/.hermes/.env. See "
                        "https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server"
                    ) from exc
                logger.error("Hermes live call failed: %s", exc)
                raise
            except Exception as exc:
                last_err = _format_error(exc)
                logger.error("Hermes live call failed: %s", exc)
                raise

    raise RuntimeError(
        f"Hermes unavailable ({last_err}). Tried {bases}. "
        "Start the Hermes API server (hermes gateway; default port 8642) and set "
        "HERMES_API_KEY to match API_SERVER_KEY in ~/.hermes/.env."
    )


async def health_check() -> bool:
    """GET /health or /v1/health on the gateway origin (non-live returns True)."""
    if not is_live_integration():
        return True
    for openai_base in hermes_openai_base_urls():
        origin = hermes_http_origin(openai_base)
        for path in ("/health", "/v1/health"):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(f"{origin}{path}")
                    if r.status_code == 200:
                        return True
            except httpx.ConnectError:
                continue
            except httpx.HTTPError:
                return False
    return False


def _mock_response(prompt: str, context: str, task_type: str) -> dict[str, Any]:
    if task_type == "academic":
        text = (
            "Mock literature summary (integration mode=mock). "
            "Local RAG uses corpus/academic/ when indexed. "
            f"Query excerpt: {prompt[:120]}"
        )
        sources = [{"file": "corpus/academic/mock.pdf", "title": "Mock et al. (2024)"}]
    elif task_type == "procedures":
        text = (
            "Mock procedure steps (integration mode=mock). "
            "Local RAG uses corpus/procedures/ when indexed. "
            f"Query excerpt: {prompt[:120]}"
        )
        sources = [{"file": "corpus/procedures/serial_dilution.md", "chunk_id": "0"}]
    elif task_type == "safety":
        text = (
            "Mock safety synthesis. decision=needs_review; risk_level=medium. "
            f"Context length: {len(context)} chars."
        )
        sources = [{"file": "corpus/safety/mock_bsl2.md", "chunk_id": "0"}]
    else:
        text = f"Mock Hermes reply for task_type={task_type}. {prompt[:120]}"
        sources = []

    return {"text": text, "sources": sources, "session_id": "mock-session"}
