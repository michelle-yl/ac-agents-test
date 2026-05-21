"""Safety protocols specialist: LlamaIndex retrieval + Hermes synthesis."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from sdl_agents.agents.research.external_fallback import (
    external_fallback_response,
    internal_context_sufficient,
)
from sdl_agents.integrations.hermes_client import run_task
from sdl_agents.integrations.llamaindex_indices import search_safety
from sdl_agents.sources import local_chunk_source, normalize_source

def _hermes_degraded(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    err = str(exc).lower()
    return any(
        token in err
        for token in ("connect", "unavailable", "timeout", "timed out")
    )


SAFETY_PROMPT = """You are a laboratory safety specialist.
Use the retrieved documentation to answer about biosafety, MSDS, PPE, and OSHA requirements.
Respond with clear decision guidance. Include risk_level (low|medium|high|critical)
and decision (approved|needs_review|blocked) when assessing an operation.
"""


def _rag_only_safety_response(
    query: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Answer from local safety corpus when Hermes gateway is unreachable."""
    if not chunks:
        text = (
            "Hermes gateway unavailable and no local safety documents matched. "
            "Start `hermes gateway` (port 8642) or build safety indices "
            "(python scripts/build_indices.py)."
        )
        decision = "needs_review"
        risk_level = "medium"
    else:
        excerpts = "\n\n".join(
            f"- {c.get('text', '')[:500]}" for c in chunks[:3] if c.get("text")
        )
        text = (
            "From local safety documents (Hermes synthesis unavailable):\n\n"
            f"{excerpts}\n\n"
            "decision=needs_review; risk_level=medium"
        )
        decision = "needs_review"
        risk_level = "medium"
        if "blocked" in text.lower():
            decision = "blocked"

    return {
        "text": text,
        "sources": [local_chunk_source(c) for c in chunks],
        "citations": chunks,
        "decision": decision,
        "risk_level": risk_level,
        "concerns": ["hermes_unavailable"],
        "specialist": "safety",
    }


async def run_safety(query: str, db_context: str = "") -> dict[str, Any]:
    chunks = search_safety(query, top_k=3)
    if not internal_context_sufficient(query, chunks):
        return await external_fallback_response(
            query,
            specialist="safety",
            extra={"decision": "needs_review", "risk_level": "medium"},
        )

    context = json.dumps(chunks, indent=2) if chunks else "No local safety documents indexed."
    if db_context:
        context = f"{context}\n\nMonitoring:\n{db_context}"

    prompt = f"{SAFETY_PROMPT}\n\nQuestion: {query}"
    try:
        result = await run_task(prompt, context=context, task_type="safety")
    except (RuntimeError, httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException) as exc:
        if _hermes_degraded(exc):
            return _rag_only_safety_response(query, chunks)
        raise

    parsed = _parse_safety_fields(result["text"])
    return {
        "text": result["text"],
        "sources": [
            normalize_source(s) for s in result.get("sources", [])
        ] + [local_chunk_source(c) for c in chunks],
        "citations": chunks,
        "decision": parsed.get("decision", "needs_review"),
        "risk_level": parsed.get("risk_level", "medium"),
        "concerns": parsed.get("concerns", []),
        "specialist": "safety",
    }


def _parse_safety_fields(text: str) -> dict[str, Any]:
    import re

    out: dict[str, Any] = {}
    dm = re.search(r"decision[=:\s]+(\w+)", text, re.I)
    rm = re.search(r"risk_level[=:\s]+(\w+)", text, re.I)
    if dm:
        out["decision"] = dm.group(1).lower()
    if rm:
        out["risk_level"] = rm.group(1).lower()
    if "blocked" in text.lower():
        out.setdefault("decision", "blocked")
    return out


def run_safety_sync(query: str, db_context: str = "") -> dict[str, Any]:
    return asyncio.run(run_safety(query, db_context))
