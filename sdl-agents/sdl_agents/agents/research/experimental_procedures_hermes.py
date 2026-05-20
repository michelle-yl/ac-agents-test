"""Experimental procedures specialist: LlamaIndex retrieval + Hermes synthesis."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from sdl_agents.config import HERMES_READ_TIMEOUT
from sdl_agents.integrations.hermes_client import run_task
from sdl_agents.integrations.llamaindex_indices import search_procedures

PROCEDURES_READ_TIMEOUT = min(HERMES_READ_TIMEOUT, 90.0)

PROCEDURES_PROMPT = """You are an experimental laboratory procedures specialist.
Use the retrieved SOPs and procedure documents to answer about pipetting, dilutions,
plate layouts, liquid handling, and volume calculations. Include clear step-by-step guidance.
"""


def dilution_calc(
    c1: float, v1: float, c2: float | None = None, v2: float | None = None
) -> dict[str, float]:
    """C1V1=C2V2 — provide three values to solve for the fourth."""
    if c2 is None and v2 is not None:
        c2 = c1 * v1 / v2
    elif v2 is None and c2 is not None:
        v2 = c1 * v1 / c2
    elif c2 is not None and v2 is not None:
        pass
    else:
        raise ValueError("Provide three of c1, v1, c2, v2")
    return {"c1": c1, "v1": v1, "c2": c2, "v2": v2}


def _extract_dilution_from_query(query: str) -> dict[str, float] | None:
    nums = re.findall(r"(\d+\.?\d*)", query)
    if len(nums) >= 3 and "dilut" in query.lower():
        c1, v1, c2 = float(nums[0]), float(nums[1]), float(nums[2])
        return dilution_calc(c1, v1, c2=c2)
    return None


def _hermes_degraded(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    err = str(exc).lower()
    return any(
        token in err
        for token in ("connect", "unavailable", "timeout", "timed out")
    )


def _rag_only_procedures_response(
    query: str, chunks: list[dict[str, Any]], dilution: dict[str, float] | None
) -> dict[str, Any]:
    parts: list[str] = []
    if not chunks:
        parts.append(
            "Hermes gateway unavailable and no local procedure documents matched. "
            "Add SOPs under corpus/procedures/ and run python scripts/build_indices.py."
        )
    else:
        excerpts = "\n\n".join(
            f"- {c.get('text', '')[:500]}" for c in chunks[:3] if c.get("text")
        )
        parts.append(
            "From local procedure documents (Hermes synthesis unavailable):\n\n"
            f"{excerpts}"
        )
    if dilution:
        parts.append(f"Dilution (C1V1=C2V2): {json.dumps(dilution)}")
    return {
        "text": "\n\n".join(parts),
        "sources": [
            {"file": c.get("metadata", {}).get("file"), "chunk": c.get("text", "")[:200]}
            for c in chunks
        ],
        "citations": chunks,
        "dilution": dilution,
        "concerns": ["hermes_unavailable"],
        "specialist": "procedures",
    }


async def run_procedures(query: str, db_context: str = "") -> dict[str, Any]:
    chunks = search_procedures(query, top_k=3)
    context = json.dumps(chunks, indent=2) if chunks else "No local procedure documents indexed."
    dilution = _extract_dilution_from_query(query)
    if dilution:
        context += f"\n\nDilution calculation (C1V1=C2V2): {json.dumps(dilution)}"
    if db_context:
        context = f"{context}\n\nMonitoring:\n{db_context}"

    prompt = f"{PROCEDURES_PROMPT}\n\nQuestion: {query}"
    try:
        result = await run_task(
            prompt,
            context=context,
            task_type="procedures",
            timeout=PROCEDURES_READ_TIMEOUT,
        )
    except (RuntimeError, httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException) as exc:
        if _hermes_degraded(exc):
            return _rag_only_procedures_response(query, chunks, dilution)
        raise

    return {
        "text": result["text"],
        "sources": result.get("sources", [])
        + [
            {"file": c.get("metadata", {}).get("file"), "chunk": c.get("text", "")[:200]}
            for c in chunks
        ],
        "citations": chunks,
        "dilution": dilution,
        "specialist": "procedures",
    }


def run_procedures_sync(query: str, db_context: str = "") -> dict[str, Any]:
    return asyncio.run(run_procedures(query, db_context))
