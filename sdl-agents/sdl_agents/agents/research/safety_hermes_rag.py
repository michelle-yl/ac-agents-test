"""Safety protocols specialist: LlamaIndex retrieval + Hermes synthesis."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from sdl_agents.integrations.hermes_client import run_task
from sdl_agents.integrations.llamaindex_indices import search_safety

SAFETY_PROMPT = """You are a laboratory safety specialist.
Use the retrieved documentation to answer about biosafety, MSDS, PPE, and OSHA requirements.
Respond with clear decision guidance. Include risk_level (low|medium|high|critical)
and decision (approved|needs_review|blocked) when assessing an operation.
"""


async def run_safety(query: str, db_context: str = "") -> dict[str, Any]:
    chunks = search_safety(query, top_k=3)
    context = json.dumps(chunks, indent=2) if chunks else "No local safety documents indexed."
    if db_context:
        context = f"{context}\n\nMonitoring:\n{db_context}"

    prompt = f"{SAFETY_PROMPT}\n\nQuestion: {query}"
    result = await run_task(prompt, context=context, task_type="safety")

    parsed = _parse_safety_fields(result["text"])
    return {
        "text": result["text"],
        "sources": result.get("sources", []) + [
            {"file": c.get("metadata", {}).get("file"), "chunk": c.get("text", "")[:200]}
            for c in chunks
        ],
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
