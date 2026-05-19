"""Academic literature specialist via Nous Hermes."""

from __future__ import annotations

import asyncio
from typing import Any

from sdl_agents.integrations.hermes_client import run_task

ACADEMIC_PROMPT = """You are an academic literature research assistant.
Search and summarize peer-reviewed work relevant to the question.
Provide citations with DOI or PMID when available. Do not fabricate papers.
"""


async def run_academic(query: str, db_context: str = "") -> dict[str, Any]:
    prompt = f"{ACADEMIC_PROMPT}\n\nQuestion: {query}"
    if db_context:
        prompt += f"\n\nLab monitoring context:\n{db_context}"
    result = await run_task(prompt, task_type="academic")
    return {
        "text": result["text"],
        "sources": result.get("sources", []),
        "session_id": result.get("session_id"),
        "specialist": "academic",
    }


def run_academic_sync(query: str, db_context: str = "") -> dict[str, Any]:
    return asyncio.run(run_academic(query, db_context))
