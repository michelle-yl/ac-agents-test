"""Academic literature specialist: LlamaIndex retrieval + Hermes synthesis."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from sdl_agents.config import HERMES_READ_TIMEOUT
from sdl_agents.integrations.hermes_client import run_task
from sdl_agents.integrations.llamaindex_indices import search_academic

ACADEMIC_READ_TIMEOUT = min(HERMES_READ_TIMEOUT, 90.0)

ACADEMIC_PROMPT = """You are an academic literature research assistant.
Use the retrieved documentation to summarize work relevant to the question.
Provide citations with DOI or PMID when available. Do not fabricate papers.
"""


def _hermes_degraded(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    err = str(exc).lower()
    return any(
        token in err
        for token in ("connect", "unavailable", "timeout", "timed out")
    )


def _rag_only_academic_response(
    query: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Answer from local academic corpus when Hermes gateway is unreachable."""
    if not chunks:
        text = (
            "Hermes gateway unavailable and no local academic documents matched. "
            "Add PDFs under corpus/academic/ and run "
            "python scripts/build_indices.py."
        )
    else:
        excerpts = "\n\n".join(
            f"- {c.get('text', '')[:500]}" for c in chunks[:3] if c.get("text")
        )
        text = (
            "From local academic documents (Hermes synthesis unavailable):\n\n"
            f"{excerpts}"
        )

    return {
        "text": text,
        "sources": [
            {"file": c.get("metadata", {}).get("file"), "chunk": c.get("text", "")[:200]}
            for c in chunks
        ],
        "citations": chunks,
        "concerns": ["hermes_unavailable"],
        "specialist": "academic",
    }


async def run_academic(query: str, db_context: str = "") -> dict[str, Any]:
    chunks = search_academic(query, top_k=3)
    context = json.dumps(chunks, indent=2) if chunks else "No local academic documents indexed."
    if db_context:
        context = f"{context}\n\nMonitoring:\n{db_context}"

    prompt = f"{ACADEMIC_PROMPT}\n\nQuestion: {query}"
    try:
        result = await run_task(
            prompt,
            context=context,
            task_type="academic",
            timeout=ACADEMIC_READ_TIMEOUT,
        )
    except (RuntimeError, httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException) as exc:
        if _hermes_degraded(exc):
            return _rag_only_academic_response(query, chunks)
        raise

    return {
        "text": result["text"],
        "sources": result.get("sources", [])
        + [
            {"file": c.get("metadata", {}).get("file"), "chunk": c.get("text", "")[:200]}
            for c in chunks
        ],
        "citations": chunks,
        "session_id": result.get("session_id"),
        "specialist": "academic",
    }


def run_academic_sync(query: str, db_context: str = "") -> dict[str, Any]:
    return asyncio.run(run_academic(query, db_context))
