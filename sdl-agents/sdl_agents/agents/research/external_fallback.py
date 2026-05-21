"""Shared internal-retrieval gate and external fallback for research specialists."""

from __future__ import annotations

from typing import Any

from sdl_agents.integrations.anthropic_web import query_external_research

MIN_INTERNAL_RELEVANCE_SCORE = 0.18


def internal_context_sufficient(
    query: str, chunks: list[dict[str, Any]], *, min_score: float = MIN_INTERNAL_RELEVANCE_SCORE
) -> bool:
    """Conservative relevance gate before using local RAG/Hermes synthesis."""
    if not chunks:
        return False
    scored = [float(c.get("score") or 0.0) for c in chunks]
    if scored and max(scored) < min_score:
        return False
    return any(str(c.get("text") or "").strip() for c in chunks)


async def external_fallback_response(
    query: str,
    *,
    specialist: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use external web search because internal documents were insufficient."""
    base: dict[str, Any] = {
        "citations": [],
        "concerns": ["internal_sources_insufficient"],
        "external_used": True,
        "specialist": specialist,
        **(extra or {}),
    }
    try:
        result = await query_external_research(query, specialist=specialist)
    except Exception as exc:
        return {
            **base,
            "text": (
                "Internal documents did not contain enough relevant information, "
                f"and external web search failed: {type(exc).__name__}: {exc}"
            ),
            "sources": [],
            "external_used": False,
            "concerns": [
                *base["concerns"],
                f"external_search_failed: {type(exc).__name__}: {exc}",
            ],
        }
    return {
        **base,
        "text": result.get("text", ""),
        "sources": result.get("sources", []),
    }
