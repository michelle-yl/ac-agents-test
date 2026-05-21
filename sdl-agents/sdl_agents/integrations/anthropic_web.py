"""Anthropic Messages API helper for external web-grounded research."""

from __future__ import annotations

from typing import Any

import httpx

from sdl_agents.config import ANTHROPIC_API_KEY, ANTHROPIC_CHAT_MODEL
from sdl_agents.sources import external_source

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"

_SPECIALIST_SYSTEM = {
    "academic": (
        "You are an academic literature research assistant. Use current web sources, "
        "prefer primary literature or reputable scientific sources, and cite URLs."
    ),
    "safety": (
        "You are a laboratory safety specialist. Use current web sources from reputable "
        "safety, regulatory, or institutional pages. Cite URLs and include decision and risk_level."
    ),
    "procedures": (
        "You are an experimental laboratory procedures and equipment specialist. Use current "
        "web sources, preferably manufacturer manuals or official documentation. Cite URLs."
    ),
}


async def query_external_research(query: str, *, specialist: str) -> dict[str, Any]:
    """Answer with Anthropic web search and return external source metadata."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is required for external web fallback")

    payload = {
        "model": ANTHROPIC_CHAT_MODEL,
        "max_tokens": 1200,
        "system": _SPECIALIST_SYSTEM.get(specialist, _SPECIALIST_SYSTEM["academic"]),
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    "Internal documents did not contain enough relevant information. "
                    "Use web search to answer and cite the external sources you used."
                ),
            }
        ],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=payload)
        response.raise_for_status()
    text, sources = _parse_message_response(response.json())
    return {
        "text": text,
        "sources": sources or [external_source("Anthropic web search", prefix="External search")],
        "external_used": True,
    }


def _parse_message_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
            sources.extend(_sources_from_citations(block.get("citations") or []))
        elif block.get("type") == "web_search_tool_result":
            sources.extend(_sources_from_tool_result(block))
    return "\n".join(part for part in text_parts if part).strip(), _dedupe_sources(sources)


def _sources_from_citations(citations: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        url = citation.get("url")
        title = citation.get("title") or url
        if url:
            out.append(external_source(str(url), title=title, url=url))
    return out


def _sources_from_tool_result(block: dict[str, Any]) -> list[dict[str, Any]]:
    content = block.get("content") or []
    if isinstance(content, dict):
        content = content.get("results") or []
    out: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        title = item.get("title") or url
        if url:
            out.append(external_source(str(url), title=title, url=url))
    return out


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for source in sources:
        key = str(source.get("url") or source.get("label"))
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out
