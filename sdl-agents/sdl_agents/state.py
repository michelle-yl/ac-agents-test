"""Shared LangGraph state for the SDL multi-agent system."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import MessagesState
from typing_extensions import TypedDict

Intent = Literal["database", "research", "hybrid", "general"]


class ResearchFlags(TypedDict, total=False):
    needs_academic: bool
    needs_safety: bool
    needs_procedures: bool


class ResearchPayload(TypedDict, total=False):
    academic: dict[str, Any] | None
    safety: dict[str, Any] | None
    procedures: dict[str, Any] | None
    gaps: list[str]


class SDLAgentState(MessagesState):
    intent: Intent
    route_reason: str
    db_payload: dict[str, Any] | None
    monitor_cache_used: bool
    research_flags: ResearchFlags
    research_payload: ResearchPayload | None
    errors: list[str]
