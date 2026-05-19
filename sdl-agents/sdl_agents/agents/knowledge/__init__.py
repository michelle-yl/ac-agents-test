"""Knowledge RAG agent (lazy import to avoid heavy deps on package import)."""

from __future__ import annotations

from typing import Any


def build_knowledge_graph() -> Any:
    from sdl_agents.agents.knowledge.graph import build_knowledge_graph as _build

    return _build()


__all__ = ["build_knowledge_graph"]
