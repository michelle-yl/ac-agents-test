"""Intent classification adapted from multi-agent-test ROUTER_SYSTEM_PROMPT."""

from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sdl_agents.config import ANTHROPIC_CHAT_MODEL, ROUTER_KEYWORD_FIRST
from sdl_agents.state import Intent

ROUTER_SYSTEM_PROMPT = """You are an intelligent query router for a laboratory information system.

Available routes (match agents.md):
1. **database** (database route) — Monitoring replica: device online/offline, sensor alerts,
   service up/down, latest snapshots. Examples: "Which devices are down?",
   "Is the incubator service up?"

2. **research** (research route) — Academic literature & safety protocols (papers, MSDS, PPE, OSHA)
   OR experimental procedures (pipetting, dilutions, volume calculations, plate layouts).
   Examples: "BSL-2 PPE for formaldehyde", "Steps for a serial dilution",
   "Recent papers on organoids"

3. **hybrid** — Needs BOTH database route (live monitoring) AND research route.
   Example: "Is the incubator service up and what are BSL-2 PPE requirements?"

4. **general** — Greetings, meta questions about capabilities, thanks.

Respond with ONLY JSON (no markdown):
{"intent": "database" | "research" | "hybrid" | "general", "reason": "brief explanation"}
"""

DB_KEYWORDS = (
    "device",
    "devices",
    "sensor",
    "sensors",
    "service",
    "services",
    "online",
    "offline",
    "down",
    "monitoring",
    "ip ",
    "temperature",
    "temp",
    "humidity",
    "reading",
    "status",
    "incubator",
    "cytomat",
    "alert",
    "alerts",
    "snapshot",
    "lab status",
)

RESEARCH_KEYWORDS = (
    "bsl",
    "msds",
    "ppe",
    "safety",
    "paper",
    "literature",
    "dilution",
    "pipet",
    "pipette",
    "protocol",
    "sop",
    "osha",
    "formaldehyde",
    "organoid",
    "volume",
    "concentration",
    "liquid",
    "plate",
)


class RouteDecision(BaseModel):
    intent: Intent = Field(description="Routing intent")
    reason: str = Field(default="")


def classify_intent_keyword(query: str) -> RouteDecision | None:
    """Deterministic routing when keywords clearly match one intent."""
    q = query.lower()
    has_db = any(k in q for k in DB_KEYWORDS)
    has_res = any(k in q for k in RESEARCH_KEYWORDS)
    if has_db and has_res:
        return RouteDecision(intent="hybrid", reason="keyword hybrid")
    if has_db:
        return RouteDecision(intent="database", reason="keyword database")
    if has_res:
        return RouteDecision(intent="research", reason="keyword research")
    if q.strip() in ("hi", "hello", "hey", "thanks", "thank you"):
        return RouteDecision(intent="general", reason="keyword general")
    return None


def classify_intent_text(query: str) -> RouteDecision:
    if ROUTER_KEYWORD_FIRST:
        keyword = classify_intent_keyword(query)
        if keyword is not None:
            return keyword

    model = init_chat_model(ANTHROPIC_CHAT_MODEL, model_provider="anthropic", temperature=0)
    try:
        structured = model.with_structured_output(RouteDecision)
        return structured.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
    except Exception:
        raw = model.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
        text = raw.content if hasattr(raw, "content") else str(raw)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return RouteDecision(**data)
        fallback = classify_intent_keyword(query)
        if fallback is not None:
            return fallback
        return RouteDecision(intent="general", reason="keyword general")


# Backward-compatible alias for tests
def _keyword_fallback(query: str) -> RouteDecision:
    result = classify_intent_keyword(query)
    if result is not None:
        return result
    return RouteDecision(intent="general", reason="keyword general")
