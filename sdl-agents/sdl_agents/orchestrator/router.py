"""Intent classification adapted from multi-agent-test ROUTER_SYSTEM_PROMPT."""

from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sdl_agents.config import ANTHROPIC_CHAT_MODEL
from sdl_agents.state import Intent

ROUTER_SYSTEM_PROMPT = """You are an intelligent query router for a laboratory information system.

Available routes:
1. **database** — Monitoring replica: device online/offline, sensor alerts, service up/down,
   latest snapshots, device config. Examples: "Which devices are down?", "Is the incubator service up?"

2. **research** — Literature, biosafety (MSDS, BSL, PPE, OSHA), experimental procedures
   (pipetting, dilutions, plate layouts). Examples: "BSL-2 PPE for formaldehyde",
   "Steps for a serial dilution", "Recent papers on organoids"

3. **hybrid** — Needs BOTH live monitoring status AND research/documentation.
   Example: "Is the incubator service up and what are BSL-2 PPE requirements?"

4. **general** — Greetings, meta questions about capabilities, thanks.

Respond with ONLY JSON (no markdown):
{"intent": "database" | "research" | "hybrid" | "general", "reason": "brief explanation"}
"""


class RouteDecision(BaseModel):
    intent: Intent = Field(description="Routing intent")
    reason: str = Field(default="")


def classify_intent_text(query: str) -> RouteDecision:
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
        return _keyword_fallback(query)


def _keyword_fallback(query: str) -> RouteDecision:
    q = query.lower()
    db_kw = ("device", "sensor", "service", "online", "offline", "down", "monitoring", "ip ")
    res_kw = (
        "bsl",
        "msds",
        "ppe",
        "safety",
        "paper",
        "literature",
        "dilution",
        "pipet",
        "protocol",
        "sop",
        "osha",
    )
    has_db = any(k in q for k in db_kw)
    has_res = any(k in q for k in res_kw)
    if has_db and has_res:
        return RouteDecision(intent="hybrid", reason="keyword hybrid")
    if has_db:
        return RouteDecision(intent="database", reason="keyword database")
    if has_res:
        return RouteDecision(intent="research", reason="keyword research")
    return RouteDecision(intent="general", reason="keyword general")
