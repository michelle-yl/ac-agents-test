"""Top-level SDL orchestrator LangGraph.

Routes (see agents.md):
- database route: monitor_snapshot cache, then database agent subgraph
- research route: classify subagent, dispatch Hermes specialists, finalize
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from sdl_agents.agents.database.graph import build_database_graph
from sdl_agents.agents.research.academic_hermes import run_academic_sync
from sdl_agents.agents.research.experimental_procedures_hermes import run_procedures_sync
from sdl_agents.agents.research.research_route_router import classify_research_subagent_text
from sdl_agents.agents.research.safety_hermes_rag import run_safety_sync
from sdl_agents.caveman import merge_system
from sdl_agents.config import ANTHROPIC_CHAT_MODEL
from sdl_agents.monitoring.cache import get_state, is_cache_fresh
from sdl_agents.monitoring.cache_answer import answer_from_cache
from sdl_agents.orchestrator.router import classify_intent_text
from sdl_agents.state import ResearchPayload, SDLAgentState

_database_graph = None


def _get_database_graph():
    global _database_graph
    if _database_graph is None:
        _database_graph = build_database_graph()
    return _database_graph


def classify_intent(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    decision = classify_intent_text(question)
    return {
        "intent": decision.intent,
        "route_reason": decision.reason,
        "errors": [],
        "db_payload": None,
        "monitor_cache_used": False,
        "research_payload": None,
        "research_flags": {},
    }


def monitor_snapshot_answer(state: SDLAgentState) -> dict[str, Any]:
    """Database route: answer from hot monitor cache when fresh (0 LLM)."""
    question = state["messages"][-1].content
    cached = get_state()
    if not is_cache_fresh() or cached is None:
        return {}
    payload = answer_from_cache(question, cached)
    if payload is None:
        return {}
    return {
        "db_payload": payload,
        "monitor_cache_used": True,
        "messages": [AIMessage(content=str(payload["answer"]))],
    }


def run_database_subgraph(state: SDLAgentState) -> dict[str, Any]:
    """Database route: LangGraph + PostgreSQL (fast path or LLM)."""
    result = _get_database_graph().invoke(state)
    return {
        "messages": result.get("messages", []),
        "db_payload": result.get("db_payload"),
        "errors": list(state.get("errors") or []),
    }


def classify_research_subagent(state: SDLAgentState) -> dict[str, Any]:
    """Research route: pick academic, safety, and/or procedures specialists."""
    question = state["messages"][-1].content
    flags = classify_research_subagent_text(question)
    return {"research_flags": flags}


def _db_context(state: SDLAgentState) -> str:
    if state.get("db_payload"):
        return json.dumps(state["db_payload"], default=str)[:2000]
    return ""


def dispatch_research_agents(state: SDLAgentState) -> dict[str, Any]:
    """Run enabled Hermes research specialists per research_flags."""
    question = state["messages"][-1].content
    flags = state.get("research_flags", {})
    ctx = _db_context(state)
    errors: list[str] = list(state.get("errors") or [])
    payload: ResearchPayload = {
        "academic": None,
        "safety": None,
        "procedures": None,
        "gaps": [],
    }

    if flags.get("needs_academic"):
        try:
            payload["academic"] = run_academic_sync(question, ctx)
        except Exception as exc:
            errors.append(f"academic: {exc}")
            payload["gaps"].append("academic literature agent failed")

    if flags.get("needs_safety"):
        try:
            payload["safety"] = run_safety_sync(question, ctx)
        except Exception as exc:
            errors.append(f"safety: {exc}")
            payload["gaps"].append("safety protocols agent failed")

    if flags.get("needs_procedures"):
        try:
            payload["procedures"] = run_procedures_sync(question, ctx)
        except Exception as exc:
            errors.append(f"procedures: {exc}")
            payload["gaps"].append("experimental procedures agent failed")

    return {"research_payload": payload, "errors": errors}


def general_chat(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    model = init_chat_model(ANTHROPIC_CHAT_MODEL, model_provider="anthropic", temperature=0)
    response = model.invoke(
        [
            SystemMessage(
                content=merge_system(
                    "You are the SDL lab assistant orchestrator. "
                    "You support the database route (live monitoring, devices, sensors) "
                    "and the research route (academic literature, safety protocols, "
                    "experimental procedures). Briefly explain what you can help with."
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return {"messages": [response]}


def finalize(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][0].content if state["messages"] else ""
    sections: list[str] = []

    if state.get("db_payload"):
        db = state["db_payload"]
        sections.append("## Monitoring database\n" + db.get("answer", json.dumps(db, default=str)))

    rp = state.get("research_payload") or {}
    academic = rp.get("academic") or {}
    if academic.get("text"):
        sections.append("## Academic literature\n" + academic.get("text", ""))
        if academic.get("sources"):
            sections.append("Sources: " + json.dumps(academic["sources"], default=str))

    safety = rp.get("safety") or {}
    if safety.get("text"):
        sections.append(
            f"## Safety protocols\n{safety.get('text', '')}\n"
            f"(decision={safety.get('decision')}, risk_level={safety.get('risk_level')})"
        )

    procedures = rp.get("procedures") or {}
    if procedures.get("text"):
        sections.append("## Experimental procedures\n" + procedures.get("text", ""))

    if rp.get("gaps"):
        sections.append("## Gaps\n" + "; ".join(rp["gaps"]))

    if state.get("errors"):
        sections.append("## Warnings\n" + "; ".join(state["errors"]))

    if not sections:
        last = state["messages"][-1]
        body = last.content if hasattr(last, "content") else str(last)
    else:
        body = f"Answer for: {question}\n\n" + "\n\n".join(sections)

    return {"messages": [AIMessage(content=body)]}


def _route_after_classify(
    state: SDLAgentState,
) -> Literal["monitor_snapshot", "classify_research_subagent", "general_chat"]:
    intent = state.get("intent", "general")
    if intent in ("database", "hybrid"):
        return "monitor_snapshot"
    if intent == "research":
        return "classify_research_subagent"
    return "general_chat"


def _route_after_monitor_cache(
    state: SDLAgentState,
) -> Literal["finalize", "classify_research_subagent", "database"]:
    intent = state.get("intent", "general")
    if state.get("monitor_cache_used"):
        if intent == "hybrid":
            return "classify_research_subagent"
        return "finalize"
    return "database"


def build_orchestrator_graph():
    workflow = StateGraph(SDLAgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("monitor_snapshot", monitor_snapshot_answer)
    workflow.add_node("database", run_database_subgraph)
    workflow.add_node("classify_research_subagent", classify_research_subagent)
    workflow.add_node("dispatch_research_agents", dispatch_research_agents)
    workflow.add_node("general_chat", general_chat)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "monitor_snapshot": "monitor_snapshot",
            "classify_research_subagent": "classify_research_subagent",
            "general_chat": "general_chat",
        },
    )

    workflow.add_conditional_edges(
        "monitor_snapshot",
        _route_after_monitor_cache,
        {
            "finalize": "finalize",
            "classify_research_subagent": "classify_research_subagent",
            "database": "database",
        },
    )

    def after_database(state: SDLAgentState) -> str:
        if state.get("intent") == "hybrid":
            return "classify_research_subagent"
        return "finalize"

    workflow.add_conditional_edges(
        "database",
        after_database,
        {"classify_research_subagent": "classify_research_subagent", "finalize": "finalize"},
    )
    workflow.add_edge("classify_research_subagent", "dispatch_research_agents")
    workflow.add_edge("dispatch_research_agents", "finalize")
    workflow.add_edge("general_chat", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
