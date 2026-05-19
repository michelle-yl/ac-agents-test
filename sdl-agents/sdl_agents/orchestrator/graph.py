"""Top-level SDL orchestrator LangGraph."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from sdl_agents.agents.database.graph import build_database_graph
from sdl_agents.agents.research.graph import build_research_graph
from sdl_agents.config import ANTHROPIC_CHAT_MODEL
from sdl_agents.orchestrator.router import classify_intent_text
from sdl_agents.state import SDLAgentState

_database_graph = None
_research_graph = None


def _get_database_graph():
    global _database_graph
    if _database_graph is None:
        _database_graph = build_database_graph()
    return _database_graph


def _get_research_graph():
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph


def classify_intent(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    decision = classify_intent_text(question)
    return {
        "intent": decision.intent,
        "route_reason": decision.reason,
        "errors": [],
        "db_payload": None,
        "research_payload": None,
        "research_flags": {},
    }


def run_database_subgraph(state: SDLAgentState) -> dict[str, Any]:
    result = _get_database_graph().invoke(state)
    return {
        "messages": result.get("messages", []),
        "db_payload": result.get("db_payload"),
        "errors": list(state.get("errors") or []),
    }


def run_research_subgraph(state: SDLAgentState) -> dict[str, Any]:
    result = _get_research_graph().invoke(state)
    errors = list(state.get("errors") or [])
    errors.extend(result.get("errors") or [])
    return {
        "research_payload": result.get("research_payload"),
        "research_flags": result.get("research_flags", {}),
        "errors": errors,
    }


def general_chat(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    model = init_chat_model(ANTHROPIC_CHAT_MODEL, model_provider="anthropic", temperature=0)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are the SDL lab assistant orchestrator. "
                    "You route database, safety, literature, and procedure questions. "
                    "Briefly explain what you can help with."
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
    if rp.get("academic"):
        a = rp["academic"]
        sections.append("## Academic literature\n" + a.get("text", ""))
        if a.get("sources"):
            sections.append("Sources: " + json.dumps(a["sources"], default=str))

    if rp.get("safety"):
        s = rp["safety"]
        sections.append(
            f"## Safety protocols\n{s.get('text', '')}\n"
            f"(decision={s.get('decision')}, risk_level={s.get('risk_level')})"
        )

    if rp.get("procedures"):
        p = rp["procedures"]
        sections.append("## Experimental procedures\n" + p.get("text", ""))

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


def _route_after_classify(state: SDLAgentState) -> Literal["database", "research", "hybrid", "general"]:
    return state.get("intent", "general")


def build_orchestrator_graph():
    workflow = StateGraph(SDLAgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("database", run_database_subgraph)
    workflow.add_node("research", run_research_subgraph)
    workflow.add_node("general_chat", general_chat)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "database": "database",
            "research": "research",
            "hybrid": "database",
            "general": "general_chat",
        },
    )

    def after_database(state: SDLAgentState) -> str:
        if state.get("intent") == "hybrid":
            return "research"
        return "finalize"

    workflow.add_conditional_edges("database", after_database, {"research": "research", "finalize": "finalize"})
    workflow.add_edge("research", "finalize")
    workflow.add_edge("general_chat", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
