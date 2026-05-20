"""LangGraph subgraph for the monitoring database agent."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from sdl_agents.agents.database.fast_path import (
    should_skip_summarize,
    try_fast_path,
)
from sdl_agents.agents.database.formatters import format_rows_answer
from sdl_agents.agents.database.prompts import DB_AGENT_SYSTEM
from sdl_agents.agents.database.tools import DATABASE_TOOLS
from sdl_agents.caveman import with_caveman
from sdl_agents.config import DB_AGENT_MODEL, DB_FAST_PATH_ENABLED
from sdl_agents.state import SDLAgentState

_response_model = None


def _model():
    global _response_model
    if _response_model is None:
        _response_model = init_chat_model(
            DB_AGENT_MODEL, model_provider="anthropic", temperature=0
        )
    return _response_model


def run_fast_path(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    payload = try_fast_path(question)
    if payload is None:
        return {}
    return {
        "messages": [AIMessage(content=payload["answer"])],
        "db_payload": payload,
    }


def _route_fast_path(state: SDLAgentState) -> Literal["plan_query", "__end__"]:
    payload = state.get("db_payload") or {}
    if payload.get("source") == "database_fast_path":
        return "__end__"
    return "plan_query"


def plan_query(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    response = _model().bind_tools(DATABASE_TOOLS).invoke(
        [
            SystemMessage(content=DB_AGENT_SYSTEM),
            HumanMessage(content=question),
        ]
    )
    return {"messages": [response]}


def _template_summarize(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][0].content
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", None) == "tool"]
    rows: list[dict[str, Any]] = []
    for m in tool_msgs:
        try:
            data = json.loads(m.content)
            if isinstance(data, list):
                rows.extend(data)
        except (json.JSONDecodeError, TypeError):
            pass
    tool_name = "get_latest_sensor_status"
    if rows:
        sample = rows[0]
        if "service_name" in sample:
            tool_name = "get_latest_service_status"
        elif "ip" in sample and "sensor_name" not in sample:
            tool_name = "get_latest_device_status"
    answer_text = format_rows_answer(tool_name, rows)
    db_payload = {
        "answer": answer_text,
        "row_count": len(rows),
        "preview_rows": rows[:10],
        "source": "database_template",
        "llm_used": False,
    }
    return {
        "messages": [AIMessage(content=answer_text)],
        "db_payload": db_payload,
    }


def summarize_db(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][0].content
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", None) == "tool"]
    row_count = 0
    for m in tool_msgs:
        try:
            data = json.loads(m.content)
            if isinstance(data, list):
                row_count += len(data)
        except (json.JSONDecodeError, TypeError):
            pass

    if should_skip_summarize(question, len(tool_msgs), row_count):
        return _template_summarize(state)

    context = "\n\n".join(getattr(m, "content", str(m)) for m in tool_msgs)
    if not context:
        context = state["messages"][-1].content

    prompt = (
        f"Question: {question}\n\nTool results:\n{context}\n\n"
        "Write a concise answer for the user. Mention key device/service/sensor names."
    )
    answer = _model().invoke(with_caveman([HumanMessage(content=prompt)]))
    preview: list[dict[str, Any]] = []
    for m in tool_msgs:
        try:
            data = json.loads(m.content)
            if isinstance(data, list):
                preview.extend(data[:5])
        except (json.JSONDecodeError, TypeError):
            pass

    db_payload = {
        "answer": answer.content,
        "row_count": row_count,
        "preview_rows": preview[:10],
        "source": "database_agent",
        "llm_used": True,
    }
    return {
        "messages": [answer],
        "db_payload": db_payload,
    }


def _after_tools(state: SDLAgentState) -> Literal["summarize_db", "plan_query"]:
    last = state["messages"][-1]
    if getattr(last, "type", None) == "tool":
        return "summarize_db"
    return "plan_query"


def build_database_graph():
    workflow = StateGraph(SDLAgentState)
    workflow.add_node("fast_path", run_fast_path)
    workflow.add_node("plan_query", plan_query)
    workflow.add_node("tools", ToolNode(DATABASE_TOOLS))
    workflow.add_node("summarize_db", summarize_db)

    workflow.set_entry_point("fast_path")
    workflow.add_conditional_edges(
        "fast_path",
        _route_fast_path,
        {"plan_query": "plan_query", "__end__": END},
    )
    workflow.add_conditional_edges(
        "plan_query",
        tools_condition,
        {"tools": "tools", "__end__": "summarize_db"},
    )
    workflow.add_conditional_edges("tools", _after_tools)
    workflow.add_edge("summarize_db", END)
    return workflow.compile()
