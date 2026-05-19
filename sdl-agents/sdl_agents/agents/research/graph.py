"""Deep research orchestrator subgraph."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from sdl_agents.agents.research.academic_hermes import run_academic_sync
from sdl_agents.agents.research.experimental_openclaw_rag import run_experimental_sync
from sdl_agents.agents.research.safety_hermes_rag import run_safety_sync
from sdl_agents.config import ANTHROPIC_GRADER_MODEL
from sdl_agents.state import ResearchFlags, ResearchPayload, SDLAgentState


class DecomposeResult(BaseModel):
    needs_academic: bool = False
    needs_safety: bool = False
    needs_procedures: bool = False
    reasoning: str = ""


DECOMPOSE_PROMPT = """Analyze the user question for a lab research assistant.
Set flags for which specialists are needed:
- needs_academic: papers, literature, hypotheses, research methodology
- needs_safety: biosafety, MSDS, PPE, OSHA, chemical hazards, BSL
- needs_procedures: pipetting, dilutions, plate layouts, liquid handling, SOP steps
Only enable flags that are clearly required.
"""


def decompose_question(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    model = init_chat_model(ANTHROPIC_GRADER_MODEL, model_provider="anthropic", temperature=0)
    try:
        result = model.with_structured_output(DecomposeResult).invoke(
            [
                SystemMessage(content=DECOMPOSE_PROMPT),
                HumanMessage(content=question),
            ]
        )
        flags: ResearchFlags = {
            "needs_academic": result.needs_academic,
            "needs_safety": result.needs_safety,
            "needs_procedures": result.needs_procedures,
        }
    except Exception:
        flags = _keyword_decompose(question)

    if not any(flags.values()):
        flags = {"needs_academic": True, "needs_safety": True, "needs_procedures": True}

    return {"research_flags": flags}


def _keyword_decompose(question: str) -> ResearchFlags:
    q = question.lower()
    return {
        "needs_academic": any(k in q for k in ("paper", "literature", "research", "hypothesis")),
        "needs_safety": any(k in q for k in ("bsl", "msds", "ppe", "safety", "osha", "hazard")),
        "needs_procedures": any(
            k in q for k in ("dilution", "pipet", "plate", "volume", "sop", "protocol", "liquid")
        ),
    }


def _db_context(state: SDLAgentState) -> str:
    if state.get("db_payload"):
        return json.dumps(state["db_payload"], default=str)[:2000]
    return ""


def dispatch_specialists(state: SDLAgentState) -> dict[str, Any]:
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
            payload["gaps"].append("academic specialist failed")

    if flags.get("needs_safety"):
        try:
            payload["safety"] = run_safety_sync(question, ctx)
        except Exception as exc:
            errors.append(f"safety: {exc}")
            payload["gaps"].append("safety specialist failed")

    if flags.get("needs_procedures"):
        try:
            proc = run_experimental_sync(question, ctx)
            payload["procedures"] = proc
            if proc.get("errors"):
                errors.extend(proc["errors"])
        except Exception as exc:
            errors.append(f"procedures: {exc}")
            payload["gaps"].append("procedures specialist failed")

    return {"research_payload": payload, "errors": errors}


class GradeResult(BaseModel):
    relevant: Literal["yes", "no"] = "yes"
    note: str = ""


def quality_gate(state: SDLAgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    payload = dict(state.get("research_payload") or {})
    context = json.dumps(payload, default=str)[:4000]

    model = init_chat_model(ANTHROPIC_GRADER_MODEL, model_provider="anthropic", temperature=0)
    try:
        grade = model.with_structured_output(GradeResult).invoke(
            [
                HumanMessage(
                    content=(
                        f"Question: {question}\n\nResearch results:\n{context}\n\n"
                        "Are these results relevant enough to answer the user?"
                    )
                )
            ]
        )
        gaps = list(payload.get("gaps") or [])
        if grade.relevant == "no":
            gaps.append(grade.note or "quality gate: low relevance")
        payload["gaps"] = gaps
    except Exception:
        pass

    return {"research_payload": payload}


def build_research_graph():
    workflow = StateGraph(SDLAgentState)
    workflow.add_node("decompose_question", decompose_question)
    workflow.add_node("dispatch_specialists", dispatch_specialists)
    workflow.add_node("quality_gate", quality_gate)

    workflow.set_entry_point("decompose_question")
    workflow.add_edge("decompose_question", "dispatch_specialists")
    workflow.add_edge("dispatch_specialists", "quality_gate")
    workflow.add_edge("quality_gate", END)
    return workflow.compile()
