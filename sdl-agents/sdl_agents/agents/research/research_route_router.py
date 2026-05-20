"""Research-route subagent classification (orchestrator research path)."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from sdl_agents.config import ANTHROPIC_GRADER_MODEL, ROUTER_KEYWORD_FIRST
from sdl_agents.state import ResearchFlags

RESEARCH_ROUTE_SYSTEM_PROMPT = """You are the research-route subagent router for a laboratory assistant.

The research route has exactly three Hermes-backed subagents:
1. **academic** — Academic literature (papers, journals, DOI, PubMed, reviews, studies)
2. **safety** — Safety protocols (biosafety, MSDS, PPE, OSHA, BSL, chemical hazards)
3. **procedures** — Experimental procedures and equipment usage (pipetting, dilutions, plate layouts, SOP steps, robotic arm handling, operation of lab equipment)

Set needs_academic, needs_safety, and/or needs_procedures only when clearly required.
Enable multiple flags when the question spans those domains.

Respond with ONLY JSON (no markdown):
{"needs_academic": true|false, "needs_safety": true|false, "needs_procedures": true|false, "reason": "brief explanation"}
"""

ACADEMIC_KEYWORDS = (
    "paper",
    "papers",
    "literature",
    "doi",
    "pubmed",
    "journal",
    "article",
    "organoid",
    "hypothesis",
    "research",
    "study",
    "studies",
    "review",
    "pmid",
)

SAFETY_KEYWORDS = (
    "bsl",
    "msds",
    "ppe",
    "safety",
    "osha",
    "formaldehyde",
    "hazard",
    "biosafety",
    "fume hood",
    "glove",
    "goggles",
    "disinfect",
)

PROCEDURES_KEYWORDS = (
    "dilution",
    "manual",
    "pipet",
    "pipette",
    "plate",
    "volume",
    "sop",
    "protocol",
    "liquid",
    "concentration",
    "serial",
    "96-well",
    "well plate",
    "c1v1",
    "transfer",
    "mix",
)


class ResearchSubagentDecision(BaseModel):
    needs_academic: bool = False
    needs_safety: bool = False
    needs_procedures: bool = False
    reason: str = ""


def classify_research_subagent_keyword(query: str) -> ResearchSubagentDecision | None:
    q = query.lower()
    has_acad = any(k in q for k in ACADEMIC_KEYWORDS)
    has_safe = any(k in q for k in SAFETY_KEYWORDS)
    has_proc = any(k in q for k in PROCEDURES_KEYWORDS)
    if not (has_acad or has_safe or has_proc):
        return None
    return ResearchSubagentDecision(
        needs_academic=has_acad,
        needs_safety=has_safe,
        needs_procedures=has_proc,
        reason="keyword",
    )


def _decision_to_flags(decision: ResearchSubagentDecision) -> ResearchFlags:
    return {
        "needs_academic": decision.needs_academic,
        "needs_safety": decision.needs_safety,
        "needs_procedures": decision.needs_procedures,
    }


def _default_literature_safety_flags() -> ResearchFlags:
    return {
        "needs_academic": True,
        "needs_safety": True,
        "needs_procedures": False,
    }


def classify_research_subagent_text(query: str) -> ResearchFlags:
    if ROUTER_KEYWORD_FIRST:
        keyword = classify_research_subagent_keyword(query)
        if keyword is not None:
            return _decision_to_flags(keyword)

    model = init_chat_model(ANTHROPIC_GRADER_MODEL, model_provider="anthropic", temperature=0)
    try:
        result = model.with_structured_output(ResearchSubagentDecision).invoke(
            [
                SystemMessage(content=RESEARCH_ROUTE_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )
        flags = _decision_to_flags(result)
    except Exception:
        keyword = classify_research_subagent_keyword(query)
        if keyword is not None:
            flags = _decision_to_flags(keyword)
        else:
            flags = _default_literature_safety_flags()

    if not any(flags.values()):
        flags = _default_literature_safety_flags()

    return flags
