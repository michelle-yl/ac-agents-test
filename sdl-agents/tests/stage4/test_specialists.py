"""Stage 4: router and specialists (mocked integrations)."""

from __future__ import annotations

import pytest
from langchain.messages import HumanMessage

from sdl_agents.agents.research.academic_hermes import run_academic_sync
from sdl_agents.agents.research.experimental_procedures_hermes import (
    dilution_calc,
    run_procedures_sync,
)
from sdl_agents.agents.research.research_route_router import classify_research_subagent_text
from sdl_agents.agents.research.safety_hermes_rag import run_safety_sync
from sdl_agents.orchestrator.graph import build_orchestrator_graph
from sdl_agents.orchestrator.router import (
    _keyword_fallback,
    classify_intent_text,
)


@pytest.mark.stage4
def test_router_database_keyword():
    d = _keyword_fallback("Which devices are offline in the latest snapshot?")
    assert d.intent == "database"


@pytest.mark.stage4
def test_router_research_keyword():
    d = _keyword_fallback("What PPE is required for BSL-2 formaldehyde work?")
    assert d.intent == "research"


@pytest.mark.stage4
def test_router_research_equipment_manual_keyword():
    d = _keyword_fallback("How do I interpret the robotic arm equipment manual?")
    assert d.intent == "research"


@pytest.mark.stage4
def test_router_hybrid_keyword():
    d = _keyword_fallback("Is the incubator service up and what are BSL-2 PPE requirements?")
    assert d.intent == "hybrid"


@pytest.mark.stage4
def test_academic_mock():
    result = run_academic_sync("organoid culture literature")
    assert "text" in result
    assert result.get("sources")
    assert result["sources"][0]["source_type"] == "internal"


@pytest.mark.stage4
def test_safety_mock():
    result = run_safety_sync("BSL-2 PPE for formaldehyde")
    assert result.get("decision")
    assert result.get("risk_level")


@pytest.mark.stage4
def test_procedures_mock():
    result = run_procedures_sync("serial dilution 96-well plate")
    assert "text" in result
    assert result.get("specialist") == "procedures"


@pytest.mark.stage4
def test_safety_rag_fallback_when_hermes_unavailable(monkeypatch):
    async def _fail_hermes(*_args, **_kwargs):
        raise RuntimeError(
            "Hermes unavailable (ConnectError: All connection attempts failed). "
            "Tried ['http://127.0.0.1:8642/v1']."
        )

    monkeypatch.setattr(
        "sdl_agents.agents.research.safety_hermes_rag.run_task",
        _fail_hermes,
    )
    result = run_safety_sync("what ppe for bsl2?")
    assert "Lab coat" in result["text"]
    assert result.get("decision")


@pytest.mark.stage4
def test_academic_rag_fallback_when_hermes_unavailable(monkeypatch):
    async def _fail_hermes(*_args, **_kwargs):
        raise RuntimeError(
            "Hermes unavailable (ConnectError: All connection attempts failed). "
            "Tried ['http://127.0.0.1:8642/v1']."
        )

    monkeypatch.setattr(
        "sdl_agents.agents.research.academic_hermes.run_task",
        _fail_hermes,
    )
    monkeypatch.setattr(
        "sdl_agents.agents.research.academic_hermes.search_academic",
        lambda _q, top_k=3: [
            {
                "text": "Bioprinting scaffold methods for tissue engineering.",
                "score": 0.9,
                "metadata": {"file": "Engineered_Assistive_Bioprinting.pdf"},
            }
        ],
    )
    result = run_academic_sync("bioprinting tissue engineering")
    assert "Bioprinting" in result["text"]
    assert result.get("concerns") == ["hermes_unavailable"]


@pytest.mark.stage4
def test_procedures_rag_fallback_when_hermes_unavailable(monkeypatch):
    async def _fail_hermes(*_args, **_kwargs):
        raise RuntimeError("Hermes timed out after 90s")

    monkeypatch.setattr(
        "sdl_agents.agents.research.experimental_procedures_hermes.run_task",
        _fail_hermes,
    )
    monkeypatch.setattr(
        "sdl_agents.agents.research.experimental_procedures_hermes.search_procedures",
        lambda _q, top_k=3: [
            {
                "text": "Serial dilution: transfer 10 uL stock into 90 uL diluent per well.",
                "score": 0.9,
                "metadata": {"file": "serial_dilution.md"},
            }
        ],
    )
    result = run_procedures_sync("serial dilution steps")
    assert "Serial dilution" in result["text"]
    assert result.get("concerns") == ["hermes_unavailable"]


@pytest.mark.stage4
@pytest.mark.parametrize(
    ("search_path", "runner", "query"),
    [
        (
            "sdl_agents.agents.research.academic_hermes.search_academic",
            run_academic_sync,
            "new paper on automated organoid culture",
        ),
        (
            "sdl_agents.agents.research.safety_hermes_rag.search_safety",
            run_safety_sync,
            "new formaldehyde safety guidance",
        ),
        (
            "sdl_agents.agents.research.experimental_procedures_hermes.search_procedures",
            run_procedures_sync,
            "how to use a UR5 robot arm",
        ),
    ],
)
def test_research_external_fallback_when_internal_empty(monkeypatch, search_path, runner, query):
    async def _web_answer(_query: str, *, specialist: str):
        return {
            "text": f"External {specialist} answer",
            "sources": [
                {
                    "source_type": "external",
                    "label": "https://example.com/source",
                    "url": "https://example.com/source",
                }
            ],
            "external_used": True,
        }

    monkeypatch.setattr(search_path, lambda _q, top_k=3: [])
    monkeypatch.setattr(
        "sdl_agents.agents.research.external_fallback.query_external_research",
        _web_answer,
    )

    result = runner(query)
    assert result["external_used"] is True
    assert result["sources"][0]["source_type"] == "external"
    assert "https://example.com/source" in result["sources"][0]["label"]
    assert not any(s.get("source_type") == "internal" for s in result["sources"])


@pytest.mark.stage4
def test_orchestrator_external_fallback_sources_render(monkeypatch):
    async def _web_answer(_query: str, *, specialist: str):
        return {
            "text": "External safety answer",
            "sources": [
                {
                    "source_type": "external",
                    "label": "https://example.com/safety",
                    "url": "https://example.com/safety",
                }
            ],
            "external_used": True,
        }

    monkeypatch.setattr(
        "sdl_agents.agents.research.safety_hermes_rag.search_safety",
        lambda _q, top_k=3: [],
    )
    monkeypatch.setattr(
        "sdl_agents.agents.research.external_fallback.query_external_research",
        _web_answer,
    )
    graph = build_orchestrator_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="what ppe for new formaldehyde guidance?")],
            "intent": "general",
            "route_reason": "",
            "db_payload": None,
            "monitor_cache_used": False,
            "research_payload": None,
            "research_flags": {},
            "errors": [],
        }
    )
    body = result["messages"][-1].content
    assert "## Sources" in body
    assert "Internal sources: none" in body
    assert "https://example.com/safety" in body


@pytest.mark.stage4
def test_research_flags_safety_only():
    flags = classify_research_subagent_text("what ppe for bsl2?")
    assert flags["needs_safety"] is True
    assert flags["needs_academic"] is False
    assert flags["needs_procedures"] is False


@pytest.mark.stage4
def test_orchestrator_research_safety_only_payload():
    graph = build_orchestrator_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="what ppe for bsl2?")],
            "intent": "general",
            "route_reason": "",
            "db_payload": None,
            "monitor_cache_used": False,
            "research_payload": None,
            "research_flags": {},
            "errors": [],
        }
    )
    rp = result.get("research_payload") or {}
    assert rp.get("safety") is not None
    assert rp.get("academic") is None
    body = result["messages"][-1].content
    assert "## Safety protocols" in body
    assert "## Academic literature" not in body
    assert "## Sources" in body
    assert "Internal sources:" in body


@pytest.mark.stage4
def test_dilution_math():
    out = dilution_calc(10.0, 100.0, c2=1.0)
    assert out["v2"] == pytest.approx(1000.0)


@pytest.mark.stage4
@pytest.mark.slow
def test_router_llm():
    d = classify_intent_text("List down services from monitoring")
    assert d.intent in ("database", "research", "hybrid", "general")
