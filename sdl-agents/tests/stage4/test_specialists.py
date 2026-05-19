"""Stage 4: router and specialists (mocked integrations)."""

from __future__ import annotations

import pytest

from sdl_agents.agents.research.experimental_openclaw_rag import dilution_calc, run_experimental_sync
from sdl_agents.agents.research.academic_hermes import run_academic_sync
from sdl_agents.agents.research.safety_hermes_rag import run_safety_sync
from sdl_agents.orchestrator.router import _keyword_fallback, classify_intent_text


@pytest.mark.stage4
def test_router_database_keyword():
    d = _keyword_fallback("Which devices are offline in the latest snapshot?")
    assert d.intent == "database"


@pytest.mark.stage4
def test_router_research_keyword():
    d = _keyword_fallback("What PPE is required for BSL-2 formaldehyde work?")
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


@pytest.mark.stage4
def test_safety_mock():
    result = run_safety_sync("BSL-2 PPE for formaldehyde")
    assert result.get("decision")
    assert result.get("risk_level")


@pytest.mark.stage4
def test_experimental_mock():
    result = run_experimental_sync("serial dilution 96-well plate")
    assert "text" in result


@pytest.mark.stage4
def test_dilution_math():
    out = dilution_calc(10.0, 100.0, c2=1.0)
    assert out["v2"] == pytest.approx(1000.0)


@pytest.mark.stage4
@pytest.mark.slow
def test_router_llm():
    d = classify_intent_text("List down services from monitoring")
    assert d.intent in ("database", "research", "hybrid", "general")
