"""Stage 6: end-to-end orchestrator tests."""

from __future__ import annotations

import pytest
from langchain.messages import HumanMessage

from sdl_agents.orchestrator.graph import build_orchestrator_graph
from tests.conftest import postgres_available


def _initial_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "intent": "general",
        "route_reason": "",
        "db_payload": None,
        "monitor_cache_used": False,
        "research_payload": None,
        "research_flags": {},
        "errors": [],
    }


@pytest.mark.stage6
def test_e2e_general_greeting():
    graph = build_orchestrator_graph()
    result = graph.invoke(_initial_state("What can you help me with?"))
    assert result["messages"][-1].content


@pytest.mark.stage6
def test_e2e_research_mock():
    graph = build_orchestrator_graph()
    result = graph.invoke(_initial_state("What PPE is required for BSL-2 work?"))
    assert result.get("research_payload") or "Safety" in result["messages"][-1].content


@pytest.mark.stage6
@pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not available")
def test_e2e_database():
    graph = build_orchestrator_graph()
    result = graph.invoke(_initial_state("Which devices are offline?"))
    assert result.get("db_payload") or "device" in result["messages"][-1].content.lower()


@pytest.mark.stage6
@pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not available")
def test_e2e_database_fast_path_no_llm():
    from datetime import datetime, timezone

    from sdl_agents.monitoring.cache import set_state
    from sdl_agents.monitoring.state import MonitorState

    set_state(
        MonitorState(
            loaded_at=datetime.now(timezone.utc),
            devices=[{"ip": "10.0.0.9", "name": "down-box", "online": False}],
            sensors=[],
            services=[],
            summary="1 offline device",
        )
    )
    graph = build_orchestrator_graph()
    result = graph.invoke(_initial_state("Which devices are offline?"))
    db = result.get("db_payload") or {}
    assert db.get("llm_used") is False
    assert result.get("monitor_cache_used") is True


@pytest.mark.stage6
@pytest.mark.integration
@pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not available")
def test_live_l1_offline_devices():
    graph = build_orchestrator_graph()
    result = graph.invoke(_initial_state("Which devices are offline in the latest snapshot?"))
    assert result.get("db_payload") is not None
