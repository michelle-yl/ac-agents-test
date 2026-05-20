"""Stage 7: orchestrator cache answers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sdl_agents.monitoring.cache import set_state
from sdl_agents.monitoring.cache_answer import answer_from_cache
from sdl_agents.monitoring.state import MonitorAlert, MonitorState


@pytest.fixture
def sample_state():
    state = MonitorState(
        loaded_at=datetime.now(timezone.utc),
        devices=[
            {"ip": "10.0.0.1", "name": "offline-dev", "online": False},
            {"ip": "10.0.0.2", "name": "online-dev", "online": True},
        ],
        sensors=[
            {
                "sensor_name": "Cytomat",
                "online": True,
                "last_reading": {"temperature": 37.0},
            }
        ],
        services=[{"service_name": "incubator", "up": False, "host": "lab-1"}],
        open_alerts=[
            MonitorAlert(
                entity_type="service",
                entity_key="incubator",
                severity="critical",
                event_type="went_offline",
                message="Service incubator is down",
            )
        ],
        summary="devices: 2 total, 1 offline; sensors: 1 total, 0 offline; services: 1 total, 1 down",
    )
    set_state(state)
    return state


@pytest.mark.stage7
def test_cache_answer_offline_devices(sample_state):
    payload = answer_from_cache("Which devices are offline?", sample_state)
    assert payload is not None
    assert payload["llm_used"] is False
    assert "offline-dev" in str(payload["answer"])


@pytest.mark.stage7
def test_cache_answer_sensor_temperature(sample_state):
    payload = answer_from_cache("Cytomat temperature", sample_state)
    assert payload is not None
    assert "37" in str(payload["answer"])


@pytest.mark.stage7
def test_cache_answer_open_alerts(sample_state):
    payload = answer_from_cache("Any alerts?", sample_state)
    assert payload is not None
    assert "incubator" in str(payload["answer"]).lower()


@pytest.mark.stage7
def test_cache_skips_complex_query(sample_state):
    payload = answer_from_cache("Compare last 3 snapshots", sample_state)
    assert payload is None
