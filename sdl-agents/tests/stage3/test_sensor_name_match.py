"""Sensor name matching: prefer specific multi-token matches."""

from __future__ import annotations

import pytest

from sdl_agents.agents.database.formatters import find_sensor_by_name
from sdl_agents.monitoring.cache_answer import answer_from_cache
from sdl_agents.monitoring.state import MonitorState
from datetime import datetime, timezone


PICO_ROWS = [
    {
        "sensor_name": "pico-poe_1",
        "online": False,
        "reason": "last_seen > 13 days",
    },
    {
        "sensor_name": "pico-poe_2",
        "online": True,
        "last_reading": {"temperature": 37.35, "humidity": 88.62},
    },
]


@pytest.mark.stage3
def test_find_sensor_prefers_pico_poe_2_over_1():
    row = find_sensor_by_name(PICO_ROWS, "Temperature of pico poe 2")
    assert row is not None
    assert row["sensor_name"] == "pico-poe_2"


@pytest.mark.stage3
def test_find_sensor_pico_poe_1_when_asked():
    row = find_sensor_by_name(PICO_ROWS, "status of pico poe 1")
    assert row is not None
    assert row["sensor_name"] == "pico-poe_1"


@pytest.mark.stage3
def test_cache_answer_pico_poe_2_temperature():
    state = MonitorState(
        loaded_at=datetime.now(timezone.utc),
        sensors=PICO_ROWS,
        summary="test",
    )
    payload = answer_from_cache("Temp of pico poe 2", state)
    assert payload is not None
    assert "pico-poe_2" in str(payload["answer"])
    assert "37.35" in str(payload["answer"])
    assert "pico-poe_1" not in str(payload["answer"])
