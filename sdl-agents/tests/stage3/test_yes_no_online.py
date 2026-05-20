"""Yes/no online question handling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sdl_agents.agents.database.formatters import (
    find_entity,
    is_yes_no_online_question,
    try_yes_no_online_answer,
)
from sdl_agents.agents.database.fast_path import try_fast_path
from sdl_agents.monitoring.cache_answer import answer_from_cache
from sdl_agents.monitoring.state import MonitorState

SENSORS = [
    {"sensor_name": "M5PoeCam_2", "online": True, "last_reading": {}},
    {"sensor_name": "pico-poe_2", "online": True, "last_reading": {"temperature": 37.0}},
    {"sensor_name": "pico-poe_1", "online": False, "reason": "stale"},
]
DEVICES = [
    {"name": "laptop_test", "ip": "192.168.11.25", "online": True},
    {"name": "biomek_dell_pc", "ip": "192.168.11.10", "online": False},
]


@pytest.mark.stage3
def test_is_yes_no_detects_single_entity_question():
    assert is_yes_no_online_question("is m5 poe cam 2 online")
    assert is_yes_no_online_question("is poe cam 2 online")
    assert not is_yes_no_online_question("which devices are offline")
    assert not is_yes_no_online_question("devices online")


@pytest.mark.stage3
def test_find_entity_prefers_m5_poecam_sensor():
    found = find_entity(
        "is m5 poe cam 2 online", sensors=SENSORS, devices=DEVICES, services=[]
    )
    assert found is not None
    assert found[0] == "sensor"
    assert found[2] == "M5PoeCam_2"


@pytest.mark.stage3
def test_try_yes_no_answer_format():
    answer = try_yes_no_online_answer(
        "is m5 poe cam 2 online",
        devices=DEVICES,
        sensors=SENSORS,
        services=[],
    )
    assert answer is not None
    assert answer.startswith("Yes")
    assert "true" in answer
    assert "M5PoeCam_2" in answer
    assert "online" in answer


@pytest.mark.stage3
def test_try_yes_no_offline_sensor():
    answer = try_yes_no_online_answer(
        "is pico-poe_1 online",
        devices=[],
        sensors=SENSORS,
        services=[],
    )
    assert answer is not None
    assert answer.startswith("No")
    assert "false" in answer
    assert "offline" in answer


@pytest.mark.stage3
def test_cache_answer_yes_no_m5():
    state = MonitorState(
        loaded_at=datetime.now(timezone.utc),
        sensors=SENSORS,
        devices=DEVICES,
        services=[],
        summary="test",
    )
    payload = answer_from_cache("is m5 poe cam 2 online", state)
    assert payload is not None
    assert "Yes" in str(payload["answer"])
    assert "M5PoeCam_2" in str(payload["answer"])
    assert "biomek_dell_pc" not in str(payload["answer"])


@pytest.mark.stage3
def test_fast_path_yes_no(monkeypatch):
    def fake_invoke(tool_name, args):
        if tool_name == "get_latest_device_status":
            return DEVICES
        if tool_name == "get_latest_sensor_status":
            return SENSORS
        return []

    monkeypatch.setattr(
        "sdl_agents.agents.database.fast_path._invoke_tool",
        fake_invoke,
    )
    payload = try_fast_path("is m5 poe cam 2 online")
    assert payload is not None
    assert payload["llm_used"] is False
    assert "M5PoeCam_2" in payload["answer"]
    assert payload["answer"].startswith("Yes")
