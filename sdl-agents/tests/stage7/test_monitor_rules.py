"""Stage 7: monitoring diff rules."""

from __future__ import annotations

import pytest

from sdl_agents.monitoring.rules import diff_devices, diff_sensors, diff_services


@pytest.mark.stage7
def test_diff_device_went_offline():
    prev = [{"ip": "10.0.0.1", "name": "a", "online": True, "consecutive_down_count": 0}]
    curr = [{"ip": "10.0.0.1", "name": "a", "online": False, "consecutive_down_count": 1}]
    alerts = diff_devices(prev, curr, {"consecutiveDownThreshold": 2})
    assert len(alerts) == 1
    assert alerts[0].event_type == "went_offline"
    assert alerts[0].severity == "critical"


@pytest.mark.stage7
def test_diff_sensor_consecutive_down_threshold():
    prev = [{"sensor_name": "s1", "online": False, "alerts": [], "consecutive_down_count": 1}]
    curr = [{"sensor_name": "s1", "online": False, "alerts": [], "consecutive_down_count": 2}]
    alerts = diff_sensors(prev, curr, {"consecutiveDownThreshold": 2})
    assert any(a.event_type == "consecutive_down" for a in alerts)


@pytest.mark.stage7
def test_diff_service_came_online():
    prev = [{"service_name": "incubator", "up": False, "consecutive_down_count": 0}]
    curr = [{"service_name": "incubator", "up": True, "consecutive_down_count": 0}]
    alerts = diff_services(prev, curr, {})
    assert len(alerts) == 1
    assert alerts[0].event_type == "came_online"
