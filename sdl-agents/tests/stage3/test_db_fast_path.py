"""Stage 3: database fast path (no LLM)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from sdl_agents.agents.database.fast_path import try_fast_path


@pytest.mark.stage3
def test_fast_path_offline_devices():
    rows = [
        {"ip": "10.0.0.1", "name": "dev-a", "online": False},
        {"ip": "10.0.0.2", "name": "dev-b", "online": True},
    ]

    def fake_invoke(tool_name, args):
        offline = [r for r in rows if args.get("online_only") is False]
        return offline

    with patch(
        "sdl_agents.agents.database.fast_path._invoke_tool",
        side_effect=fake_invoke,
    ):
        payload = try_fast_path("Which devices are offline?")

    assert payload is not None
    assert payload["llm_used"] is False
    assert payload["source"] == "database_fast_path"
    assert payload["sources"][0]["source_type"] == "internal"
    assert "dev-a" in payload["answer"]
    assert payload["row_count"] == 1


@pytest.mark.stage3
def test_fast_path_skips_complex_query():
    payload = try_fast_path("Compare the last 3 device snapshots over time")
    assert payload is None


@pytest.mark.stage3
def test_fast_path_sensor_temperature():
    rows = [
        {
            "sensor_name": "Cytomat",
            "online": True,
            "last_reading": {"temperature": 37.0, "humidity": 2.4},
        }
    ]

    with patch(
        "sdl_agents.agents.database.fast_path._invoke_tool",
        return_value=rows,
    ):
        payload = try_fast_path("What is the Cytomat temperature?")

    assert payload is not None
    assert payload["llm_used"] is False
    assert "37" in payload["answer"]
    assert "Cytomat" in payload["answer"]
