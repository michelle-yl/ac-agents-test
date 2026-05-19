"""Stage 3: database agent tools and graph."""

from __future__ import annotations

import json

import pytest

from sdl_agents.agents.database.tools import (
    describe_schema,
    get_latest_device_status,
    get_latest_service_status,
)


@pytest.mark.stage3
def test_describe_schema():
    text = describe_schema.invoke({})
    assert "monitoring_device_entries" in text


@pytest.mark.stage3
def test_get_latest_device_status(require_postgres):
    raw = get_latest_device_status.invoke({"online_only": None})
    rows = json.loads(raw)
    assert isinstance(rows, list)


@pytest.mark.stage3
def test_offline_devices_filter(require_postgres):
    raw = get_latest_device_status.invoke({"online_only": False})
    rows = json.loads(raw)
    for row in rows:
        assert row["online"] is False


@pytest.mark.stage3
def test_service_status(require_postgres):
    raw = get_latest_service_status.invoke({"up_only": None})
    rows = json.loads(raw)
    assert isinstance(rows, list)
