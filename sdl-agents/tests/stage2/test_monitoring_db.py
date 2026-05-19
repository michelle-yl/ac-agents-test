"""Stage 2: monitoring PostgreSQL tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdl_agents.integrations.postgres import assert_read_only_sql, fetch_all

SEED_DEVICE_STATUS = (
    Path(__file__).resolve().parents[3]
    / "db"
    / "seed"
    / "data"
    / "monitoring"
    / "device-status.json"
)


@pytest.mark.stage2
def test_read_only_rejects_insert():
    with pytest.raises(ValueError, match="SELECT|Write"):
        assert_read_only_sql("INSERT INTO monitoring_device_config DEFAULT VALUES")


@pytest.mark.stage2
def test_select_one(require_postgres):
    rows = fetch_all("SELECT 1 AS one")
    assert rows[0]["one"] == 1


@pytest.mark.stage2
def test_latest_device_snapshot(require_postgres):
    rows = fetch_all(
        """
        SELECT COUNT(*)::int AS cnt
        FROM monitoring_device_entries e
        JOIN monitoring_device_snapshots s ON s.id = e.snapshot_id
        WHERE s.id = (SELECT id FROM monitoring_device_snapshots ORDER BY loaded_at DESC LIMIT 1)
        """
    )
    if SEED_DEVICE_STATUS.is_file():
        expected = len(json.loads(SEED_DEVICE_STATUS.read_text(encoding="utf-8")).get("devices", {}))
        assert rows[0]["cnt"] == expected
    else:
        assert rows[0]["cnt"] >= 0
