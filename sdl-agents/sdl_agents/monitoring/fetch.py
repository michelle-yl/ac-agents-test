"""Fetch latest monitoring rows from PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

from sdl_agents.agents.database.tools import (
    get_latest_device_status,
    get_latest_sensor_status,
    get_latest_service_status,
)
from sdl_agents.integrations.postgres import fetch_all


def _parse_tool_json(raw: str) -> list[dict[str, Any]]:
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def fetch_devices(online_only: bool | None = None) -> list[dict[str, Any]]:
    return _parse_tool_json(
        get_latest_device_status.invoke({"online_only": online_only})
    )


def fetch_sensors() -> list[dict[str, Any]]:
    return _parse_tool_json(get_latest_sensor_status.invoke({}))


def fetch_services(up_only: bool | None = None) -> list[dict[str, Any]]:
    return _parse_tool_json(get_latest_service_status.invoke({"up_only": up_only}))


def latest_snapshot_ids() -> dict[str, int | None]:
    """Return latest snapshot id per domain for diffing."""
    out: dict[str, int | None] = {
        "device": None,
        "sensor": None,
        "service": None,
    }
    rows = fetch_all(
        """
        SELECT 'device' AS domain, id FROM monitoring_device_snapshots
        ORDER BY loaded_at DESC LIMIT 1
        """
    )
    if rows:
        out["device"] = rows[0]["id"]
    rows = fetch_all(
        """
        SELECT 'sensor' AS domain, id FROM monitoring_sensor_snapshots
        ORDER BY loaded_at DESC LIMIT 1
        """
    )
    if rows:
        out["sensor"] = rows[0]["id"]
    rows = fetch_all(
        """
        SELECT 'service' AS domain, id FROM monitoring_service_snapshots
        ORDER BY loaded_at DESC LIMIT 1
        """
    )
    if rows:
        out["service"] = rows[0]["id"]
    return out


def previous_snapshot_id(domain: str, current_id: int) -> int | None:
    table = {
        "device": "monitoring_device_snapshots",
        "sensor": "monitoring_sensor_snapshots",
        "service": "monitoring_service_snapshots",
    }.get(domain)
    if not table:
        return None
    rows = fetch_all(
        f"""
        SELECT id FROM {table}
        WHERE id < %s
        ORDER BY loaded_at DESC
        LIMIT 1
        """,
        (current_id,),
    )
    return rows[0]["id"] if rows else None


def fetch_device_entries(snapshot_id: int) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT ip, name, online, consecutive_down_count
        FROM monitoring_device_entries
        WHERE snapshot_id = %s
        """,
        (snapshot_id,),
    )


def fetch_sensor_entries(snapshot_id: int) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT sensor_name, online, alerts, reason, last_reading, consecutive_down_count
        FROM monitoring_sensor_entries
        WHERE snapshot_id = %s
        """,
        (snapshot_id,),
    )


def fetch_service_entries(snapshot_id: int) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT service_name, up, note, consecutive_down_count
        FROM monitoring_service_entries
        WHERE snapshot_id = %s
        """,
        (snapshot_id,),
    )


def fetch_alert_policy(domain: str, snapshot_id: int) -> dict[str, Any]:
    table = {
        "device": "monitoring_device_snapshots",
        "sensor": "monitoring_sensor_snapshots",
        "service": "monitoring_service_snapshots",
    }.get(domain)
    if not table:
        return {}
    rows = fetch_all(
        f"SELECT alert_policy FROM {table} WHERE id = %s",
        (snapshot_id,),
    )
    if not rows:
        return {}
    policy = rows[0].get("alert_policy")
    return policy if isinstance(policy, dict) else {}
