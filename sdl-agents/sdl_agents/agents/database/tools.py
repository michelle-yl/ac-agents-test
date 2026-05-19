"""Read-only tools for monitoring PostgreSQL tables."""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool

from sdl_agents.integrations.postgres import fetch_all

SCHEMA_DESCRIPTION = """
Tables:
- monitoring_device_config(id, source, query_url, query_sql, ping_timeout_seconds, ping_count)
- monitoring_device_snapshots(id, last_checked, last_successful_probe, last_probe_result, probe_note, method, alert_policy, loaded_at)
- monitoring_device_entries(snapshot_id, ip, name, online, ssh, smb, rdp, ports, last_change, last_verified, consecutive_down_count)
- monitoring_sensor_snapshots(id, last_check, summary, alert_policy, loaded_at)
- monitoring_sensor_entries(snapshot_id, sensor_name, online, alerts, last_change, last_seen, reason, last_reading, consecutive_down_count)
- monitoring_service_snapshots(id, last_check, check_status, check_note, alert_policy, loaded_at)
- monitoring_service_entries(snapshot_id, service_name, up, host, ip, port, protocol, note, consecutive_down_count)
""".strip()


@tool
def describe_schema() -> str:
    """Return allowed monitoring database tables and columns."""
    return SCHEMA_DESCRIPTION


@tool
def get_device_config() -> str:
    """Return the current device monitoring configuration row."""
    rows = fetch_all(
        """
        SELECT source, query_url, query_sql, ping_timeout_seconds, ping_count, updated_at
        FROM monitoring_device_config
        ORDER BY id DESC
        LIMIT 1
        """
    )
    return json.dumps(rows, default=str)


@tool
def get_latest_device_status(online_only: bool | None = None) -> str:
    """Return device entries from the most recent device status snapshot.

    online_only: if True, only online devices; if False, only offline; if None, all.
    """
    sql = """
        SELECT e.ip, e.name, e.online, e.ssh, e.smb, e.rdp, e.ports,
               e.consecutive_down_count, s.loaded_at AS snapshot_loaded_at
        FROM monitoring_device_entries e
        JOIN monitoring_device_snapshots s ON s.id = e.snapshot_id
        WHERE s.id = (SELECT id FROM monitoring_device_snapshots ORDER BY loaded_at DESC LIMIT 1)
    """
    params: tuple[Any, ...] = ()
    if online_only is True:
        sql += " AND e.online = TRUE"
    elif online_only is False:
        sql += " AND e.online = FALSE"
    sql += " ORDER BY e.ip"
    return json.dumps(fetch_all(sql, params), default=str)


@tool
def get_latest_sensor_status() -> str:
    """Return sensor entries from the most recent sensor status snapshot."""
    rows = fetch_all(
        """
        SELECT e.sensor_name, e.online, e.alerts, e.reason, e.last_reading,
               e.consecutive_down_count, s.summary, s.loaded_at AS snapshot_loaded_at
        FROM monitoring_sensor_entries e
        JOIN monitoring_sensor_snapshots s ON s.id = e.snapshot_id
        WHERE s.id = (SELECT id FROM monitoring_sensor_snapshots ORDER BY loaded_at DESC LIMIT 1)
        ORDER BY e.sensor_name
        """
    )
    return json.dumps(rows, default=str)


@tool
def get_latest_service_status(up_only: bool | None = None) -> str:
    """Return service entries from the most recent service status snapshot.

    up_only: if True, only services that are up; if False, only down; if None, all.
    """
    sql = """
        SELECT e.service_name, e.up, e.host, e.ip, e.port, e.protocol, e.note,
               e.consecutive_down_count, s.check_status, s.loaded_at AS snapshot_loaded_at
        FROM monitoring_service_entries e
        JOIN monitoring_service_snapshots s ON s.id = e.snapshot_id
        WHERE s.id = (SELECT id FROM monitoring_service_snapshots ORDER BY loaded_at DESC LIMIT 1)
    """
    if up_only is True:
        sql += " AND e.up = TRUE"
    elif up_only is False:
        sql += " AND e.up = FALSE"
    sql += " ORDER BY e.service_name"
    return json.dumps(fetch_all(sql), default=str)


DATABASE_TOOLS = [
    describe_schema,
    get_device_config,
    get_latest_device_status,
    get_latest_sensor_status,
    get_latest_service_status,
]
