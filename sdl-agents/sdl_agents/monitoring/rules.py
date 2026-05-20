"""Diff snapshots and produce monitoring alerts."""

from __future__ import annotations

from typing import Any

from sdl_agents.monitoring.fetch import (
    fetch_alert_policy,
    fetch_device_entries,
    fetch_sensor_entries,
    fetch_service_entries,
    latest_snapshot_ids,
    previous_snapshot_id,
)
from sdl_agents.monitoring.state import MonitorAlert


def _key_device(row: dict[str, Any]) -> str:
    return str(row.get("ip") or row.get("name") or "unknown")


def _key_sensor(row: dict[str, Any]) -> str:
    return str(row.get("sensor_name") or "unknown")


def _key_service(row: dict[str, Any]) -> str:
    return str(row.get("service_name") or "unknown")


def _index(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    return {key_fn(r): r for r in rows}


def _threshold(policy: dict[str, Any]) -> int:
    raw = policy.get("consecutiveDownThreshold") or policy.get("consecutive_down_threshold")
    try:
        return int(raw) if raw is not None else 2
    except (TypeError, ValueError):
        return 2


def diff_devices(
    prev: list[dict[str, Any]], curr: list[dict[str, Any]], policy: dict[str, Any]
) -> list[MonitorAlert]:
    alerts: list[MonitorAlert] = []
    prev_by = _index(prev, _key_device)
    curr_by = _index(curr, _key_device)
    threshold = _threshold(policy)

    for key, row in curr_by.items():
        was = prev_by.get(key)
        online = row.get("online")
        name = row.get("name") or key
        down_count = row.get("consecutive_down_count") or 0

        if online is False:
            if was is None or was.get("online") is not False:
                alerts.append(
                    MonitorAlert(
                        entity_type="device",
                        entity_key=key,
                        severity="critical",
                        event_type="went_offline",
                        message=f"Device {name} ({key}) went offline",
                        payload={"name": name, "ip": key},
                    )
                )
            elif down_count >= threshold:
                alerts.append(
                    MonitorAlert(
                        entity_type="device",
                        entity_key=key,
                        severity="warning",
                        event_type="consecutive_down",
                        message=f"Device {name} down {down_count} consecutive checks",
                        payload={"consecutive_down_count": down_count},
                    )
                )
        elif was is not None and was.get("online") is False and online is True:
            alerts.append(
                MonitorAlert(
                    entity_type="device",
                    entity_key=key,
                    severity="info",
                    event_type="came_online",
                    message=f"Device {name} ({key}) came online",
                    payload={"name": name},
                )
            )
    return alerts


def diff_sensors(
    prev: list[dict[str, Any]], curr: list[dict[str, Any]], policy: dict[str, Any]
) -> list[MonitorAlert]:
    alerts: list[MonitorAlert] = []
    prev_by = _index(prev, _key_sensor)
    curr_by = _index(curr, _key_sensor)
    threshold = _threshold(policy)

    for key, row in curr_by.items():
        was = prev_by.get(key)
        name = row.get("sensor_name") or key
        online = row.get("online")
        sensor_alerts = row.get("alerts") or []
        down_count = row.get("consecutive_down_count") or 0

        if sensor_alerts and (was is None or not (was.get("alerts") or [])):
            alerts.append(
                MonitorAlert(
                    entity_type="sensor",
                    entity_key=key,
                    severity="warning",
                    event_type="alert",
                    message=f"Sensor {name} has alerts: {sensor_alerts}",
                    payload={"alerts": sensor_alerts},
                )
            )

        if online is False:
            if was is None or was.get("online") is not False:
                reason = row.get("reason") or ""
                alerts.append(
                    MonitorAlert(
                        entity_type="sensor",
                        entity_key=key,
                        severity="critical",
                        event_type="went_offline",
                        message=f"Sensor {name} offline" + (f": {reason}" if reason else ""),
                        payload={"reason": reason},
                    )
                )
            elif down_count >= threshold:
                alerts.append(
                    MonitorAlert(
                        entity_type="sensor",
                        entity_key=key,
                        severity="warning",
                        event_type="consecutive_down",
                        message=f"Sensor {name} down {down_count} consecutive checks",
                        payload={"consecutive_down_count": down_count},
                    )
                )
        elif was is not None and was.get("online") is False and online is True:
            alerts.append(
                MonitorAlert(
                    entity_type="sensor",
                    entity_key=key,
                    severity="info",
                    event_type="came_online",
                    message=f"Sensor {name} came online",
                    payload={},
                )
            )
    return alerts


def diff_services(
    prev: list[dict[str, Any]], curr: list[dict[str, Any]], policy: dict[str, Any]
) -> list[MonitorAlert]:
    alerts: list[MonitorAlert] = []
    prev_by = _index(prev, _key_service)
    curr_by = _index(curr, _key_service)
    threshold = _threshold(policy)

    for key, row in curr_by.items():
        was = prev_by.get(key)
        name = row.get("service_name") or key
        up = row.get("up")
        down_count = row.get("consecutive_down_count") or 0

        if up is False:
            if was is None or was.get("up") is not False:
                alerts.append(
                    MonitorAlert(
                        entity_type="service",
                        entity_key=key,
                        severity="critical",
                        event_type="went_offline",
                        message=f"Service {name} is down",
                        payload={"note": row.get("note")},
                    )
                )
            elif down_count >= threshold:
                alerts.append(
                    MonitorAlert(
                        entity_type="service",
                        entity_key=key,
                        severity="warning",
                        event_type="consecutive_down",
                        message=f"Service {name} down {down_count} consecutive checks",
                        payload={"consecutive_down_count": down_count},
                    )
                )
        elif was is not None and was.get("up") is False and up is True:
            alerts.append(
                MonitorAlert(
                    entity_type="service",
                    entity_key=key,
                    severity="info",
                    event_type="came_online",
                    message=f"Service {name} is up",
                    payload={},
                )
            )
    return alerts


def compute_diff_alerts() -> list[MonitorAlert]:
    """Compare latest two snapshots per domain; return new alerts."""
    ids = latest_snapshot_ids()
    all_alerts: list[MonitorAlert] = []

    for domain, fetch_entries, diff_fn in (
        ("device", fetch_device_entries, diff_devices),
        ("sensor", fetch_sensor_entries, diff_sensors),
        ("service", fetch_service_entries, diff_services),
    ):
        current_id = ids.get(domain)
        if current_id is None:
            continue
        prev_id = previous_snapshot_id(domain, current_id)
        if prev_id is None:
            continue
        curr_rows = fetch_entries(current_id)
        prev_rows = fetch_entries(prev_id)
        policy = fetch_alert_policy(domain, current_id)
        all_alerts.extend(diff_fn(prev_rows, curr_rows, policy))

    return all_alerts
