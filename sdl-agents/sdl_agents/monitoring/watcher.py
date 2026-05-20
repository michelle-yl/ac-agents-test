"""Monitoring watcher: ingest, fetch, diff, cache."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sdl_agents.monitoring.cache import set_state
from sdl_agents.monitoring.events import fetch_open_alerts, persist_events
from sdl_agents.monitoring.fetch import (
    fetch_devices,
    fetch_sensors,
    fetch_services,
    latest_snapshot_ids,
)
from sdl_agents.monitoring.ingest import run_ingest
from sdl_agents.monitoring.rules import compute_diff_alerts
from sdl_agents.monitoring.state import MonitorAlert, MonitorState

logger = logging.getLogger(__name__)


def _build_summary(devices: list, sensors: list, services: list) -> str:
    dev_off = sum(1 for d in devices if d.get("online") is False)
    sen_off = sum(1 for s in sensors if s.get("online") is False)
    svc_down = sum(1 for s in services if s.get("up") is False)
    return (
        f"devices: {len(devices)} total, {dev_off} offline; "
        f"sensors: {len(sensors)} total, {sen_off} offline; "
        f"services: {len(services)} total, {svc_down} down"
    )


def _rows_to_alerts(rows: list[dict]) -> list[MonitorAlert]:
    out: list[MonitorAlert] = []
    for row in rows:
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            import json

            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        message = payload.get("message", str(payload))
        out.append(
            MonitorAlert(
                entity_type=str(row.get("entity_type", "")),
                entity_key=str(row.get("entity_key", "")),
                severity=str(row.get("severity", "info")),
                event_type=str(row.get("event_type", "alert")),
                message=message,
                payload=payload if isinstance(payload, dict) else {},
            )
        )
    return out


def refresh_state(*, run_ingest_step: bool = True) -> MonitorState:
    """Ingest (optional), diff, update cache, persist new events."""
    if run_ingest_step:
        try:
            run_ingest()
            logger.info("ingest completed")
        except Exception as exc:
            logger.warning("ingest skipped or failed: %s", exc)

    snapshot_ids = latest_snapshot_ids()
    new_alerts = compute_diff_alerts()
    if new_alerts:
        try:
            n = persist_events(new_alerts, snapshot_ids=snapshot_ids)
            logger.info("persisted %s monitoring event(s)", n)
        except Exception as exc:
            logger.warning("could not persist events (table missing?): %s", exc)

    devices = fetch_devices()
    sensors = fetch_sensors()
    services = fetch_services()

    open_alerts = new_alerts
    try:
        db_alerts = _rows_to_alerts(fetch_open_alerts())
        if db_alerts:
            open_alerts = db_alerts
    except Exception:
        pass

    state = MonitorState(
        loaded_at=datetime.now(timezone.utc),
        device_snapshot_id=snapshot_ids.get("device"),
        sensor_snapshot_id=snapshot_ids.get("sensor"),
        service_snapshot_id=snapshot_ids.get("service"),
        devices=devices,
        sensors=sensors,
        services=services,
        open_alerts=open_alerts,
        summary=_build_summary(devices, sensors, services),
    )
    set_state(state)
    return state
