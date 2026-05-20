"""Persist monitoring events to PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

from sdl_agents.integrations.postgres import connect
from sdl_agents.monitoring.state import MonitorAlert


def persist_events(
    alerts: list[MonitorAlert],
    *,
    snapshot_ids: dict[str, int | None] | None = None,
) -> int:
    if not alerts:
        return 0
    snapshot_ids = snapshot_ids or {}
    inserted = 0
    with connect() as conn:
        with conn.cursor() as cur:
            for alert in alerts:
                snap = snapshot_ids.get(alert.entity_type)
                cur.execute(
                    """
                    INSERT INTO monitoring_events (
                        entity_type, entity_key, severity, event_type,
                        payload, snapshot_id
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        alert.entity_type,
                        alert.entity_key,
                        alert.severity,
                        alert.event_type,
                        json.dumps(
                            {
                                "message": alert.message,
                                **alert.payload,
                            },
                            default=str,
                        ),
                        snap,
                    ),
                )
                inserted += 1
        conn.commit()
    return inserted


def fetch_open_alerts(limit: int = 50) -> list[dict[str, Any]]:
    from sdl_agents.integrations.postgres import fetch_all

    return fetch_all(
        """
        SELECT entity_type, entity_key, severity, event_type, payload, fired_at
        FROM monitoring_events
        WHERE acknowledged = FALSE
        ORDER BY fired_at DESC
        LIMIT %s
        """,
        (limit,),
    )
