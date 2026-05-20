"""In-memory monitoring snapshot state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MonitorAlert:
    entity_type: str
    entity_key: str
    severity: str
    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorState:
    loaded_at: datetime
    device_snapshot_id: int | None = None
    sensor_snapshot_id: int | None = None
    service_snapshot_id: int | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    open_alerts: list[MonitorAlert] = field(default_factory=list)
    summary: str = ""
