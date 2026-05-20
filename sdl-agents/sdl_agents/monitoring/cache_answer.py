"""Answer simple monitoring questions from in-memory cache (no LLM)."""

from __future__ import annotations

import re

from sdl_agents.agents.database.formatters import (
    extract_name_tokens,
    find_entity,
    find_sensor_by_name,
    format_device_rows,
    format_sensor_rows,
    format_service_rows,
    try_yes_no_online_answer,
)
from sdl_agents.monitoring.state import MonitorState

_COMPLEX_PATTERNS = (
    r"\bcompare\b",
    r"\bhistory\b",
    r"\bover time\b",
    r"\blast \d+",
    r"\bsnapshot",
    r"\btrend",
    r"\bsql\b",
)


def _is_complex(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _COMPLEX_PATTERNS)


def answer_from_cache(question: str, state: MonitorState) -> dict[str, object] | None:
    """Return db_payload dict or None if cache cannot answer."""
    if _is_complex(question):
        return None

    q = question.lower()

    yes_no = try_yes_no_online_answer(
        question,
        devices=state.devices,
        sensors=state.sensors,
        services=state.services,
    )
    if yes_no is not None:
        found = find_entity(
            question,
            devices=state.devices,
            sensors=state.sensors,
            services=state.services,
        )
        preview = [found[1]] if found else []
        return _payload(yes_no, source="monitor_cache", preview=preview)

    if re.search(r"\b(alert|alerts|warning|issue|problem|error)\b", q):
        if not state.open_alerts:
            return _payload("No open monitoring alerts.", source="monitor_cache")
        lines = [f"Open alerts ({len(state.open_alerts)}):"]
        for a in state.open_alerts[:20]:
            lines.append(f"  - [{a.severity}] {a.message}")
        return _payload("\n".join(lines), source="monitor_cache")

    if re.search(r"\b(summary|overview|status)\b", q) and not re.search(
        r"\b(device|sensor|service)\b", q
    ):
        return _payload(state.summary, source="monitor_cache")

    if re.search(r"\b(sensor|temperature|temp|humidity|reading)\b", q):
        if extract_name_tokens(question):
            row = find_sensor_by_name(state.sensors, question)
            if row:
                return _payload(
                    format_sensor_rows([row], label=str(row.get("sensor_name", ""))),
                    source="monitor_cache",
                    preview=[row],
                )
        offline = [s for s in state.sensors if s.get("online") is False]
        if re.search(r"\b(offline|down)\b", q):
            return _payload(
                format_sensor_rows(offline, label="Offline sensor"),
                source="monitor_cache",
                preview=offline[:10],
            )
        return _payload(
            format_sensor_rows(state.sensors),
            source="monitor_cache",
            preview=state.sensors[:10],
        )

    if re.search(r"\b(service|services)\b", q):
        down = [s for s in state.services if s.get("up") is False]
        if re.search(r"\b(down|offline|failed)\b", q):
            return _payload(
                format_service_rows(down, label="down"),
                source="monitor_cache",
                preview=down[:10],
            )
        return _payload(
            format_service_rows(state.services, label="all"),
            source="monitor_cache",
            preview=state.services[:10],
        )

    if re.search(r"\b(device|devices|offline|down)\b", q) or (
        re.search(r"\bonline\b", q) and re.search(r"\b(devices|device)\b", q)
    ):
        offline = [d for d in state.devices if d.get("online") is False]
        if re.search(r"\b(offline|down)\b", q):
            return _payload(
                format_device_rows(offline, label="offline"),
                source="monitor_cache",
                preview=offline[:10],
            )
        return _payload(
            format_device_rows(state.devices, label="all"),
            source="monitor_cache",
            preview=state.devices[:10],
        )

    if re.search(r"\b(monitoring|lab|what.?s)\b", q):
        parts = [state.summary]
        if state.open_alerts:
            parts.append(f"Alerts: {len(state.open_alerts)} open")
        return _payload("\n".join(parts), source="monitor_cache")

    return None


def _payload(
    answer: str,
    *,
    source: str,
    preview: list | None = None,
) -> dict[str, object]:
    preview = preview or []
    return {
        "answer": answer,
        "row_count": len(preview),
        "preview_rows": preview[:10],
        "source": source,
        "llm_used": False,
    }
