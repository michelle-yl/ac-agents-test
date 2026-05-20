"""Template formatters for monitoring query results."""

from __future__ import annotations

import json
import re
from typing import Any


def _reading_str(reading: Any) -> str:
    if reading is None:
        return ""
    if isinstance(reading, str):
        try:
            reading = json.loads(reading)
        except json.JSONDecodeError:
            return reading
    if isinstance(reading, dict):
        parts = []
        if "temperature" in reading:
            parts.append(f"{reading['temperature']}°C")
        if "humidity" in reading:
            parts.append(f"humidity {reading['humidity']}%")
        if "level" in reading:
            parts.append(f"level {reading['level']}")
        return ", ".join(parts) if parts else json.dumps(reading)
    return str(reading)


def format_device_rows(rows: list[dict[str, Any]], *, label: str) -> str:
    if not rows:
        return f"No {label} devices in the latest snapshot."
    lines = [f"{label.capitalize()} devices ({len(rows)}):"]
    for r in rows[:25]:
        name = r.get("name") or r.get("ip") or "?"
        status = "online" if r.get("online") else "offline"
        lines.append(f"  - {name} ({r.get('ip', '?')}): {status}")
    if len(rows) > 25:
        lines.append(f"  ... and {len(rows) - 25} more")
    return "\n".join(lines)


def format_sensor_rows(rows: list[dict[str, Any]], *, label: str = "Sensor") -> str:
    if not rows:
        return "No sensors in the latest snapshot."
    lines = [f"{label} status ({len(rows)}):"]
    for r in rows[:25]:
        name = r.get("sensor_name") or "?"
        status = "online" if r.get("online") else "offline"
        reading = _reading_str(r.get("last_reading"))
        extra = f" — {reading}" if reading else ""
        reason = r.get("reason")
        if reason and not r.get("online"):
            extra += f" ({reason})"
        lines.append(f"  - {name}: {status}{extra}")
    if len(rows) > 25:
        lines.append(f"  ... and {len(rows) - 25} more")
    return "\n".join(lines)


def format_service_rows(rows: list[dict[str, Any]], *, label: str) -> str:
    if not rows:
        return f"No {label} services in the latest snapshot."
    lines = [f"{label.capitalize()} services ({len(rows)}):"]
    for r in rows[:25]:
        name = r.get("service_name") or "?"
        status = "up" if r.get("up") else "down"
        host = r.get("host") or r.get("ip") or ""
        suffix = f" @ {host}" if host else ""
        lines.append(f"  - {name}: {status}{suffix}")
    if len(rows) > 25:
        lines.append(f"  ... and {len(rows) - 25} more")
    return "\n".join(lines)


def format_rows_answer(tool_name: str, rows: list[dict[str, Any]]) -> str:
    if tool_name == "get_latest_device_status":
        if rows and all(r.get("online") is False for r in rows):
            return format_device_rows(rows, label="offline")
        if rows and all(r.get("online") is True for r in rows):
            return format_device_rows(rows, label="online")
        return format_device_rows(rows, label="all")
    if tool_name == "get_latest_sensor_status":
        return format_sensor_rows(rows)
    if tool_name == "get_latest_service_status":
        if rows and rows[0].get("up") is False:
            return format_service_rows(rows, label="down")
        return format_service_rows(rows, label="all")
    return f"Found {len(rows)} row(s)."


_NAME_STOPWORDS = frozenset(
    {
        "what",
        "is",
        "the",
        "temperature",
        "temp",
        "humidity",
        "status",
        "of",
        "for",
        "show",
        "get",
        "latest",
        "sensor",
        "device",
        "devices",
        "service",
        "services",
        "sensors",
        "online",
        "offline",
        "down",
        "up",
        "which",
        "are",
        "any",
        "all",
        "list",
        "tell",
        "me",
    }
)


def _normalize_entity_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_name_tokens(question: str) -> list[str]:
    """Meaningful name tokens from a user question (e.g. pico, poe, 2)."""
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_-]*", question.lower()):
        if token in _NAME_STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 2:
            tokens.append(token)
    return tokens


def extract_name_hint(question: str) -> str | None:
    """Legacy single-token hint; prefer extract_name_tokens + find_sensor_by_name."""
    tokens = extract_name_tokens(question)
    return tokens[0] if tokens else None


def _score_name_match(entity_name: str, tokens: list[str]) -> int:
    if not tokens or not entity_name:
        return 0
    norm_entity = _normalize_entity_name(entity_name)
    norm_query = "".join(tokens)
    lower_entity = entity_name.lower()

    if norm_query and norm_query == norm_entity:
        return 10_000 + len(norm_query)
    if norm_query and norm_query in norm_entity:
        return 5_000 + len(norm_query)
    if norm_entity and norm_entity in norm_query:
        return 4_000 + len(norm_entity)

    matched = 0
    score = 0
    for token in tokens:
        norm_token = _normalize_entity_name(token)
        if (
            token in lower_entity
            or norm_token in norm_entity
            or token.replace("_", "-") in lower_entity
        ):
            matched += 1
            score += len(token) * 100

    if len(tokens) > 1 and matched < len(tokens):
        return 0
    if matched == 0:
        return 0
    return score + matched * 50


def find_sensor_by_name(
    rows: list[dict[str, Any]], question: str
) -> dict[str, Any] | None:
    """Best matching sensor; prefers longest / most specific match, not list order."""
    tokens = extract_name_tokens(question)
    if not tokens:
        return None

    best_row: dict[str, Any] | None = None
    best_score = 0
    for r in rows:
        sensor = str(r.get("sensor_name") or "")
        if not sensor:
            continue
        score = _score_name_match(sensor, tokens)
        if score > best_score:
            best_score = score
            best_row = r
    return best_row


def is_yes_no_online_question(question: str) -> bool:
    """True for single-entity questions like 'is m5 poe cam 2 online'."""
    q = question.lower().strip()
    if not re.match(r"^(is|are)\b", q):
        return False
    if re.search(r"\b(which|list|any|all|what|how many)\b", q):
        return False
    if not re.search(r"\b(online|offline|up|down)\b", q):
        return False
    if not extract_name_tokens(question):
        return False
    if re.search(
        r"\b(devices|device|sensors|sensor|services|service)\s+(are\s+)?(online|offline|up|down)\b",
        q,
    ):
        return False
    return True


def find_entity(
    question: str,
    *,
    devices: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
    services: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], str] | None:
    """Best-matching entity as (kind, row, display_name) or None."""
    tokens = extract_name_tokens(question)
    if not tokens:
        return None

    best: tuple[str, dict[str, Any], str] | None = None
    best_score = 0

    for r in sensors:
        name = str(r.get("sensor_name") or "")
        score = _score_name_match(name, tokens)
        if score > best_score:
            best_score = score
            best = ("sensor", r, name)

    for r in devices:
        name = str(r.get("name") or r.get("ip") or "")
        score = _score_name_match(name, tokens)
        if score > best_score:
            best_score = score
            best = ("device", r, name)

    for r in services:
        name = str(r.get("service_name") or "")
        score = _score_name_match(name, tokens)
        if score > best_score:
            best_score = score
            best = ("service", r, name)

    return best


def format_online_answer(name: str, *, online: bool) -> str:
    if online:
        return f"Yes (true) — {name} is online."
    return f"No (false) — {name} is offline."


def try_yes_no_online_answer(
    question: str,
    *,
    devices: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
    services: list[dict[str, Any]],
) -> str | None:
    if not is_yes_no_online_question(question):
        return None
    found = find_entity(
        question, devices=devices, sensors=sensors, services=services
    )
    if not found:
        return None
    kind, row, name = found
    if kind == "service":
        online = bool(row.get("up"))
    else:
        online = bool(row.get("online"))
    return format_online_answer(name, online=online)
