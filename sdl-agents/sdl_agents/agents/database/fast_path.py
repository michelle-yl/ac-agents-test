"""Deterministic DB query routing without LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from sdl_agents.agents.database.formatters import (
    extract_name_tokens,
    find_entity,
    find_sensor_by_name,
    format_rows_answer,
    format_sensor_rows,
    is_yes_no_online_question,
    try_yes_no_online_answer,
)
from sdl_agents.agents.database.tools import (
    get_latest_device_status,
    get_latest_sensor_status,
    get_latest_service_status,
)
from sdl_agents.config import DB_FAST_PATH_ENABLED, DB_FAST_PATH_MAX_ROWS
from sdl_agents.sources import internal_source

_TOOL_MAP = {
    "get_latest_device_status": get_latest_device_status,
    "get_latest_sensor_status": get_latest_sensor_status,
    "get_latest_service_status": get_latest_service_status,
}

_COMPLEX_PATTERNS = (
    r"\bcompare\b",
    r"\bhistory\b",
    r"\bover time\b",
    r"\blast \d+",
    r"\bsnapshot",
    r"\btrend",
    r"\bsql\b",
    r"\bexplain\b",
    r"\bwhy\b",
)


def _is_complex(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _COMPLEX_PATTERNS)


def _match_tool(question: str) -> tuple[str, dict[str, Any]] | None:
    q = question.lower()

    if re.search(r"\b(service|services)\b", q):
        if re.search(r"\b(down|offline|failed|unavailable)\b", q):
            return "get_latest_service_status", {"up_only": False}
        if re.search(r"\b(up|running|online)\b", q):
            return "get_latest_service_status", {"up_only": True}
        return "get_latest_service_status", {"up_only": None}

    if re.search(
        r"\b(sensor|sensors|temperature|temp|humidity|reading|incubator|cytomat|pico)\b",
        q,
    ):
        return "get_latest_sensor_status", {}

    if is_yes_no_online_question(question):
        return None

    if re.search(r"\b(device|devices|ip|ping|ssh|smb|rdp)\b", q):
        if re.search(r"\b(offline|down|unreachable)\b", q):
            return "get_latest_device_status", {"online_only": False}
        if re.search(r"\b(online|up)\b", q) and not extract_name_tokens(question):
            return "get_latest_device_status", {"online_only": True}
        return "get_latest_device_status", {"online_only": None}

    if re.search(r"\b(offline|down)\b", q) and not re.search(r"\bservice", q):
        return "get_latest_device_status", {"online_only": False}

    if re.search(r"\b(status|monitoring|lab)\b", q):
        return "get_latest_sensor_status", {}

    return None


def _invoke_tool(tool_name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    tool = _TOOL_MAP[tool_name]
    raw = tool.invoke(args)
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _sensor_single_answer(rows: list[dict[str, Any]], question: str) -> str | None:
    if not extract_name_tokens(question):
        return None
    row = find_sensor_by_name(rows, question)
    if not row:
        return None
    return format_sensor_rows([row], label=row.get("sensor_name", "Sensor"))


def _yes_no_fast_path(question: str) -> dict[str, Any] | None:
    if not is_yes_no_online_question(question):
        return None
    try:
        devices = _invoke_tool("get_latest_device_status", {"online_only": None})
        sensors = _invoke_tool("get_latest_sensor_status", {})
        services = _invoke_tool("get_latest_service_status", {"up_only": None})
    except Exception:
        return None
    answer = try_yes_no_online_answer(
        question, devices=devices, sensors=sensors, services=services
    )
    if answer is None:
        return None
    found = find_entity(question, devices=devices, sensors=sensors, services=services)
    preview = [found[1]] if found else []
    return {
        "answer": answer,
        "row_count": len(preview),
        "preview_rows": preview,
        "source": "database_fast_path",
        "sources": [
            internal_source("monitoring PostgreSQL", prefix="Database"),
            internal_source("yes_no_online", prefix="Database tool"),
        ],
        "llm_used": False,
        "tool": "yes_no_online",
    }


def try_fast_path(question: str) -> dict[str, Any] | None:
    """Return db_payload if question matches a simple monitoring pattern."""
    if not DB_FAST_PATH_ENABLED:
        return None
    if _is_complex(question):
        return None

    yes_no_payload = _yes_no_fast_path(question)
    if yes_no_payload is not None:
        return yes_no_payload

    matched = _match_tool(question)
    if not matched:
        return None

    tool_name, args = matched
    try:
        rows = _invoke_tool(tool_name, args)
    except Exception:
        return None

    if len(rows) > DB_FAST_PATH_MAX_ROWS:
        return None

    answer = None
    if tool_name == "get_latest_sensor_status":
        answer = _sensor_single_answer(rows, question)
    if not answer:
        if tool_name == "get_latest_device_status" and args.get("online_only") is False:
            offline = [r for r in rows if r.get("online") is False]
            rows = offline
        if tool_name == "get_latest_service_status" and args.get("up_only") is False:
            down = [r for r in rows if r.get("up") is False]
            rows = down
        answer = format_rows_answer(tool_name, rows)

    return {
        "answer": answer,
        "row_count": len(rows),
        "preview_rows": rows[:10],
        "source": "database_fast_path",
        "sources": [
            internal_source("monitoring PostgreSQL", prefix="Database"),
            internal_source(tool_name, prefix="Database tool"),
        ],
        "llm_used": False,
        "tool": tool_name,
    }


def should_skip_summarize(question: str, tool_msgs_count: int, row_count: int) -> bool:
    """Skip summarize_db LLM when a single simple tool returned modest data."""
    if not DB_FAST_PATH_ENABLED:
        return False
    if tool_msgs_count != 1 or row_count > DB_FAST_PATH_MAX_ROWS:
        return False
    return not _is_complex(question)
