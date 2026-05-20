"""Thread-safe singleton cache for latest MonitorState."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from sdl_agents.config import MONITOR_CACHE_MAX_AGE_SEC
from sdl_agents.monitoring.state import MonitorState

_lock = threading.Lock()
_state: MonitorState | None = None


def get_state() -> MonitorState | None:
    with _lock:
        return _state


def set_state(state: MonitorState) -> None:
    global _state
    with _lock:
        _state = state


def is_cache_fresh(max_age_seconds: int | None = None) -> bool:
    max_age = max_age_seconds if max_age_seconds is not None else MONITOR_CACHE_MAX_AGE_SEC
    state = get_state()
    if state is None:
        return False
    now = datetime.now(timezone.utc)
    loaded = state.loaded_at
    if loaded.tzinfo is None:
        loaded = loaded.replace(tzinfo=timezone.utc)
    age = (now - loaded).total_seconds()
    return age <= max_age


def clear_state() -> None:
    global _state
    with _lock:
        _state = None
