"""Stage 1: foundation tests."""

from __future__ import annotations

import pytest

from sdl_agents.config import REPO_ROOT, SDL_AGENTS_ROOT, database_url, is_live_integration
from sdl_agents.state import SDLAgentState


@pytest.mark.stage1
def test_repo_paths_exist():
    assert REPO_ROOT.is_dir()
    assert SDL_AGENTS_ROOT.is_dir()


@pytest.mark.stage1
def test_database_url_default():
    url = database_url()
    assert "postgresql://" in url
    assert "angie_monitoring" in url or "5433" in url


@pytest.mark.stage1
def test_integration_mode_mock_by_default():
    assert is_live_integration() is False


@pytest.mark.stage1
def test_state_typing():
    state: SDLAgentState = {
        "messages": [],
        "intent": "general",
        "route_reason": "",
        "db_payload": None,
        "research_payload": None,
        "research_flags": {},
        "errors": [],
    }
    assert state["intent"] == "general"
