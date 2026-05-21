"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Before importing sdl_agents.config: force mock integration for non-integration tests
# (developer sdl-agents/.env may set SDL_INTEGRATION_MODE=live).
os.environ["SDL_TEST_USE_MOCK_INTEGRATION"] = "1"


@pytest.fixture(autouse=True)
def _sdl_integration_mode(request):
    """Non-integration tests always use mock Hermes; integration uses live."""
    if request.node.get_closest_marker("integration"):
        os.environ.pop("SDL_TEST_USE_MOCK_INTEGRATION", None)
        os.environ["SDL_INTEGRATION_MODE"] = "live"
    else:
        os.environ["SDL_TEST_USE_MOCK_INTEGRATION"] = "1"
        os.environ["SDL_INTEGRATION_MODE"] = "mock"


@pytest.fixture
def repo_root() -> Path:
    return ROOT.parent


@pytest.fixture
def database_url() -> str:
    from sdl_agents.config import database_url as db_url

    return db_url()


def postgres_available() -> bool:
    try:
        from sdl_agents.integrations.postgres import fetch_all

        fetch_all("SELECT 1 AS one")
        return True
    except Exception:
        return False


@pytest.fixture
def require_postgres():
    if not postgres_available():
        pytest.skip("Monitoring PostgreSQL not available")
