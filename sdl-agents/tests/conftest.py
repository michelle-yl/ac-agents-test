"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force mock integrations in unit tests unless integration marker
os.environ.setdefault("SDL_INTEGRATION_MODE", "mock")


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
