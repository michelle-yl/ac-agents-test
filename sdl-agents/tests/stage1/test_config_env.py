"""Config loads from sdl-agents/.env only (not agent-langgraph/.env)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

SDL_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reload_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_CHAT_MODEL", raising=False)
    env_file = SDL_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key == "ANTHROPIC_CHAT_MODEL":
                monkeypatch.setenv("ANTHROPIC_CHAT_MODEL", val)
                break
    import sdl_agents.config as cfg

    importlib.reload(cfg)
    yield cfg
    importlib.reload(cfg)


def test_no_agent_langgraph_env_path(reload_config):
    assert not hasattr(reload_config, "AGENT_LANGGRAPH_ENV")
    assert reload_config.LOCAL_ENV == SDL_ROOT / ".env"


def test_seed_urls_default_when_unset(reload_config, monkeypatch):
    monkeypatch.delenv("SEED_URLS", raising=False)
    importlib.reload(reload_config)
    urls = reload_config.seed_urls()
    assert len(urls) >= 1
    assert all(u.startswith("http") for u in urls)
