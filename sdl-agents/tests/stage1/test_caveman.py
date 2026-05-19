"""Caveman mode helpers."""

from __future__ import annotations

import os

import pytest
from langchain.messages import HumanMessage, SystemMessage

from sdl_agents.caveman import (
    instruction_suffix,
    is_enabled,
    merge_system,
    with_caveman,
)


@pytest.fixture(autouse=True)
def _reset_caveman_env(monkeypatch):
    monkeypatch.delenv("CAVEMAN_ENABLED", raising=False)
    monkeypatch.delenv("CAVEMAN_LEVEL", raising=False)


def test_disabled_by_default_env_off(monkeypatch):
    monkeypatch.setenv("CAVEMAN_ENABLED", "0")
    assert not is_enabled()
    assert with_caveman([HumanMessage(content="hi")]) == [HumanMessage(content="hi")]
    assert instruction_suffix() == ""


def test_enabled_prepends_system(monkeypatch):
    monkeypatch.setenv("CAVEMAN_ENABLED", "1")
    monkeypatch.setenv("CAVEMAN_LEVEL", "lite")
    out = with_caveman([HumanMessage(content="hi")])
    assert isinstance(out[0], SystemMessage)
    assert "no filler" in out[0].content.lower()
    assert out[1].content == "hi"


def test_merge_system_appends_rules(monkeypatch):
    monkeypatch.setenv("CAVEMAN_ENABLED", "1")
    merged = merge_system("Base role.")
    assert merged.startswith("Base role.")
    assert "caveman" in merged.lower() or "terse" in merged.lower()
