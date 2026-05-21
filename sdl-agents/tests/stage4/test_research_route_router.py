"""Research-route subagent keyword routing."""

from __future__ import annotations

import pytest

from sdl_agents.agents.research.research_route_router import (
    classify_research_subagent_keyword,
    classify_research_subagent_text,
)


@pytest.mark.stage4
def test_research_subagent_safety_only():
    d = classify_research_subagent_keyword("What PPE is required for BSL-2 formaldehyde work?")
    assert d is not None
    assert d.needs_academic is False
    assert d.needs_safety is True
    assert d.needs_procedures is False


@pytest.mark.stage4
def test_research_subagent_academic_only():
    d = classify_research_subagent_keyword("Recent papers on organoids")
    assert d is not None
    assert d.needs_academic is True
    assert d.needs_safety is False
    assert d.needs_procedures is False


@pytest.mark.stage4
def test_research_subagent_procedures_only():
    d = classify_research_subagent_keyword(
        "What volume of 0.1M stock do I add to make 10ml of 0.01M dilution?"
    )
    assert d is not None
    assert d.needs_academic is False
    assert d.needs_safety is False
    assert d.needs_procedures is True


@pytest.mark.stage4
def test_research_subagent_equipment_manual_procedures_only():
    d = classify_research_subagent_keyword(
        "How do I interpret the liquid handler manual for robotic arm calibration?"
    )
    assert d is not None
    assert d.needs_academic is False
    assert d.needs_safety is False
    assert d.needs_procedures is True


@pytest.mark.stage4
def test_research_subagent_all_three():
    d = classify_research_subagent_keyword(
        "BSL-2 PPE for formaldehyde and steps for serial dilution in a 96-well plate"
    )
    assert d is not None
    assert d.needs_academic is False
    assert d.needs_safety is True
    assert d.needs_procedures is True


@pytest.mark.stage4
def test_research_subagent_text_keyword_path():
    flags = classify_research_subagent_text("Recent papers on organoids")
    assert flags["needs_academic"] is True
    assert flags["needs_safety"] is False
    assert flags["needs_procedures"] is False
