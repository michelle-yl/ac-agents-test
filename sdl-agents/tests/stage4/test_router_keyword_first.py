"""Stage 4: keyword-first router (no LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sdl_agents.orchestrator.router import (
    classify_intent_keyword,
    classify_intent_text,
)


@pytest.mark.stage4
def test_classify_intent_keyword_temperature():
    d = classify_intent_keyword("What is the incubator temperature?")
    assert d is not None
    assert d.intent == "database"


@pytest.mark.stage4
def test_classify_intent_keyword_hybrid():
    d = classify_intent_keyword(
        "Is the incubator service up and what are BSL-2 PPE requirements?"
    )
    assert d is not None
    assert d.intent == "hybrid"


@pytest.mark.stage4
def test_classify_intent_text_uses_keyword_without_llm():
    mock_model = MagicMock()
    with patch(
        "sdl_agents.orchestrator.router.init_chat_model", return_value=mock_model
    ):
        d = classify_intent_text("Which sensors are offline?")
    assert d.intent == "database"
    assert d.reason == "keyword database"
    mock_model.invoke.assert_not_called()
    mock_model.with_structured_output.assert_not_called()
