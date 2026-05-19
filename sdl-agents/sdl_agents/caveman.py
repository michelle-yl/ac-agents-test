"""Caveman terse output mode — https://github.com/juliusbrussee/caveman

Synced with upstream skills/caveman/SKILL.md intensity levels.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain.messages import SystemMessage
from langchain_core.messages import BaseMessage, convert_to_messages

# Intensity rules — keep in sync with upstream SKILL.md
_CAVEMAN_INTENSITY: dict[str, str] = {
    "lite": (
        "Output style: no filler or hedging. Keep articles and full sentences. "
        "Professional but tight. Technical terms exact; leave code blocks unchanged."
    ),
    "full": (
        "Output style: respond terse like smart caveman. All technical substance stay; fluff die. "
        "Drop articles (a/an/the), filler (just/really/basically/actually/simply), "
        "pleasantries, hedging. Fragments OK. Short synonyms. Technical terms exact; "
        "code blocks unchanged. Pattern: [thing] [action] [reason]. [next step]."
    ),
    "ultra": (
        "Output style: ultra-terse. Abbreviate prose words (DB/auth/config/req/res/fn/impl), "
        "strip conjunctions, use arrows for causality (X → Y). "
        "Never abbreviate code symbols, function names, API names, error strings."
    ),
    "wenyan-lite": (
        "Output style: semi-classical Chinese register. Drop filler/hedging; "
        "keep clearer grammar. Technical tokens (API/code) stay as in source."
    ),
    "wenyan-full": (
        "Output style: maximum classical Chinese terseness (文言文). "
        "Classical particles, subjects often omitted where clear."
    ),
    "wenyan-ultra": (
        "Output style: extreme classical abbreviation; maximum compression; still precise."
    ),
}


def _env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def is_enabled() -> bool:
    return _env_truthy("CAVEMAN_ENABLED", "1")


def level() -> str:
    return os.environ.get("CAVEMAN_LEVEL", "full").strip().lower()


def intensity_block() -> str:
    if not is_enabled():
        return ""
    lvl = level()
    intensity = _CAVEMAN_INTENSITY.get(lvl, _CAVEMAN_INTENSITY["full"])
    return (
        f"{intensity}\n\n"
        "ACTIVE on every assistant reply in this chat unless user says "
        '"stop caveman" or "normal mode".\n'
        "When security, irreversible actions, or ambiguity would suffer from terseness, "
        "switch to clear full sentences for that part only, then resume terse style.\n"
    )


def system_text(*, role_prefix: str = "You are a helpful assistant.") -> str:
    block = intensity_block()
    if not block:
        return role_prefix
    return f"{role_prefix}\n\n{block}"


def merge_system(base: str) -> str:
    """Append caveman output rules to an existing system prompt."""
    block = intensity_block()
    if not block:
        return base
    return f"{base.rstrip()}\n\n{block}"


def instruction_suffix() -> str:
    """Plain-text suffix for non-chat APIs (Hermes user prompt, OpenClaw message)."""
    block = intensity_block()
    if not block:
        return ""
    return f"\n\n---\n{block}"


def with_caveman(messages: Sequence[Any]) -> list[BaseMessage]:
    """Prepend caveman system message for user-visible model calls."""
    chain = list(convert_to_messages(messages))
    if not is_enabled():
        return chain
    return [SystemMessage(content=system_text()), *chain]
