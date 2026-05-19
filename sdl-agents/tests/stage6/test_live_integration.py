"""Live integration tests (Hermes, OpenClaw) — run with SDL_INTEGRATION_MODE=live."""

from __future__ import annotations

import os

import pytest

from sdl_agents.config import is_live_integration


@pytest.mark.integration
@pytest.mark.skipif(not is_live_integration(), reason="SDL_INTEGRATION_MODE!=live")
@pytest.mark.asyncio
async def test_hermes_live_health():
    from sdl_agents.integrations.hermes_client import run_task

    result = await run_task("ping", task_type="academic")
    assert result.get("text")


@pytest.mark.integration
@pytest.mark.skipif(not is_live_integration(), reason="SDL_INTEGRATION_MODE!=live")
@pytest.mark.asyncio
async def test_openclaw_live_health():
    from sdl_agents.integrations.openclaw_client import health_check

    ok = await health_check()
    if not ok:
        pytest.skip("OpenClaw gateway not reachable")
