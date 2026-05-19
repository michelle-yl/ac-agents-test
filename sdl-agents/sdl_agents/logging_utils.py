"""Simple structured logging for agents."""

from __future__ import annotations

import logging
import os

_LOG_LEVEL = os.environ.get("SDL_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"sdl_agents.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
    return logger
