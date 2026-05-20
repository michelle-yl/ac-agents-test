#!/usr/bin/env python3
"""Background monitoring watcher: ingest JSON, diff snapshots, update cache."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdl_agents.config import (  # noqa: E402
    MONITOR_INGEST_INTERVAL_SEC,
    MONITOR_POLL_INTERVAL_SEC,
    monitor_json_dir,
)
from sdl_agents.monitoring.watcher import refresh_state  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_monitor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh (ingest + poll) then exit",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip JSON ingest; only poll Postgres",
    )
    args = parser.parse_args()

    json_dir = monitor_json_dir()
    logger.info("monitor json dir: %s", json_dir)
    logger.info(
        "intervals: ingest=%ss poll=%ss",
        MONITOR_INGEST_INTERVAL_SEC,
        MONITOR_POLL_INTERVAL_SEC,
    )

    last_ingest = 0.0

    def tick(*, force_ingest: bool = False) -> None:
        nonlocal last_ingest
        now = time.monotonic()
        do_ingest = force_ingest or (
            not args.no_ingest and (now - last_ingest) >= MONITOR_INGEST_INTERVAL_SEC
        )
        if do_ingest:
            last_ingest = now
        state = refresh_state(run_ingest_step=do_ingest)
        logger.info("cache updated: %s", state.summary)
        if state.open_alerts:
            for alert in state.open_alerts[:5]:
                logger.warning("%s", alert.message)

    if args.once:
        tick(force_ingest=not args.no_ingest)
        return 0

    tick(force_ingest=not args.no_ingest)
    while True:
        time.sleep(MONITOR_POLL_INTERVAL_SEC)
        tick()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
