"""Append-only ingest from Angie monitoring JSON."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sdl_agents.config import DB_DIR, monitor_json_dir


def run_ingest(json_dir: Path | None = None) -> int:
    """Run load_monitoring.py without --truncate (append snapshots)."""
    directory = json_dir or monitor_json_dir()
    script = DB_DIR / "seed" / "load_monitoring.py"
    if not script.is_file():
        raise FileNotFoundError(f"Loader not found: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--json-dir",
        str(directory),
    ]
    result = subprocess.run(cmd, cwd=str(DB_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ingest failed ({result.returncode}): {stderr}")
    return result.returncode
