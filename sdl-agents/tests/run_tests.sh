#!/usr/bin/env bash
# Run SDL agents test suite (pass stage name or pytest args)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
if [[ -x .venv/Scripts/python.exe ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi
if [[ $# -eq 0 ]]; then
  exec "$PY" -m pytest tests/ -v -m "not integration"
elif [[ -d "tests/$1" ]]; then
  exec "$PY" -m pytest "tests/$1" -v "${@:2}"
else
  exec "$PY" -m pytest tests/ -v "$@"
fi
