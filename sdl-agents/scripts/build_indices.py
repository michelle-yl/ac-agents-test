#!/usr/bin/env python3
"""Build LlamaIndex persisted stores for safety and procedure corpora."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdl_agents.config import INDICES_DIR, PROCEDURES_DOCS_DIR, SAFETY_DOCS_DIR
from sdl_agents.integrations.llamaindex_indices import get_index


def main() -> int:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    for name, docs_dir in (("safety", SAFETY_DOCS_DIR), ("procedures", PROCEDURES_DOCS_DIR)):
        print(f"Building index '{name}' from {docs_dir}...")
        idx = get_index(name, docs_dir)
        if idx is None:
            print(f"  No documents found under {docs_dir}")
        else:
            print(f"  Index '{name}' ready under {INDICES_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
