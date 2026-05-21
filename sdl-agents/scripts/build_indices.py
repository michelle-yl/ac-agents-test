#!/usr/bin/env python3
"""Build LlamaIndex persisted stores for safety, academic, and procedure corpora."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdl_agents.config import (
    ACADEMIC_DOCS_DIR,
    INDICES_DIR,
    PROCEDURES_DOCS_DIR,
    SAFETY_DOCS_DIR,
    SUPPORTED_DOC_EXTENSIONS,
)
from sdl_agents.integrations.llamaindex_indices import get_index

MANIFEST_NAME = ".corpus_manifest.json"
MANIFEST_VERSION = 1


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def _corpus_snapshot(docs_dir: Path) -> dict[str, Any]:
    root = docs_dir.resolve()
    files: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
                continue
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
            )
    return {"version": MANIFEST_VERSION, "files": files}


def _load_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(index_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(index_dir: Path, snapshot: dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(index_dir).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rebuild_if_stale(name: str, docs_dir: Path) -> dict[str, Any]:
    index_dir = INDICES_DIR / name
    snapshot = _corpus_snapshot(docs_dir)
    existing = _load_manifest(index_dir)
    if index_dir.is_dir() and existing != snapshot:
        print(f"  Corpus changed; rebuilding {index_dir}")
        shutil.rmtree(index_dir)
    return snapshot


def main() -> int:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    for name, docs_dir in (
        ("safety", SAFETY_DOCS_DIR),
        ("academic", ACADEMIC_DOCS_DIR),
        ("procedures", PROCEDURES_DOCS_DIR),
    ):
        print(f"Building index '{name}' from {docs_dir}...")
        snapshot = _rebuild_if_stale(name, docs_dir)
        idx = get_index(name, docs_dir)
        if idx is None:
            print(f"  No documents found under {docs_dir}")
        else:
            index_dir = INDICES_DIR / name
            _write_manifest(index_dir, snapshot)
            print(f"  Index '{name}' ready under {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
