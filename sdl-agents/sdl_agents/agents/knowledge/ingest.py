"""Corpus ingest for the knowledge RAG agent (web seeds + local docs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from sdl_agents.config import SDL_AGENTS_ROOT, local_docs_dir, seed_urls
from sdl_agents.integrations.documents import (
    ingest_log_summary,
    load_directory,
    to_langchain_documents,
)


def _mark_web_documents(docs: list[Document]) -> None:
    for d in docs:
        md = dict(d.metadata) if d.metadata else {}
        md.setdefault("source_type", "web")
        d.metadata = md


def load_web_seed_documents(url_list: list[str]) -> list[Document]:
    out: list[Document] = []
    for url in url_list:
        try:
            batch = WebBaseLoader(url).load()
        except Exception as exc:
            print(f"[ingest] WebBaseLoader skip {url!r}: {exc}", file=sys.stderr)
            continue
        _mark_web_documents(batch)
        out.extend(batch)
    return out


def _is_under_anchor(path: Path, anchor: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(anchor.resolve(strict=False))
        return True
    except ValueError:
        return False


def read_access_anchor() -> Path:
    root = local_docs_dir()
    return root if root is not None else SDL_AGENTS_ROOT


def resolve_user_local_path(raw: str) -> tuple[Path | None, str | None]:
    anchor = read_access_anchor()
    anchor_resolved = anchor.resolve(strict=False)
    candidate = Path(raw.strip()).expanduser()
    resolved = (
        candidate if candidate.is_absolute() else (anchor / candidate)
    ).resolve(strict=False)

    try:
        if not _is_under_anchor(resolved, anchor_resolved):
            return (
                None,
                f"Path must be under {anchor_resolved} "
                "(set LOCAL_DOCS_DIR for a docs folder)",
            )
    except (OSError, RuntimeError):
        return None, f"Could not normalize path: {raw!r}"

    if not resolved.is_file():
        return None, f"File not found: {resolved}"
    return resolved, None


def build_startup_corpus() -> list[Document]:
    urls = seed_urls()
    web_docs = load_web_seed_documents(urls)
    local_docs: list[Document] = []
    docs_root = local_docs_dir()
    if docs_root is not None:
        local_docs = to_langchain_documents(load_directory(docs_root))
    elif os.environ.get("LOCAL_DOCS_DIR", "").strip():
        print(
            "[ingest] LOCAL_DOCS_DIR is set but not a usable directory; "
            "local ingest skipped.",
            file=sys.stderr,
        )

    ingest_log_summary(
        web_count=len(web_docs),
        local_count=len(local_docs),
        local_root=docs_root,
        seed_urls=urls,
    )
    return [*web_docs, *local_docs]
