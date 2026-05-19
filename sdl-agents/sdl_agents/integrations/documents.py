"""Shared document loading via LlamaIndex file readers (no unstructured)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sdl_agents.config import SUPPORTED_DOC_EXTENSIONS
from sdl_agents.logging_utils import get_logger

logger = get_logger("documents")

_TEXT_SUFFIXES = frozenset({".md", ".txt", ".csv", ".json"})


def _file_extractor() -> dict[str, Any]:
    from llama_index.readers.file import DocxReader, PDFReader

    return {
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
    }


def load_directory(directory: Path) -> list[Any]:
    """Load supported files under directory as LlamaIndex Document objects."""
    from llama_index.core import Document

    if not directory.is_dir():
        return []

    anchor = directory.resolve()
    docs: list[Document] = []

    for path in sorted(anchor.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_DOC_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(anchor)
        except ValueError:
            continue
        docs.extend(load_file(path))

    return docs


def load_file(path: Path) -> list[Any]:
    """Load one file; returns LlamaIndex Document list (may be empty)."""
    from llama_index.core import Document

    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOC_EXTENSIONS:
        return []

    if suffix in _TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Skip %s: %s", path, exc)
            return []
        return [
            Document(
                text=text,
                metadata={
                    "file": path.name,
                    "path": str(path),
                    "source": str(path),
                    "source_type": "local",
                },
            )
        ]

    try:
        from llama_index.readers.file import DocxReader, PDFReader

        reader = PDFReader() if suffix == ".pdf" else DocxReader()
        loaded = reader.load_data(file=path)
    except Exception as exc:
        logger.warning("Reader failed on %s: %s", path, exc)
        return []

    for doc in loaded:
        md = dict(doc.metadata or {})
        md.setdefault("file", path.name)
        md.setdefault("path", str(path))
        md.setdefault("source", str(path))
        md.setdefault("source_type", "local")
        doc.metadata = md
    return loaded


def load_directory_simple(directory: Path) -> list[Any]:
    """Load via SimpleDirectoryReader when a batch load is preferred."""
    from llama_index.core import SimpleDirectoryReader

    if not directory.is_dir():
        return []
    try:
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            required_exts=sorted(SUPPORTED_DOC_EXTENSIONS),
            file_extractor=_file_extractor(),
            recursive=True,
        )
        docs = reader.load_data()
        for doc in docs:
            md = dict(doc.metadata or {})
            md.setdefault("source_type", "local")
            doc.metadata = md
        return docs
    except Exception as exc:
        logger.warning("SimpleDirectoryReader failed on %s: %s", directory, exc)
        return load_directory(directory)


def to_langchain_documents(docs: list[Any]) -> list[Any]:
    """Convert LlamaIndex documents to langchain_core Document."""
    from langchain_core.documents import Document as LCDocument

    out: list[Any] = []
    for doc in docs:
        if hasattr(doc, "text"):
            text = doc.text
        elif hasattr(doc, "get_content"):
            text = doc.get_content()
        else:
            text = str(doc)
        md = dict(getattr(doc, "metadata", None) or {})
        out.append(LCDocument(page_content=str(text), metadata=md))
    return out


def ingest_log_summary(
    *,
    web_count: int,
    local_count: int,
    local_root: Path | None,
    seed_urls: list[str],
) -> None:
    if web_count:
        print(
            f"[ingest] web docs: {web_count} parts from {len(seed_urls)} URL(s)",
            file=sys.stderr,
        )
    if local_root is not None:
        print(
            f"[ingest] local docs: {local_count} parts under {local_root}",
            file=sys.stderr,
        )
