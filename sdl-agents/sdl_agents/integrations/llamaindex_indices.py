"""LlamaIndex retrieval for safety and procedure corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdl_agents.config import (
    HF_EMBEDDING_MODEL,
    INDICES_DIR,
    PROCEDURES_DOCS_DIR,
    SAFETY_DOCS_DIR,
)
from sdl_agents.logging_utils import get_logger

logger = get_logger("llamaindex")

_index_cache: dict[str, Any] = {}


def _load_documents(directory: Path) -> list[Any]:
    from llama_index.core import Document

    docs: list[Document] = []
    if not directory.is_dir():
        return docs
    for path in directory.rglob("*"):
        if path.suffix.lower() in {".md", ".txt", ".csv"}:
            try:
                docs.append(
                    Document(
                        text=path.read_text(encoding="utf-8", errors="replace"),
                        metadata={"file": str(path.name), "path": str(path)},
                    )
                )
            except OSError as exc:
                logger.warning("Skip %s: %s", path, exc)
    return docs


def get_index(name: str, docs_dir: Path):
    if name in _index_cache:
        return _index_cache[name]

    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)
    persist_dir = INDICES_DIR / name
    if persist_dir.is_dir():
        from llama_index.core import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
    else:
        documents = _load_documents(docs_dir)
        if not documents:
            _index_cache[name] = None
            return None
        index = VectorStoreIndex.from_documents(documents)
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))

    _index_cache[name] = index
    return index


def retrieve(name: str, docs_dir: Path, query: str, top_k: int = 3) -> list[dict[str, Any]]:
    index = get_index(name, docs_dir)
    if index is None:
        return []

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [
        {
            "text": n.get_content(),
            "score": float(n.score or 0.0),
            "metadata": dict(n.metadata or {}),
        }
        for n in nodes
    ]


def search_safety(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    return retrieve("safety", SAFETY_DOCS_DIR, query, top_k)


def search_procedures(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    return retrieve("procedures", PROCEDURES_DOCS_DIR, query, top_k)
