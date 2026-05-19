"""Knowledge ingest without network."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sdl_agents.integrations.documents import load_directory, load_file, to_langchain_documents


@pytest.fixture(autouse=True)
def _no_web_seeds(monkeypatch):
    monkeypatch.setenv("SEED_URLS", "")


def test_load_markdown_fixture(tmp_path: Path):
    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nBody text.", encoding="utf-8")
    loaded = load_file(doc)
    assert len(loaded) == 1
    assert "Body text" in loaded[0].text


def test_load_directory_skips_unknown(tmp_path: Path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"\x00")
    docs = load_directory(tmp_path)
    assert len(docs) == 1


def test_to_langchain_documents(tmp_path: Path):
    (tmp_path / "b.txt").write_text("content", encoding="utf-8")
    lc = to_langchain_documents(load_directory(tmp_path))
    assert len(lc) == 1
    assert lc[0].page_content == "content"


def test_build_corpus_local_only(tmp_path: Path, monkeypatch):
    (tmp_path / "local.md").write_text("local corpus", encoding="utf-8")
    import importlib

    import sdl_agents.config as cfg

    importlib.reload(cfg)
    monkeypatch.setenv("LOCAL_DOCS_DIR", str(tmp_path))
    ingest = importlib.import_module("sdl_agents.agents.knowledge.ingest")
    monkeypatch.setattr(ingest, "load_web_seed_documents", lambda _urls: [])
    build_startup_corpus = ingest.build_startup_corpus

    docs = build_startup_corpus()
    assert any("local corpus" in d.page_content for d in docs)
