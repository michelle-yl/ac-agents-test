"""Source normalization and rendering for user-facing answers."""

from __future__ import annotations

from typing import Any


def internal_source(label: str, **extra: Any) -> dict[str, Any]:
    return {"source_type": "internal", "label": label, **extra}


def external_source(label: str, **extra: Any) -> dict[str, Any]:
    return {"source_type": "external", "label": label, **extra}


def local_chunk_source(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    label = metadata.get("file") or metadata.get("source") or metadata.get("path") or "local corpus"
    return internal_source(
        str(label),
        file=metadata.get("file"),
        path=metadata.get("path"),
        chunk=chunk.get("text", "")[:200],
    )


def normalize_source(source: dict[str, Any], *, default_type: str = "internal") -> dict[str, Any]:
    out = dict(source)
    if "source_type" not in out:
        label = str(out.get("source") or out.get("url") or out.get("file") or out.get("path") or "")
        if label.startswith(("http://", "https://")):
            out["source_type"] = "external"
        elif default_type:
            out["source_type"] = default_type
    if "label" not in out:
        out["label"] = out.get("file") or out.get("source") or out.get("url") or out.get("path")
    return out


def append_sources_section(text: str, sources: list[dict[str, Any]] | None) -> str:
    body = text.rstrip()
    if "## Sources" in body:
        return body
    return f"{body}\n\n{format_sources_section(sources or [])}"


def format_sources_section(sources: list[dict[str, Any]]) -> str:
    internal: list[str] = []
    external: list[str] = []
    for source in sources:
        line = _format_source_line(source)
        if _source_type(source) == "external":
            external.append(line)
        else:
            internal.append(line)

    if not internal and not external:
        external.append("LLM synthesis only; no retrieved documents were cited.")

    lines = ["## Sources"]
    lines.append(_format_group("Internal sources", internal))
    lines.append("")
    lines.append(_format_group("External sources", external))
    return "\n".join(lines)


def _source_type(source: dict[str, Any]) -> str:
    raw = str(source.get("source_type") or source.get("type") or "internal").lower()
    if raw in {"external", "web", "internet", "llm"}:
        return "external"
    return "internal"


def _format_source_line(source: dict[str, Any]) -> str:
    label = (
        source.get("label")
        or source.get("file")
        or source.get("source")
        or source.get("url")
        or source.get("path")
        or source.get("tool")
        or "unknown source"
    )
    prefix = source.get("prefix")
    if prefix:
        return f"{prefix}: `{label}`"
    return f"`{label}`"


def _format_group(title: str, lines: list[str]) -> str:
    if not lines:
        return f"{title}: none"
    deduped = list(dict.fromkeys(lines))
    return title + ":\n" + "\n".join(f"- {line}" for line in deduped)
