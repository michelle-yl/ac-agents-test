"""Experimental procedures: LlamaIndex + OpenClaw with dilution math helper."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from sdl_agents.integrations.llamaindex_indices import search_procedures
from sdl_agents.integrations.openclaw_client import query_procedures


def dilution_calc(c1: float, v1: float, c2: float | None = None, v2: float | None = None) -> dict[str, float]:
    """C1V1=C2V2 — provide three values to solve for the fourth."""
    if c2 is None and v2 is not None:
        c2 = c1 * v1 / v2
    elif v2 is None and c2 is not None:
        v2 = c1 * v1 / c2
    elif c2 is not None and v2 is not None:
        pass
    else:
        raise ValueError("Provide three of c1, v1, c2, v2")
    return {"c1": c1, "v1": v1, "c2": c2, "v2": v2}


def _extract_dilution_from_query(query: str) -> dict[str, float] | None:
    """Simple pattern: 'C1=10 V1=100 C2=1' style hints in query."""
    nums = re.findall(r"(\d+\.?\d*)", query)
    if len(nums) >= 3 and "dilut" in query.lower():
        c1, v1, c2 = float(nums[0]), float(nums[1]), float(nums[2])
        return dilution_calc(c1, v1, c2=c2)
    return None


async def run_experimental(query: str, db_context: str = "") -> dict[str, Any]:
    chunks = search_procedures(query, top_k=3)
    context = json.dumps(chunks, indent=2) if chunks else ""
    dilution = _extract_dilution_from_query(query)
    if dilution:
        context += f"\n\nDilution calculation (C1V1=C2V2): {json.dumps(dilution)}"

    openclaw = await query_procedures(query, context=context)
    errors: list[str] = []
    if not openclaw.get("ok"):
        errors.append(openclaw.get("error", "OpenClaw unavailable; LlamaIndex-only response"))

    text_parts = []
    if chunks:
        text_parts.append("From procedure documents:\n" + "\n".join(c["text"][:500] for c in chunks))
    if openclaw.get("text"):
        text_parts.append("From OpenClaw:\n" + openclaw["text"])
    if not text_parts:
        text_parts.append(
            "No indexed procedure documents. Add SOPs to corpus/procedures/ and run scripts/build_indices.py"
        )

    return {
        "text": "\n\n".join(text_parts),
        "citations": chunks,
        "dilution": dilution,
        "openclaw_skill": openclaw.get("skill"),
        "errors": errors,
        "specialist": "procedures",
    }


def run_experimental_sync(query: str, db_context: str = "") -> dict[str, Any]:
    return asyncio.run(run_experimental(query, db_context))
