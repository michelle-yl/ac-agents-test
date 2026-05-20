#!/usr/bin/env python3
"""Run a single query through the SDL orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain.messages import HumanMessage

from sdl_agents.orchestrator.graph import build_orchestrator_graph


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="User question")
    args = parser.parse_args()

    query = args.query or input("Question: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        return 1

    graph = build_orchestrator_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "intent": "general",
            "route_reason": "",
            "db_payload": None,
            "monitor_cache_used": False,
            "research_payload": None,
            "research_flags": {},
            "errors": [],
        }
    )
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
