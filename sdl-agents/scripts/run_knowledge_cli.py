#!/usr/bin/env python3
"""Run a query through the knowledge RAG agent (web + LOCAL_DOCS_DIR)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain.messages import HumanMessage

from sdl_agents.agents.knowledge import build_knowledge_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="User question")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream graph updates instead of printing final answer only",
    )
    args = parser.parse_args()

    query = args.query or input("Question: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        return 1

    graph = build_knowledge_graph()
    state = {"messages": [HumanMessage(content=query)]}

    if args.stream:
        for chunk in graph.stream(state):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print()
        return 0

    result = graph.invoke(state)
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
