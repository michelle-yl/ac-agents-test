# SDL agent hierarchy

```
orchestrator (run_cli.py)
|--- database agent
|--- deep research orchestrator agent
    |--- academic literature research agent (articles, papers)
    |--- safety protocols research agent (laboratory biosafety requirements, MSDS, PPE, OSHA)
    |--- experimental procedures agent (liquid handling, pipetting, concentration and volume calculations)

knowledge RAG (run_knowledge_cli.py) — general web + LOCAL_DOCS_DIR; not routed by orchestrator
```

## Implementation map

| Agent | Runtime | Entry |
|-------|---------|--------|
| Orchestrator | LangGraph | [sdl-agents/sdl_agents/orchestrator/graph.py](sdl-agents/sdl_agents/orchestrator/graph.py) |
| Database | LangGraph + monitoring PostgreSQL | [sdl-agents/sdl_agents/agents/database/](sdl-agents/sdl_agents/agents/database/) |
| Deep research orchestrator | LangGraph subgraph | [sdl-agents/sdl_agents/agents/research/graph.py](sdl-agents/sdl_agents/agents/research/graph.py) |
| Academic literature | Nous Hermes | [sdl-agents/sdl_agents/agents/research/academic_hermes.py](sdl-agents/sdl_agents/agents/research/academic_hermes.py) |
| Safety protocols | Hermes + LlamaIndex | [sdl-agents/sdl_agents/agents/research/safety_hermes_rag.py](sdl-agents/sdl_agents/agents/research/safety_hermes_rag.py) |
| Experimental procedures | OpenClaw + LlamaIndex | [sdl-agents/sdl_agents/agents/research/experimental_openclaw_rag.py](sdl-agents/sdl_agents/agents/research/experimental_openclaw_rag.py) |
| Knowledge RAG | LangGraph + local embeddings | [sdl-agents/sdl_agents/agents/knowledge/](sdl-agents/sdl_agents/agents/knowledge/) |

Configuration: single [sdl-agents/.env](sdl-agents/.env) (template: [sdl-agents/.env.example](sdl-agents/.env.example)). Former [agent-langgraph/](agent-langgraph/) app is retired; see [agent-langgraph/README.md](agent-langgraph/README.md).

Run and test: [sdl-agents/README.md](sdl-agents/README.md). Reference architecture: [michelle-yl/multi-agent-test](https://github.com/michelle-yl/multi-agent-test).

## Caveman mode

All user-facing agents use [caveman](https://github.com/juliusbrussee/caveman) terse output by default (shared module: `sdl_agents/caveman.py`). Set `CAVEMAN_ENABLED=0` to disable. Levels: `lite`, `full` (default), `ultra`, `wenyan-lite`, `wenyan-full`, `wenyan-ultra` via `CAVEMAN_LEVEL`. Routing/grading calls stay verbose for structured JSON.

