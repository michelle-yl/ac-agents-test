# SDL agent hierarchy

```
orchestrator
|--- database agent
|--- deep research orchestrator agent
    |--- academic literature research agent (articles, papers)
    |--- safety protocols research agent (laboratory biosafety requirements, MSDS, PPE, OSHA)
    |--- experimental procedures agent (liquid handling, pipetting, concentration and volume calculations)
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

Run and test: [sdl-agents/README.md](sdl-agents/README.md). Reference architecture: [michelle-yl/multi-agent-test](https://github.com/michelle-yl/multi-agent-test).

