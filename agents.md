# SDL agent hierarchy

```
orchestrator (run_cli.py)
|--- 1. database route
    |--- monitoring watcher (run_monitor.py) — ingest JSON, diff snapshots, hot cache; not LLM
    |--- database agent — fast path (0 LLM) or LangGraph + PostgreSQL
|--- 2. research route (orchestrator classifies, then dispatches)
    |--- academic literature agent (Hermes + corpus/academic)
    |--- safety protocols agent (Hermes + corpus/safety)
    |--- experimental procedures agent (Hermes + corpus/procedures)

knowledge RAG (run_knowledge_cli.py) — general web + LOCAL_DOCS_DIR; not routed by orchestrator
```

## Implementation map

| Agent | Runtime | Entry |
|-------|---------|--------|
| Orchestrator | LangGraph | [sdl-agents/sdl_agents/orchestrator/graph.py](sdl-agents/sdl_agents/orchestrator/graph.py) |
| Monitoring watcher | Background loop + cache | [sdl-agents/sdl_agents/monitoring/](sdl-agents/sdl_agents/monitoring/) |
| Database agent | LangGraph + monitoring PostgreSQL | [sdl-agents/sdl_agents/agents/database/](sdl-agents/sdl_agents/agents/database/) |
| Research subagent router | Keyword + LLM | [sdl-agents/sdl_agents/agents/research/research_route_router.py](sdl-agents/sdl_agents/agents/research/research_route_router.py) |
| Academic literature | Hermes + LlamaIndex | [sdl-agents/sdl_agents/agents/research/academic_hermes.py](sdl-agents/sdl_agents/agents/research/academic_hermes.py) |
| Safety protocols | Hermes + LlamaIndex | [sdl-agents/sdl_agents/agents/research/safety_hermes_rag.py](sdl-agents/sdl_agents/agents/research/safety_hermes_rag.py) |
| Experimental procedures | Hermes + LlamaIndex | [sdl-agents/sdl_agents/agents/research/experimental_procedures_hermes.py](sdl-agents/sdl_agents/agents/research/experimental_procedures_hermes.py) |
| Knowledge RAG | LangGraph + local embeddings | [sdl-agents/sdl_agents/agents/knowledge/](sdl-agents/sdl_agents/agents/knowledge/) |

Configuration: single [sdl-agents/.env](sdl-agents/.env) (template: [sdl-agents/.env.example](sdl-agents/.env.example)).

Run and test: [sdl-agents/README.md](sdl-agents/README.md). Reference architecture: [michelle-yl/multi-agent-test](https://github.com/michelle-yl/multi-agent-test).

## Caveman mode

All user-facing agents use [caveman](https://github.com/juliusbrussee/caveman) terse output by default (shared module: `sdl_agents/caveman.py`). Set `CAVEMAN_ENABLED=0` to disable. Levels: `lite`, `full` (default), `ultra`, `wenyan-lite`, `wenyan-full`, `wenyan-ultra` via `CAVEMAN_LEVEL`. Routing calls stay verbose for structured JSON.

## Monitoring fast path

**Windows (PowerShell)** — do not use bash `<` redirect for migrations:

```powershell
cd db
docker compose up -d
.\scripts\apply-002-events.ps1
python seed\load_monitoring.py --truncate
cd ..\sdl-agents
python scripts\run_monitor.py --once
python scripts\run_cli.py "Which devices are offline?"
```

**Linux/macOS:** `db/scripts/apply-002-events.sh` then the same Python steps.

Use `ROUTER_KEYWORD_FIRST=1` and `DB_FAST_PATH_ENABLED=1` (defaults). Simple monitoring queries use keyword routing, monitor cache, or deterministic SQL; remaining DB LLM steps use `DB_AGENT_MODEL` (default Haiku).
