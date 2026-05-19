# SDL Multi-Agent System

LangGraph orchestrator for lab monitoring (PostgreSQL replica) and deep research (Hermes, LlamaIndex, OpenClaw).

Hierarchy is defined in [../agents.md](../agents.md). Architectural reference: [michelle-yl/multi-agent-test](https://github.com/michelle-yl/multi-agent-test).

## Prerequisites

1. **Python 3.11+** and dependencies:

   ```bash
   cd sdl-agents
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API keys** — reuse [../agent-langgraph/.env](../agent-langgraph/.env) (`ANTHROPIC_API_KEY`, `HF_TOKEN`). Optional overrides in `sdl-agents/.env` (see `.env.example`).

3. **Monitoring database**:

   ```bash
   cd ../db
   docker compose up -d
   pip install -r requirements-db.txt
   python seed/load_monitoring.py --truncate
   ```

4. **Optional corpora** — fixture docs in `corpus/safety/` and `corpus/procedures/`. Build indices:

   ```bash
   python scripts/build_indices.py
   ```

5. **Live integration (optional)** — set `SDL_INTEGRATION_MODE=live` and run Hermes + OpenClaw (see plan: Future live integration requirements).

## Run CLI

```bash
python scripts/run_cli.py "Which devices are offline?"
python scripts/run_cli.py "What PPE is required for BSL-2 work?"
```

## Tests

```bash
pytest tests/ -v
pytest tests/stage1/ -v
pytest -m "not integration" -v
pytest -m integration -v   # requires live Hermes/OpenClaw + SDL_INTEGRATION_MODE=live
```

## Implementation map

| Agent | Runtime | Module |
|-------|---------|--------|
| Orchestrator | LangGraph | `sdl_agents/orchestrator/graph.py` |
| Database | LangGraph + psycopg | `sdl_agents/agents/database/` |
| Deep research orchestrator | LangGraph subgraph | `sdl_agents/agents/research/graph.py` |
| Academic literature | Nous Hermes | `sdl_agents/agents/research/academic_hermes.py` |
| Safety protocols | Hermes + LlamaIndex | `sdl_agents/agents/research/safety_hermes_rag.py` |
| Experimental procedures | OpenClaw + LlamaIndex | `sdl_agents/agents/research/experimental_openclaw_rag.py` |

Default integration mode is **mock** (`SDL_INTEGRATION_MODE=mock`) so tests and CLI work without Hermes/OpenClaw.
