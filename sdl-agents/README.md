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

5. **Live integration (optional)** — set `SDL_INTEGRATION_MODE=live` and run Hermes + OpenClaw.

### Hermes Agent (Nous) — API Server

`sdl-agents` calls Hermes using the **OpenAI Chat Completions** API (`POST /v1/chat/completions`), not a custom `/v1/task` route. See the official guide: [Hermes API Server](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server).

1. In `~/.hermes/.env` on the machine running Hermes, enable the server and set a bearer key (example from docs):

   ```
   API_SERVER_ENABLED=true
   API_SERVER_KEY=change-me-local-dev
   ```

2. Start the gateway: `hermes gateway` — you should see e.g. `[API Server] API server listening on http://127.0.0.1:8642`.

3. In `sdl-agents/.env` (or env vars), align with that instance:

   | Variable | Purpose |
   |----------|---------|
   | `HERMES_BASE_URL` | OpenAI API root host (default `http://127.0.0.1:8642`); normalized to `…/v1` automatically |
   | `HERMES_API_KEY` | Same value as Hermes `API_SERVER_KEY` (Bearer token); omit only if your Hermes allows unauthenticated local access |
   | `HERMES_MODEL` | Model name in requests (default `hermes-agent`) |

4. Quick curl check (from Hermes docs):

   ```bash
   curl http://127.0.0.1:8642/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_SERVER_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"model\": \"hermes-agent\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"stream\": false}"
   ```

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
