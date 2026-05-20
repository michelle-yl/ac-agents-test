# SDL Multi-Agent System

LangGraph orchestrator for lab monitoring (PostgreSQL replica) and deep research (Hermes + LlamaIndex RAG for academic, safety, and procedures).

Hierarchy is defined in [../agents.md](../agents.md). Architectural reference: [michelle-yl/multi-agent-test](https://github.com/michelle-yl/multi-agent-test).

## Prerequisites

1. **Python 3.11+** and dependencies:

   ```bash
   cd sdl-agents
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configuration** — copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`, optional `HF_TOKEN`, and integration URLs.

3. **Monitoring database**:

   ```bash
   cd ../db
   docker compose up -d
   pip install -r requirements-db.txt
   python seed/load_monitoring.py --truncate
   ```

4. **Optional corpora** — local docs in `corpus/safety/`, `corpus/procedures/`, and `corpus/academic/` (PDF, MD, etc.). Build indices:

   ```bash
   python scripts/build_indices.py
   ```

   Re-run after adding or changing files under `corpus/academic/`.

5. **Live integration (optional)** — set `SDL_INTEGRATION_MODE=live` and run the Hermes gateway (`hermes gateway`, port 8642).

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
   | `HERMES_WSL_AUTO` | On Windows, retry Hermes via WSL eth0 IP when `localhost:8642` fails (default `1`) |

4. Quick curl check (from Hermes docs):

   ```bash
   curl http://127.0.0.1:8642/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_SERVER_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"model\": \"hermes-agent\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"stream\": false}"
   ```

If Hermes is not running, safety and academic queries still return excerpts from local corpora (`corpus/safety/`, `corpus/academic/` + `python scripts/build_indices.py`). Live Hermes improves synthesis when the gateway is up.

### OpenClaw gateway (optional, not used by default)

Research **experimental procedures** use Hermes like academic and safety agents. OpenClaw remains available in `sdl_agents/integrations/openclaw_client.py` for separate deployments. See [OpenClaw HTTP API](https://docs.openclaw.ai/gateway/openai-http-api) if you wire it manually.

1. In `openclaw.json`, enable chat completions and set auth:

   ```json5
   {
     gateway: {
       auth: { mode: "token", token: "YOUR_TOKEN" },
       http: { endpoints: { chatCompletions: { enabled: true } } },
     },
   }
   ```

2. In `sdl-agents/.env`: `OPENCLAW_BASE_URL=http://127.0.0.1:18789`, `OPENCLAW_GATEWAY_TOKEN=<same token>`, `OPENCLAW_MODEL=openclaw/default`.

3. **Windows + WSL Docker:** if `127.0.0.1:18789` is unreachable from PowerShell but the gateway is healthy inside WSL, leave `OPENCLAW_WSL_AUTO=1` (default). SDL retries via the WSL eth0 IP. Or set `OPENCLAW_BASE_URL` to that IP explicitly.

4. Smoke test (replace token):

   ```bash
   curl -sS http://127.0.0.1:18789/v1/chat/completions \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model":"openclaw/default","messages":[{"role":"user","content":"hi"}]}'
   ```

   A `404` here means `chatCompletions` is still disabled — fix config and restart the gateway container.

## Run CLI

**Monitoring watcher** (ingest Angie JSON, diff snapshots, hot cache — no LLM):

```bash
python scripts/run_monitor.py --once
python scripts/run_monitor.py
```

Point `MONITOR_JSON_DIR` at your Angie `obsidian-vault/monitoring` folder. On existing Postgres volumes, run `db/scripts/apply-002-events.ps1` (Windows) or `apply-002-events.sh` (bash) — see [db/README.md](../db/README.md).

**Lab orchestrator** (monitoring DB, safety, literature, procedures):

```bash
python scripts/run_cli.py "Which devices are offline?"
python scripts/run_cli.py "What PPE is required for BSL-2 work?"
```

Simple status/temp queries use **0 LLM calls** when the watcher cache is fresh (`ROUTER_KEYWORD_FIRST`, `DB_FAST_PATH_ENABLED`, orchestrator cache short-circuit).

**Knowledge RAG** (web `SEED_URLS` + `LOCAL_DOCS_DIR`; separate from lab router):

```bash
python scripts/run_knowledge_cli.py "Your question about indexed docs"
```

Supported local types: `.md`, `.txt`, `.csv`, `.json`, `.pdf`, `.docx` (via LlamaIndex readers; no unstructured).

## Tests

```bash
pytest tests/ -v
pytest tests/stage1/ -v
pytest -m "not integration" -v
pytest -m integration -v   # requires live Hermes + SDL_INTEGRATION_MODE=live
```

## Implementation map

| Agent | Runtime | Module |
|-------|---------|--------|
| Orchestrator | LangGraph | `sdl_agents/orchestrator/graph.py` |
| Monitoring watcher | Background loop | `sdl_agents/monitoring/` |
| Database agent | LangGraph + psycopg | `sdl_agents/agents/database/` |
| Research subagent router | Keyword + LLM (orchestrator) | `sdl_agents/agents/research/research_route_router.py` |
| Academic literature | Hermes + LlamaIndex | `sdl_agents/agents/research/academic_hermes.py` |
| Safety protocols | Hermes + LlamaIndex | `sdl_agents/agents/research/safety_hermes_rag.py` |
| Experimental procedures | Hermes + LlamaIndex | `sdl_agents/agents/research/experimental_procedures_hermes.py` |
| Knowledge RAG | LangGraph + HF embeddings | `sdl_agents/agents/knowledge/` |

Default integration mode is **mock** (`SDL_INTEGRATION_MODE=mock`) so tests and CLI work without a live Hermes gateway.

### Caveman mode

User-facing replies use [caveman](https://github.com/juliusbrussee/caveman) terse style via `sdl_agents/caveman.py` (on by default):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAVEMAN_ENABLED` | `1` | Set `0` to disable |
| `CAVEMAN_LEVEL` | `full` | `lite`, `full`, `ultra`, `wenyan-lite`, `wenyan-full`, `wenyan-ultra` |

Router and research subagent classification stay normal for structured JSON.

### Monitoring and DB fast path

| Variable | Default | Purpose |
|----------|---------|---------|
| `MONITOR_JSON_DIR` | `../db/seed/data/monitoring` | Angie monitoring JSON source |
| `MONITOR_INGEST_INTERVAL_SEC` | `60` | Append ingest interval |
| `MONITOR_POLL_INTERVAL_SEC` | `15` | Poll/diff interval |
| `MONITOR_CACHE_MAX_AGE_SEC` | `120` | Orchestrator cache trust window |
| `ROUTER_KEYWORD_FIRST` | `1` | Keyword router before LLM |
| `DB_FAST_PATH_ENABLED` | `1` | Deterministic DB tool dispatch |
| `DB_AGENT_MODEL` | `ANTHROPIC_GRADER_MODEL` | Model for remaining DB LLM steps |
| `DB_FAST_PATH_MAX_ROWS` | `100` | Max rows for template answers |
