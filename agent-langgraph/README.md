# Moved to sdl-agents

The agentic RAG LangGraph app formerly in `agent_langgraph.py` now lives in:

- **Module:** `sdl-agents/sdl_agents/agents/knowledge/`
- **CLI:** `sdl-agents/scripts/run_knowledge_cli.py`
- **Config:** single `sdl-agents/.env` (see `sdl-agents/.env.example`)

```bash
cd sdl-agents
python scripts/run_knowledge_cli.py "Your question here"
```

Lab multi-agent orchestrator (database, research, Hermes, OpenClaw):

```bash
python scripts/run_cli.py "Your question here"
```
