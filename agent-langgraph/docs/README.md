# OpenClaw Lab Gateway

Docker configuration for [OpenClaw](https://docs.openclaw.ai/) with custom skills for laboratory AI integration.

## Features

- 🤖 **OpenClaw Gateway** - AI chat platform with multi-channel support
- 🔬 **Lab Database Integration** - Query lab equipment, sensors, imaging databases
- 📚 **RAG Knowledge Base** - Search biosafety protocols, manuals, research papers
- 📊 **Data Visualization** - Generate 25+ chart types via MCP
- 🎯 **Bayesian Optimization** - Experiment optimization campaigns

## Custom Skills

| Skill | Description | Endpoint |
|-------|-------------|----------|
| 🔬 `lab-db` | Query 3 lab databases (94+ tables) | 192.168.11.27:8020-8022 |
| 📚 `lab-rag` | Search knowledge base (122 docs, 13K chunks) | 192.168.142.185:8005 |
| 📊 `charts` | Generate visualizations (AntV MCP) | localhost:1122 |
| 🎯 `bo-mcp` | Bayesian optimization campaigns | localhost:8046 |

## Quick Start

### 1. Configure Environment

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
SLACK_BOT_TOKEN=xoxb-...        # Optional
SLACK_APP_TOKEN=xapp-...        # Optional
```

### 2. Build and Start

```bash
docker compose up -d
```

### 3. Access Dashboard

```
http://localhost:18789/
```

Get tokenized URL:
```bash
docker exec openclaw-gateway openclaw dashboard --no-open
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| OpenClaw Gateway | 18789 | AI chat gateway |
| MCP Chart Server | 1122 | Data visualization |

## Skills Directory

Skills are stored in `./skills/` and mounted into the container:

```
skills/
├── lab-db/SKILL.md      # Database queries
├── lab-rag/SKILL.md     # RAG knowledge search
├── charts/SKILL.md      # Chart generation
└── bo-mcp/SKILL.md      # Bayesian optimization
```

To add a new skill, create a folder with a `SKILL.md` file following the format:

```markdown
---
name: my-skill
description: Description of the skill
metadata: {"openclaw":{"emoji":"🔧"}}
---

# Skill Name

Documentation and examples...
```

## Commands

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f openclaw

# List skills
docker exec openclaw-gateway openclaw skills list

# Access shell
docker exec -it openclaw-gateway /bin/sh

# Restart
docker compose restart
```

## Example Queries

**Database + Charts:**
> "Query device counts by status and create a pie chart"

**RAG Search:**
> "Find the biosafety protocol for BSL-2 work"

**Optimization:**
> "Create an optimization campaign for cell culture parameters"

## Network Architecture

See [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) for the full lab network topology.

## Persistent Data

Data is stored in Docker volumes:
- `openclaw-data` - Credentials, agents, workspace, chat history

Skills are version controlled in `./skills/` directory.

## Documentation

- [OpenClaw Docs](https://docs.openclaw.ai/)
- [AntV MCP Chart](https://github.com/antvis/mcp-server-chart)
