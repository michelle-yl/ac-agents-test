# Angie monitoring PostgreSQL replica

Local PostgreSQL database that stores [michelle-yl/angie](https://github.com/michelle-yl/angie) `obsidian-vault/monitoring/*.json` in normalized tables.

## Quick start

```bash
cd db
cp .env.example .env
docker compose up -d
pip install -r requirements-db.txt
python seed/load_monitoring.py --truncate
```

Wait until the container is healthy (`docker compose ps`).

## Connection

```
postgresql://angie:angie@localhost:5433/angie_monitoring_replica
```

Override via `.env` or `DATABASE_URL`.

## Seed data

Committed snapshots live in `seed/data/monitoring/` (from angie vault). To refresh:

```powershell
Copy-Item path\to\angie\obsidian-vault\monitoring\*.json seed\data\monitoring\
python seed/load_monitoring.py --truncate
```

Or set `MONITORING_JSON_DIR` to another folder.

### Loader flags

| Flag | Effect |
|------|--------|
| `--truncate` | Remove prior snapshots, then load (default for clean re-seed) |
| `--json-dir PATH` | Override JSON source directory |

Without `--truncate`, each run appends new snapshot rows (history).

## Schema

| Table | Source JSON |
|-------|-------------|
| `monitoring_device_config` | `devices.json` |
| `monitoring_device_snapshots` + `monitoring_device_entries` | `device-status.json` |
| `monitoring_sensor_snapshots` + `monitoring_sensor_entries` | `sensor-status.json` |
| `monitoring_service_snapshots` + `monitoring_service_entries` | `service-status.json` |

DDL: `schema/001_monitoring.sql` (applied on first `docker compose up`).

### Migration: monitoring events (002)

The SDL monitoring watcher writes to `monitoring_events`. On an **existing** volume (already ran `docker compose up` before this table existed), apply manually:

```powershell
# PowerShell (recommended on Windows)
.\scripts\apply-002-events.ps1
```

```bash
# Bash
./scripts/apply-002-events.sh
```

Or inline:

```powershell
Get-Content schema\002_monitoring_events.sql | docker compose exec -T postgres psql -U angie -d angie_monitoring_replica
```

Or from the host with `psql` and `DATABASE_URL`:

```bash
psql "$DATABASE_URL" -f schema/002_monitoring_events.sql
```

New volumes: add `002_monitoring_events.sql` to `docker-compose.yml` `docker-entrypoint-initdb.d` or run the command above once after first boot.

## Verify

```sql
SELECT relname, n_live_tup
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY relname;

SELECT e.name, e.online
FROM monitoring_device_entries e
JOIN monitoring_device_snapshots s ON s.id = e.snapshot_id
ORDER BY s.loaded_at DESC, e.ip
LIMIT 10;
```

## Reset database

```bash
docker compose down -v
docker compose up -d
python seed/load_monitoring.py --truncate
```
