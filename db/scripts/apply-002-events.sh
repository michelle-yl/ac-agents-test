#!/usr/bin/env bash
# Apply monitoring_events schema (bash).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose exec -T postgres psql -U angie -d angie_monitoring_replica \
  -f - < schema/002_monitoring_events.sql
echo "Applied 002_monitoring_events.sql"
