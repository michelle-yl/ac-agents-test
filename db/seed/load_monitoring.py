#!/usr/bin/env python3
"""Load obsidian-vault/monitoring JSON into the local PostgreSQL replica."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

DB_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_DIR = DB_DIR / "seed" / "data" / "monitoring"

META_KEYS_SENSOR = frozenset({"_lastCheck", "_summary", "_alertPolicy"})


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    user = os.environ.get("POSTGRES_USER", "angie")
    password = os.environ.get("POSTGRES_PASSWORD", "angie")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "angie_monitoring_replica")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def upsert_device_config(conn: psycopg.Connection, data: dict) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM monitoring_device_config")
        cur.execute(
            """
            INSERT INTO monitoring_device_config (
                source, query_url, query_sql, ping_timeout_seconds, ping_count
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                data.get("source"),
                data.get("query_url"),
                data.get("query_sql"),
                data.get("pingTimeoutSeconds"),
                data.get("pingCount"),
            ),
        )


def load_device_status(conn: psycopg.Connection, data: dict) -> int:
    devices = data.get("devices") or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO monitoring_device_snapshots (
                last_checked, last_successful_probe, last_probe_result,
                probe_note, method, alert_policy
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                parse_ts(data.get("lastChecked")),
                parse_ts(data.get("lastSuccessfulProbe")),
                data.get("lastProbeResult"),
                data.get("probeNote"),
                data.get("method"),
                Jsonb(data.get("alertPolicy")),
            ),
        )
        snapshot_id = cur.fetchone()[0]

        for ip, entry in devices.items():
            ports = entry.get("ports")
            if ports is not None and not isinstance(ports, list):
                ports = None
            cur.execute(
                """
                INSERT INTO monitoring_device_entries (
                    snapshot_id, ip, name, online, ssh, smb, rdp, ports,
                    last_change, last_verified, consecutive_down_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    snapshot_id,
                    ip,
                    entry.get("name"),
                    entry.get("online"),
                    entry.get("ssh"),
                    entry.get("smb"),
                    entry.get("rdp"),
                    ports,
                    parse_ts(entry.get("lastChange")),
                    parse_ts(entry.get("lastVerified")),
                    entry.get("consecutiveDownCount"),
                ),
            )
    return len(devices)


def load_sensor_status(conn: psycopg.Connection, data: dict) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO monitoring_sensor_snapshots (last_check, summary, alert_policy)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                parse_ts(data.get("_lastCheck")),
                data.get("_summary"),
                Jsonb(data.get("_alertPolicy")),
            ),
        )
        snapshot_id = cur.fetchone()[0]

        count = 0
        for name, entry in data.items():
            if name in META_KEYS_SENSOR or not isinstance(entry, dict):
                continue
            cur.execute(
                """
                INSERT INTO monitoring_sensor_entries (
                    snapshot_id, sensor_name, online, alerts, last_change,
                    last_seen, reason, last_reading, consecutive_down_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    snapshot_id,
                    name,
                    entry.get("online"),
                    Jsonb(entry.get("alerts")),
                    parse_ts(entry.get("lastChange")),
                    parse_ts(entry.get("lastSeen")),
                    entry.get("reason"),
                    Jsonb(entry.get("lastReading")),
                    entry.get("consecutiveDownCount"),
                ),
            )
            count += 1
    return count


def load_service_status(conn: psycopg.Connection, data: dict) -> int:
    services = data.get("services") or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO monitoring_service_snapshots (
                last_check, check_status, check_note, alert_policy
            ) VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                parse_ts(data.get("lastCheck")),
                data.get("checkStatus"),
                data.get("checkNote"),
                Jsonb(data.get("alertPolicy")),
            ),
        )
        snapshot_id = cur.fetchone()[0]

        for name, entry in services.items():
            cur.execute(
                """
                INSERT INTO monitoring_service_entries (
                    snapshot_id, service_name, up, host, ip, port,
                    protocol, note, consecutive_down_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    snapshot_id,
                    name,
                    entry.get("up"),
                    entry.get("host"),
                    entry.get("ip"),
                    entry.get("port"),
                    entry.get("protocol"),
                    entry.get("note"),
                    entry.get("consecutiveDownCount"),
                ),
            )
    return len(services)


def truncate_snapshots(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE monitoring_device_snapshots CASCADE")
        cur.execute("TRUNCATE monitoring_sensor_snapshots CASCADE")
        cur.execute("TRUNCATE monitoring_service_snapshots CASCADE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path(os.environ.get("MONITORING_JSON_DIR", DEFAULT_JSON_DIR)),
        help="Directory containing monitoring JSON files",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear existing snapshots before loading (keeps device_config history via replace)",
    )
    args = parser.parse_args()
    json_dir = args.json_dir.resolve()

    required = {
        "devices.json": upsert_device_config,
        "device-status.json": load_device_status,
        "sensor-status.json": load_sensor_status,
        "service-status.json": load_service_status,
    }

    missing = [name for name in required if not (json_dir / name).exists()]
    if missing:
        print(f"Missing files in {json_dir}: {', '.join(missing)}", file=sys.stderr)
        return 1

    url = database_url()
    print(f"Connecting to {url.split('@')[-1]}")

    with psycopg.connect(url) as conn:
        if args.truncate:
            truncate_snapshots(conn)
            print("Truncated existing snapshots")

        counts: dict[str, int] = {}

        devices = load_json(json_dir / "devices.json")
        upsert_device_config(conn, devices)
        counts["monitoring_device_config"] = 1

        device_status = load_json(json_dir / "device-status.json")
        counts["monitoring_device_entries"] = load_device_status(conn, device_status)

        sensor_status = load_json(json_dir / "sensor-status.json")
        counts["monitoring_sensor_entries"] = load_sensor_status(conn, sensor_status)

        service_status = load_json(json_dir / "service-status.json")
        counts["monitoring_service_entries"] = load_service_status(
            conn, service_status
        )

        conn.commit()

    for table, n in counts.items():
        print(f"  {table}: {n} row(s) loaded")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
