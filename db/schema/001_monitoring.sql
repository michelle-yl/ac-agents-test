-- Lab monitoring replica schema (normalized from obsidian-vault/monitoring/*.json)

CREATE TABLE monitoring_device_config (
    id              SERIAL PRIMARY KEY,
    source          TEXT,
    query_url       TEXT,
    query_sql       TEXT,
    ping_timeout_seconds INT,
    ping_count      INT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE monitoring_device_snapshots (
    id                      SERIAL PRIMARY KEY,
    last_checked            TIMESTAMPTZ,
    last_successful_probe   TIMESTAMPTZ,
    last_probe_result       TEXT,
    probe_note              TEXT,
    method                  TEXT,
    alert_policy            JSONB,
    loaded_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE monitoring_device_entries (
    snapshot_id             INT NOT NULL REFERENCES monitoring_device_snapshots(id) ON DELETE CASCADE,
    ip                      TEXT NOT NULL,
    name                    TEXT,
    online                  BOOLEAN,
    ssh                     BOOLEAN,
    smb                     BOOLEAN,
    rdp                     BOOLEAN,
    ports                   TEXT[],
    last_change             TIMESTAMPTZ,
    last_verified           TIMESTAMPTZ,
    consecutive_down_count  INT,
    PRIMARY KEY (snapshot_id, ip)
);

CREATE INDEX idx_monitoring_device_entries_snapshot ON monitoring_device_entries(snapshot_id);
CREATE INDEX idx_monitoring_device_entries_ip ON monitoring_device_entries(ip);

CREATE TABLE monitoring_sensor_snapshots (
    id              SERIAL PRIMARY KEY,
    last_check      TIMESTAMPTZ,
    summary         TEXT,
    alert_policy    JSONB,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE monitoring_sensor_entries (
    snapshot_id             INT NOT NULL REFERENCES monitoring_sensor_snapshots(id) ON DELETE CASCADE,
    sensor_name             TEXT NOT NULL,
    online                  BOOLEAN,
    alerts                  JSONB,
    last_change             TIMESTAMPTZ,
    last_seen               TIMESTAMPTZ,
    reason                  TEXT,
    last_reading            JSONB,
    consecutive_down_count  INT,
    PRIMARY KEY (snapshot_id, sensor_name)
);

CREATE INDEX idx_monitoring_sensor_entries_snapshot ON monitoring_sensor_entries(snapshot_id);
CREATE INDEX idx_monitoring_sensor_entries_name ON monitoring_sensor_entries(sensor_name);

CREATE TABLE monitoring_service_snapshots (
    id              SERIAL PRIMARY KEY,
    last_check      TIMESTAMPTZ,
    check_status    TEXT,
    check_note      TEXT,
    alert_policy    JSONB,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE monitoring_service_entries (
    snapshot_id             INT NOT NULL REFERENCES monitoring_service_snapshots(id) ON DELETE CASCADE,
    service_name            TEXT NOT NULL,
    up                      BOOLEAN,
    host                    TEXT,
    ip                      TEXT,
    port                    INT,
    protocol                TEXT,
    note                    TEXT,
    consecutive_down_count  INT,
    PRIMARY KEY (snapshot_id, service_name)
);

CREATE INDEX idx_monitoring_service_entries_snapshot ON monitoring_service_entries(snapshot_id);
CREATE INDEX idx_monitoring_service_entries_name ON monitoring_service_entries(service_name);
