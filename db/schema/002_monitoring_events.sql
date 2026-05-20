-- Alert/event log for monitoring watcher (apply after 001_monitoring.sql)

CREATE TABLE IF NOT EXISTS monitoring_events (
    id              BIGSERIAL PRIMARY KEY,
    fired_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    entity_type     TEXT NOT NULL,
    entity_key      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         JSONB,
    snapshot_id     INT,
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_monitoring_events_fired ON monitoring_events(fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_events_entity ON monitoring_events(entity_type, entity_key);
