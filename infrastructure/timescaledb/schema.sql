-- TimescaleDB schema — runs in the same PostgreSQL instance
-- Requires TimescaleDB extension (included in pgvector/pgvector:pg16 base)

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── System Events (hypertable) ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_events (
    event_id        UUID DEFAULT gen_random_uuid(),
    event_time      TIMESTAMPTZ NOT NULL,
    event_type      VARCHAR(100) NOT NULL,
    camera_id       VARCHAR(100),
    zone_id         VARCHAR(100),
    person_id       UUID,
    object_id       UUID,
    track_id        VARCHAR(100),
    severity        VARCHAR(20) DEFAULT 'INFO'
                    CHECK (severity IN ('INFO','LOW','MEDIUM','HIGH','CRITICAL')),
    confidence      FLOAT,
    payload         JSONB DEFAULT '{}',
    PRIMARY KEY (event_id, event_time)
);

SELECT create_hypertable('system_events', 'event_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_events_person_time
    ON system_events (person_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON system_events (event_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_zone_time
    ON system_events (zone_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity_time
    ON system_events (severity, event_time DESC)
    WHERE severity IN ('HIGH','CRITICAL');

-- ── Object Counts Time-Series ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS object_counts (
    bucket_time     TIMESTAMPTZ NOT NULL,
    camera_id       VARCHAR(100) NOT NULL,
    zone_id         VARCHAR(100),
    object_class    VARCHAR(100) NOT NULL,
    count           INT NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket_time, camera_id, object_class)
);

SELECT create_hypertable('object_counts', 'bucket_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ── Camera Health Time-Series ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS camera_health (
    recorded_at     TIMESTAMPTZ NOT NULL,
    camera_id       VARCHAR(100) NOT NULL,
    status          VARCHAR(20) NOT NULL,    -- online, offline, degraded, tampered
    fps_actual      FLOAT,
    decode_latency_ms FLOAT,
    frame_count     BIGINT,
    PRIMARY KEY (recorded_at, camera_id)
);

SELECT create_hypertable('camera_health', 'recorded_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ── Inference Latency Time-Series ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inference_metrics (
    recorded_at     TIMESTAMPTZ NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    camera_id       VARCHAR(100),
    latency_ms      FLOAT NOT NULL,
    batch_size      INT DEFAULT 1,
    vram_used_mb    INT,
    PRIMARY KEY (recorded_at, model_name)
);

SELECT create_hypertable('inference_metrics', 'recorded_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ── Continuous Aggregates ─────────────────────────────────────────────────────

-- Hourly zone occupancy (for temporal reasoning and NL queries)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_zone_occupancy
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time)   AS bucket,
    zone_id,
    COUNT(DISTINCT person_id)           AS unique_persons,
    COUNT(*)                            AS total_events
FROM system_events
WHERE event_type IN ('PERSON_ENTERED','PERSON_EXITED')
  AND zone_id IS NOT NULL
GROUP BY bucket, zone_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('hourly_zone_occupancy',
    start_offset => INTERVAL '3 hours',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily security event summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_security_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time)    AS bucket,
    event_type,
    severity,
    COUNT(*)                            AS event_count,
    COUNT(DISTINCT person_id)           AS unique_persons,
    COUNT(DISTINCT camera_id)           AS cameras_involved
FROM system_events
WHERE severity IN ('HIGH','CRITICAL')
GROUP BY bucket, event_type, severity
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_security_summary',
    start_offset => INTERVAL '2 days',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ── Retention Policies ────────────────────────────────────────────────────────
SELECT add_retention_policy('system_events',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('object_counts',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('camera_health',
    INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('inference_metrics',
    INTERVAL '30 days', if_not_exists => TRUE);
