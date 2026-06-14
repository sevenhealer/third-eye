-- Time-series schema for the event/metrics layer.
--
-- Works on BOTH a plain PostgreSQL/pgvector image (creates plain tables) and a
-- TimescaleDB image (additionally converts them to hypertables + retention).
-- The pgvector/pgvector:pg16 image does NOT ship the timescaledb extension, so
-- we must NOT hard-require it — a bare `CREATE EXTENSION timescaledb` aborts
-- the whole init script and leaves these tables missing. Degrade gracefully.

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'timescaledb unavailable — using plain tables (functionally fine for the app)';
END $$;

-- ── System Events ─────────────────────────────────────────────────────────────
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

CREATE INDEX IF NOT EXISTS idx_events_person_time
    ON system_events (person_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON system_events (event_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_zone_time
    ON system_events (zone_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity_time
    ON system_events (severity, event_time DESC)
    WHERE severity IN ('HIGH','CRITICAL');

-- ── Object Counts ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS object_counts (
    bucket_time     TIMESTAMPTZ NOT NULL,
    camera_id       VARCHAR(100) NOT NULL,
    zone_id         VARCHAR(100),
    object_class    VARCHAR(100) NOT NULL,
    count           INT NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket_time, camera_id, object_class)
);

-- ── Camera Health ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS camera_health (
    recorded_at     TIMESTAMPTZ NOT NULL,
    camera_id       VARCHAR(100) NOT NULL,
    status          VARCHAR(20) NOT NULL,    -- online, offline, degraded, tampered
    fps_actual      FLOAT,
    decode_latency_ms FLOAT,
    frame_count     BIGINT,
    PRIMARY KEY (recorded_at, camera_id)
);

-- ── Inference Latency ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inference_metrics (
    recorded_at     TIMESTAMPTZ NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    camera_id       VARCHAR(100),
    latency_ms      FLOAT NOT NULL,
    batch_size      INT DEFAULT 1,
    vram_used_mb    INT,
    PRIMARY KEY (recorded_at, model_name)
);

-- ── TimescaleDB optimizations (only if the extension actually loaded) ──────────
-- Hypertables + retention. Wrapped so init never aborts on plain PostgreSQL.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('system_events', 'event_time',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('object_counts', 'bucket_time',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('camera_health', 'recorded_at',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('inference_metrics', 'recorded_at',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM add_retention_policy('system_events',
            INTERVAL '90 days', if_not_exists => TRUE);
        PERFORM add_retention_policy('object_counts',
            INTERVAL '90 days', if_not_exists => TRUE);
        PERFORM add_retention_policy('camera_health',
            INTERVAL '30 days', if_not_exists => TRUE);
        PERFORM add_retention_policy('inference_metrics',
            INTERVAL '30 days', if_not_exists => TRUE);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'timescaledb optimization step skipped: %', SQLERRM;
END $$;

-- NOTE: TimescaleDB continuous aggregates (hourly_zone_occupancy,
-- daily_security_summary) cannot be created inside a transaction block, so they
-- are NOT part of this auto-loaded init script. On a real TimescaleDB
-- deployment, apply them separately (see docs). The app does not depend on them.
