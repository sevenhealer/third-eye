-- Third-Eye PostgreSQL Schema
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── RBAC Roles ────────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'thirdeye_readonly') THEN
        CREATE ROLE thirdeye_readonly;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'thirdeye_app') THEN
        CREATE ROLE thirdeye_app;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'thirdeye_admin') THEN
        CREATE ROLE thirdeye_admin;
    END IF;
END
$$;

-- ── Users (operators who log in) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username    VARCHAR(100) UNIQUE NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    hashed_pw   VARCHAR(255) NOT NULL,
    role        VARCHAR(50) NOT NULL DEFAULT 'readonly'
                CHECK (role IN ('readonly','operator','analyst','admin','security_officer')),
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    last_login  TIMESTAMPTZ
);

-- ── Enrolled Identities ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persons (
    person_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name    VARCHAR(255) NOT NULL,
    role            VARCHAR(100),
    notes           TEXT,
    enrolled_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    enrolled_by     UUID REFERENCES users(user_id),
    is_active       BOOLEAN DEFAULT TRUE,
    metadata        JSONB DEFAULT '{}',
    last_seen_at    TIMESTAMPTZ,
    last_seen_zone  VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_persons_active ON persons(is_active);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons USING gin(display_name gin_trgm_ops);

-- ── Face Gallery (pgvector HNSW) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS face_gallery (
    embedding_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id           UUID NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding           VECTOR(512) NOT NULL,
    embedding_enc       BYTEA,               -- AES-256 encrypted copy for export
    quality_score       FLOAT NOT NULL,
    source_camera_id    VARCHAR(100),
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_primary          BOOLEAN DEFAULT FALSE,
    model_version       VARCHAR(50) DEFAULT 'adaface-r100'
);
CREATE INDEX IF NOT EXISTS idx_face_gallery_person ON face_gallery(person_id);
CREATE INDEX IF NOT EXISTS idx_face_gallery_hnsw
    ON face_gallery USING hnsw(embedding vector_cosine_ops)
    WITH (m = 48, ef_construction = 200);

-- ── Enrollment Candidates (pending operator review) ───────────────────────────
CREATE TABLE IF NOT EXISTS enrollment_candidates (
    candidate_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    track_id        VARCHAR(100) NOT NULL,
    camera_id       VARCHAR(100) NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    crop_paths      JSONB DEFAULT '[]',      -- paths to stored face crops
    mean_embedding  VECTOR(512),
    quality_scores  JSONB DEFAULT '[]',
    status          VARCHAR(30) DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','expired')),
    reviewed_by     UUID REFERENCES users(user_id),
    reviewed_at     TIMESTAMPTZ,
    assigned_person_id UUID REFERENCES persons(person_id)
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON enrollment_candidates(status);

-- ── Cameras ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cameras (
    camera_id       VARCHAR(100) PRIMARY KEY,
    display_name    VARCHAR(255) NOT NULL,
    stream_url      TEXT NOT NULL,           -- RTSPS URL (credentials stored encrypted)
    zone_id         VARCHAR(100),
    location_desc   TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    resolution_w    INT DEFAULT 1920,
    resolution_h    INT DEFAULT 1080,
    fps_target      INT DEFAULT 30,
    metadata        JSONB DEFAULT '{}'
);

-- ── Zones ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS zones (
    zone_id         VARCHAR(100) PRIMARY KEY,
    display_name    VARCHAR(255) NOT NULL,
    zone_type       VARCHAR(50) DEFAULT 'general'
                    CHECK (zone_type IN ('general','restricted','entrance','exit','monitored')),
    description     TEXT,
    polygon_points  JSONB,                   -- [{x,y}, ...] for UI overlay
    metadata        JSONB DEFAULT '{}'
);

-- ── Zone Presence History ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS zone_presence (
    id              BIGSERIAL PRIMARY KEY,
    person_id       UUID REFERENCES persons(person_id),
    zone_id         VARCHAR(100) REFERENCES zones(zone_id),
    camera_id       VARCHAR(100) REFERENCES cameras(camera_id),
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    confidence      FLOAT,
    track_id        VARCHAR(100),
    is_unknown      BOOLEAN DEFAULT FALSE    -- true for unresolved persons
);
CREATE INDEX IF NOT EXISTS idx_presence_person_time ON zone_presence(person_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_presence_zone_time ON zone_presence(zone_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_presence_entry ON zone_presence(entry_time DESC);

-- ── Object Inventory ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tracked_objects (
    object_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name    VARCHAR(255),
    object_class    VARCHAR(100) NOT NULL,
    zone_id         VARCHAR(100) REFERENCES zones(zone_id),
    first_seen_at   TIMESTAMPTZ DEFAULT now(),
    last_seen_at    TIMESTAMPTZ,
    is_present      BOOLEAN DEFAULT TRUE,
    metadata        JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_objects_zone ON tracked_objects(zone_id, is_present);

-- ── Alerts ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type      VARCHAR(100) NOT NULL,
    severity        VARCHAR(20) NOT NULL
                    CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    camera_id       VARCHAR(100),
    zone_id         VARCHAR(100),
    person_id       UUID REFERENCES persons(person_id),
    object_id       UUID REFERENCES tracked_objects(object_id),
    description     TEXT,
    payload         JSONB DEFAULT '{}',
    triggered_at    TIMESTAMPTZ DEFAULT now(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID REFERENCES users(user_id),
    resolved_at     TIMESTAMPTZ,
    resolved_by     UUID REFERENCES users(user_id),
    status          VARCHAR(20) DEFAULT 'open'
                    CHECK (status IN ('open','acknowledged','resolved','suppressed'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_status_time ON alerts(status, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, triggered_at DESC);

-- ── NL Query History ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_history (
    query_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(user_id),
    query_text      TEXT NOT NULL,
    intent          VARCHAR(100),
    answer_text     TEXT,
    confidence      FLOAT,
    latency_ms      INT,
    data_sources    JSONB DEFAULT '[]',
    queried_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_queries_user_time ON query_history(user_id, queried_at DESC);

-- ── Model Registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(100) NOT NULL,
    model_type      VARCHAR(100) NOT NULL,
    version         VARCHAR(50) NOT NULL,
    stage           VARCHAR(20) DEFAULT 'staging'
                    CHECK (stage IN ('development','staging','production','archived')),
    weight_path     TEXT NOT NULL,
    sha256_hash     VARCHAR(64) NOT NULL,
    metrics         JSONB DEFAULT '{}',
    approved_by     UUID[],
    deployed_at     TIMESTAMPTZ,
    archived_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(model_name, version)
);

-- ── Append-Only Hash-Chained Audit Log ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    log_id          BIGSERIAL PRIMARY KEY,
    event_time      TIMESTAMPTZ DEFAULT now() NOT NULL,
    actor_id        UUID,
    actor_username  VARCHAR(100),
    action_type     VARCHAR(100) NOT NULL,
    resource_type   VARCHAR(100),
    resource_id     VARCHAR(255),
    details         JSONB DEFAULT '{}',
    source_ip       INET,
    prev_hash       VARCHAR(64) DEFAULT '0000000000000000000000000000000000000000000000000000000000000000',
    current_hash    VARCHAR(64) GENERATED ALWAYS AS (
        encode(
            sha256(
                (   COALESCE(log_id::text, '')
                 || COALESCE(event_time::text, '')
                 || COALESCE(actor_id::text, '')
                 || COALESCE(action_type, '')
                 || COALESCE(resource_type, '')
                 || COALESCE(resource_id, '')
                 || COALESCE(details::text, '')
                 || COALESCE(prev_hash, '')
                )::bytea
            ),
            'hex'
        )
    ) STORED
);

-- Prevent any modification of audit log rows
CREATE OR REPLACE RULE no_update_audit AS
    ON UPDATE TO audit_log DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_delete_audit AS
    ON DELETE TO audit_log DO INSTEAD NOTHING;

CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action_type, event_time DESC);

-- ── Default zones and seed data ───────────────────────────────────────────────
INSERT INTO zones (zone_id, display_name, zone_type, description) VALUES
    ('building_entrance', 'Building Entrance', 'entrance', 'Main building entry/exit'),
    ('room_a', 'Room A', 'general', 'General purpose room'),
    ('server_room', 'Server Room', 'restricted', 'Data center / server racks'),
    ('corridor_main', 'Main Corridor', 'monitored', 'Primary hallway')
ON CONFLICT (zone_id) DO NOTHING;

-- ── Seed admin user (change password immediately) ─────────────────────────────
-- Password: 'admin' (bcrypt hash — MUST change in production)
INSERT INTO users (username, email, hashed_pw, role) VALUES
    ('admin', 'admin@thirdeye.local',
     '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',
     'admin')
ON CONFLICT (username) DO NOTHING;
