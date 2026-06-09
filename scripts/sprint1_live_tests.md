# Sprint 1 — Live Test Guide

Run all commands from the repo root: `/Users/iamrohanchatterjee/Documents/Code/third-eye`

Passwords used in these tests match `.env`:
- Postgres: `testpass123`
- Redis: `redispass123`
- Neo4j: `neo4jpass123`
- Admin login: `admin / admin`

---

## STEP 1 — Check .env exists and is configured

```bash
# Should print a file, not "No such file"
ls .env

# Should NOT show "change-me" for any of these
grep -E "POSTGRES_PASSWORD|REDIS_PASSWORD|NEO4J_PASSWORD|JWT_SECRET_KEY|APP_SECRET_KEY" .env
```

**Expected:** All 5 vars are set to real values (not `change-me-*`).

---

## STEP 2 — Start infrastructure services

```bash
docker-compose up postgres redis kafka kafka-init neo4j prometheus grafana mlflow -d
```

Wait 30 seconds, then check all containers:

```bash
docker-compose ps
```

**Expected output (all services):**

| Container | Status |
|---|---|
| third-eye-postgres-1 | Up (healthy) |
| third-eye-redis-1 | Up (healthy) |
| third-eye-kafka-1 | Up (healthy) |
| third-eye-neo4j-1 | Up (healthy) |
| third-eye-kafka-init-1 | Exited (0) — one-shot job, this is correct |
| third-eye-prometheus-1 | Up |
| third-eye-grafana-1 | Up |
| third-eye-mlflow-1 | Up |

---

## STEP 3a — Verify PostgreSQL

```bash
# List all tables — should show 12 tables
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "\dt"

# Check seed admin user exists
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye \
  -c "SELECT username, role, is_active FROM users;"

# Test audit_log hash trigger — should return a 64-char hex hash
docker exec -i third-eye-postgres-1 psql -U thirdeye -d thirdeye \
  -c "INSERT INTO audit_log (action_type, actor_username) VALUES ('SPRINT1_TEST', 'test_runner') RETURNING current_hash;"
```

**Expected:**
- 12 tables including `audit_log`
- `admin | admin | t`
- A 64-character hex string like `1436f8f9f968337932a8...`

---

## STEP 3b — Verify Redis

```bash
docker exec third-eye-redis-1 redis-cli -a redispass123 ping
```

**Expected:** `PONG`

---

## STEP 3c — Verify Kafka topics

```bash
docker exec third-eye-kafka-1 /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list
```

**Expected:** 14 topics listed:
```
alerts.outbound
antispoofing.results
camera.frames
camera.health
counts.snapshot
detections.faces
detections.objects
enrollment.candidates
events.actions
events.detected
feedback.corrections
recognition.identity
scene.context
tracking.states
```

---

## STEP 3d — Verify web UIs (open in browser)

| Service | URL | Login |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | no login |
| Neo4j Browser | http://localhost:7474 | neo4j / neo4jpass123 |
| MLflow | http://localhost:5001 | no login |

**Expected:** All 4 pages load without errors.

> Note: Port 5000 is blocked on macOS by AirPlay. MLflow runs on 5001.

---

## STEP 4 — Start the API

```bash
.venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Expected log lines (in order):**
```
third_eye_starting
model_manifest_absent_skipping_verification
third_eye_ready
Uvicorn running on http://127.0.0.1:8000
```

In a second terminal, check health:

```bash
curl http://127.0.0.1:8000/health
```

**Expected:** `{"status":"ok","version":"0.1.0"}`

---

## STEP 5 — Test JWT login

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -F "username=admin" \
  -F "password=admin" | python3 -m json.tool
```

**Expected:**
```json
{
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer"
}
```

Save the token for the next steps:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -F "username=admin" -F "password=admin" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo $TOKEN
```

---

## STEP 5b — Test /me endpoint

```bash
curl -s http://127.0.0.1:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected:**
```json
{
    "user_id": "...",
    "username": "admin",
    "role": "admin"
}
```

---

## STEP 6 — Test RBAC (unauthorized access)

```bash
# No token — should return 401
curl -sL -o /dev/null -w "HTTP %{http_code}\n" \
  http://127.0.0.1:8000/api/v1/identities/

# Bad token — should return 401
curl -sL -o /dev/null -w "HTTP %{http_code}\n" \
  http://127.0.0.1:8000/api/v1/identities/ \
  -H "Authorization: Bearer this-is-a-fake-token"
```

**Expected:** Both return `HTTP 401`

---

## STEP 7 — Verify audit log captured login events

```bash
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye \
  -c "SELECT action_type, actor_username, current_hash IS NOT NULL AS has_hash, event_time FROM audit_log ORDER BY log_id DESC LIMIT 5;"
```

**Expected:** Rows with `LOGIN_SUCCESS`, `LOGIN_FAILED`, `has_hash = t`

---

## Stopping everything

```bash
# Stop API: press Ctrl+C in the terminal where uvicorn is running

# Stop all Docker containers (keeps data volumes)
docker-compose down

# To also wipe all data and start fully fresh next time:
# docker-compose down -v
```

---

## Sprint 1 Pass Criteria

- [ ] All 8 Docker services up (7 running + kafka-init exited 0)
- [ ] 12 Postgres tables present including `audit_log`
- [ ] Seed admin user `admin / admin` exists
- [ ] Audit log hash trigger writes 64-char hashes
- [ ] Redis responds PONG
- [ ] All 14 Kafka topics exist
- [ ] All 4 web UIs load in browser
- [ ] API starts and `/health` returns ok
- [ ] Login returns JWT tokens
- [ ] `/me` returns correct user and role
- [ ] Unauthenticated requests return 401
- [ ] Audit log records login events with hashes
