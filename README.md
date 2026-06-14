# Third-Eye 👁️

**Production-grade, local-first visual intelligence platform.**

Third-Eye ingests multi-camera video, identifies people and objects in real time, understands human actions and scenes, stores long-term memory in a knowledge graph, and answers natural-language questions about your physical environment — entirely on-premises with zero cloud dependency.

> "Who is currently in Room A?" → answered in under 500 ms, locally, with a full audit trail.

---

## Project Status

**Implemented (Sprints 1–6):** infrastructure + API + hash-chained audit log, the full
face pipeline (detection, anti-spoofing, recognition, enrollment), camera ingestion
(USB + RTSP via PyAV, up to 8 cameras), object detection, multi-object tracking,
cross-camera ReID, zone object events, model registry with SHA-256 signing, and the
anti-spoofing fine-tune pipeline.

**Live validation is in progress sprint-by-sprint** against real infrastructure:

- Sprint 1 (infra, API, audit log) — ✅ passed ([guide](live_testing/sprint1_live_tests.md))
- Sprint 2 (face pipeline + live recognition) — ✅ passed ([guide](live_testing/sprint2_live_tests.md))
- Sprint 3 (objects, events, alerts, tracking) — 🔄 testing now ([guide](live_testing/sprint3_live_tests.md))

All sprint live-test guides live in [`live_testing/`](live_testing/).

Development runs on macOS (CPU/CoreML on Apple Silicon, `DEVICE=cpu`, MLflow on
port 5001 due to AirPlay) and Linux + RTX 3090 (`DEVICE=cuda:0`). Production
targets an NVIDIA H100.

---

## Key Capabilities

| Capability | Status |
|---|---|
| Real-time multi-camera ingestion (up to 8 × 1080p30, PyAV RTSP) | ✅ Implemented |
| Face detection (SCRFD-10GF) | ✅ Implemented |
| Anti-spoofing ensemble — MiniFASNet-V2 + CDCN++ + temporal consistency | ✅ Implemented |
| Face recognition + identity enrollment (ArcFace R50 → AdaFace IR-101 planned) | ✅ Implemented |
| Object detection + zone events (YOLO26 + YOLO-World open-vocab; domain fine-tuning pipeline) | ✅ Implemented |
| Multi-object tracking + cross-camera ReID (ByteTrack/StrongSORT + OSNet → SOLIDER planned) | ✅ Implemented |
| Model registry (SHA-256) + anti-spoofing fine-tune pipeline | ✅ Implemented |
| Knowledge graph (Neo4j) + memory consolidation | 🏗️ Sprint 7 |
| Human action recognition (X3D-M + VideoMAE-B) | 🏗️ Sprint 8 |
| Event detection — rule engine + anomaly detection | 🏗️ Sprint 8 |
| Natural-language queries (Qwen3.5, local) | 🏗️ Sprints 9–10 |
| Scene understanding (CLIP + Qwen3.5) | 🏗️ Sprint 10 |
| Forensic search + video replay | 🏗️ Phase 3 |
| Continuous learning via operator feedback | 🏗️ Phase 3 |
| Temporal reasoning + behavioral baselines | 🏗️ Phase 4 |
| Animal detection + species classification | 🏗️ Phase 4 |

Model choices are tracked against the June 2026 state of the art in
[AGILE_PLAN.md §15](AGILE_PLAN.md) (Epic E11): AdaFace IR-101 recognition,
RF-DETR/D-FINE + YOLO-World detection, SOLIDER/CLIP-ReID, CLIP-guided
anti-spoofing with IR/depth hardware, Qwen3.5 replacing Mistral-7B + LLaVA-1.5,
and Triton + TensorRT serving. No model ships without a before/after run on the
deployment evaluation harness.

---

## Architecture

```
┌─────────────────── Multi-Camera Input ───────────────────────┐
│  RTSP/RTSPS  │  USB/V4L2  │  Recorded MP4 (forensic replay)  │
└──────────────┴────────────┴───────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Layer 1        │  GStreamer + NVDEC
                    │  Video Ingest   │  H.264/H.265 archive → NVMe
                    └────────┬────────┘
                             │  Kafka: camera.frames
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  Layer 2      │  │  Layer 5      │  │  Layer 8      │
  │  Face Detect  │  │  Object Det.  │  │  Action Recog │
  │  SCRFD-10GF   │  │  YOLO26       │  │  X3D-M +      │
  └───────┬───────┘  └───────┬───────┘  │  VideoMAE-B   │
          │                  │          └───────┬───────┘
  ┌───────▼───────┐  ┌───────▼───────┐          │
  │  Layer 3      │  │  Layer 6      │  ┌───────▼───────┐
  │  Anti-Spoof   │  │  Counting     │  │  Layer 9      │
  │  MiniFASNet+  │  │  Track-based  │  │  Animal Det.  │
  │  CDCN++       │  └───────┬───────┘  │  EfficientNet │
  └───────┬───────┘          │          └───────┬───────┘
          │          ┌───────▼───────┐          │
  ┌───────▼───────┐  │  Layer 7      │  ┌───────▼───────┐
  │  Layer 4      │  │  MOT + ReID   │  │  Layer 10     │
  │  Face Recog.  │  │  ByteTrack +  │  │  Scene Under. │
  │  AdaFace R100 │  │  StrongSORT + │  │  CLIP +       │
  │  pgvector     │  │  OSNet        │  │  Qwen3.5      │
  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
          └──────────────────┼──────────────────┘
                             │  Kafka: events.*
                    ┌────────▼────────┐
                    │  Layer 11       │  YAML rule engine
                    │  Event Detect.  │  + Z-score anomaly
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  Layer 12       │  TimescaleDB
                    │  Temporal Rsn.  │  + PrefixSpan
                    └────────┬────────┘
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐  ┌───────────────┐  ┌──────────────┐
   │  Redis      │  │  PostgreSQL   │  │  Neo4j       │
   │  Live state │  │  TimescaleDB  │  │  Knowledge   │
   │  < 1ms read │  │  Events 90d   │  │  Graph       │
   │  10-min TTL │  │  pgvector     │  │  Cypher      │
   └─────────────┘  │  HNSW index   │  └──────────────┘
                    └───────────────┘
                             │
                    ┌────────▼────────┐
                    │  Layer 14       │  Qwen3.5 (local —
                    │  NL Query       │  vLLM / Ollama)
                    │  Engine         │  Template SQL/Cypher
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐  ┌───────────────┐  ┌──────────────┐
   │  FastAPI    │  │  Alert Engine │  │  Forensic    │
   │  REST +     │  │  Webhook /    │  │  Search +    │
   │  WebSocket  │  │  Email / Push │  │  Replay      │
   └─────────────┘  └───────────────┘  └──────────────┘
```

**Every layer publishes to Apache Kafka. All state flows through three memory tiers. All operator actions are recorded in a tamper-evident hash-chained audit log.**

---

## Hardware Requirements

| Component | Dev/Test | Production |
|---|---|---|
| GPU | RTX 3090 (24 GB) — macOS CPU/CoreML also supported for development | NVIDIA H100 (80 GB) |
| RAM | 32 GB | 64 GB+ |
| Storage | 500 GB NVMe | 2 TB NVMe |
| OS | Linux (Ubuntu 22.04+) or macOS (dev only) | Ubuntu 22.04 LTS |
| CUDA | 12.1+ | 12.3+ |
| Docker | 24+ | 26+ |

**VRAM budget on RTX 3090 at peak load: ~11.5 GB continuous + 4 GB time-multiplexed = ~15.5 GB total. Well within 24 GB.** The dev profile is the production profile at smaller scale — same architecture, smaller VLM weights (Qwen3.5-9B quantized vs 27B) and fewer cameras.

---

## Quick Start

> **Developing right now?** The sprint-by-sprint live test guides in
> [`live_testing/`](live_testing/) are the fastest path:
> [infra + API](live_testing/sprint1_live_tests.md),
> [face pipeline](live_testing/sprint2_live_tests.md),
> [objects + events](live_testing/sprint3_live_tests.md).
> Key CLI tools: `scripts/download_models.py` (one-time, ~200 MB),
> `scripts/run_live.py --source 0 --show` (live face feed),
> `scripts/run_objects.py --source 0 --show` (live object feed),
> `scripts/enroll.py --name "..."` (gallery enrollment).

### 1. Prerequisites

```bash
# NVIDIA Container Toolkit (for GPU access in Docker)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Docker Compose v2
docker compose version   # should be 2.x
```

### 2. Clone and configure

```bash
git clone https://github.com/sevenhealer/third-eye.git
cd third-eye

# Create your .env from the template
make setup

# Edit .env — at minimum set:
#   POSTGRES_PASSWORD=<strong-password>
#   REDIS_PASSWORD=<strong-password>
#   NEO4J_PASSWORD=<strong-password>
#   S3_SECRET_KEY=<strong-password>   # MinIO object storage (clips/evidence)
#   JWT_SECRET_KEY=<64-char-random-string>
#   APP_SECRET_KEY=<32-char-random-string>
nano .env
```

### 3. Start all services

```bash
make up
```

This starts: PostgreSQL + pgvector, TimescaleDB, Redis, Neo4j, Kafka, Ollama, all pipeline services, FastAPI, Prometheus, and Grafana.

### 4. Pull LLM models (first time, ~8 GB download)

```bash
make models-pull
```

### 5. Verify everything is healthy

```bash
# API health
curl http://localhost:8000/health

# Kafka topics
make kafka-topics

# Audit log integrity
make audit-verify
```

### 6. Access the interfaces

| Service | URL | Default credentials |
|---|---|---|
| API docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / see `GRAFANA_PASSWORD` in `.env` |
| MLflow | http://localhost:5001 | — |
| Neo4j Browser | http://localhost:7474 | neo4j / see `NEO4J_PASSWORD` |
| Prometheus | http://localhost:9090 | — |

### 7. Log in and enroll an identity

```bash
# Get a JWT token (default admin password is 'admin' — change immediately)
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=admin" | jq -r .access_token)

# Enroll a person
THIRDEYE_TOKEN=$TOKEN make enroll NAME="Rahul Kumar" ROLE="engineer"
```

### 8. Ask a question

```bash
curl -s -X POST http://localhost:8000/api/v1/queries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is currently in Room A?"}' | jq .
```

---

## Example Queries

The NL query engine answers these in real time:

```
"Who is currently in Room A?"
"What is Rahul doing right now?"
"When did John last enter the building?"
"Show every appearance of Person X in the last 30 days."
"How many monitors are currently present?"
"Did anyone access Server Rack 1 today?"
"Was there any unusual activity between 2 PM and 4 PM?"
"Did the cat enter the room today?"
"Show all events involving Desk 3 this week."
"What changed since yesterday?"
```

---

## Project Structure

```
third-eye/
├── configs/                    # YAML config for cameras, models, alert rules
│   ├── cameras/                # Camera registry + RTSP profiles
│   ├── models/                 # Per-model inference configs
│   ├── pipelines/              # Pipeline orchestration config
│   └── alerting/               # Declarative alert rules
│
├── src/
│   ├── core/                   # Config, logging, exceptions, GPU manager, audit log
│   ├── ingestion/              # Layer 1: GStreamer/FFmpeg video ingest
│   ├── face_detection/         # Layer 2: SCRFD face detection
│   ├── antispoofing/           # Layer 3: MiniFASNet-V2 + CDCN++ ensemble
│   ├── face_recognition/       # Layer 4: AdaFace R100 + pgvector gallery
│   ├── object_detection/       # Layer 5: YOLO26 (+ YOLO-World open-vocab)
│   ├── object_counting/        # Layer 6: Track-based + DM-Count
│   ├── tracking/               # Layer 7: ByteTrack + StrongSORT + OSNet ReID
│   ├── action_recognition/     # Layer 8: X3D-M + VideoMAE-B
│   ├── animal_detection/       # Layer 9: YOLO26 + EfficientNet-B3
│   ├── scene_understanding/    # Layer 10: CLIP + Qwen3.5
│   ├── event_detection/        # Layer 11: Rule engine + anomaly detector
│   ├── temporal_reasoning/     # Layer 12: Timelines + pattern miner
│   ├── memory/                 # Layer 13: Redis + PostgreSQL + Neo4j + pgvector
│   ├── nlq/                    # Layer 14: NL query engine (Qwen3.5)
│   ├── alerts/                 # Alert manager + notification routing
│   ├── forensics/              # Forensic search + video replay
│   ├── api/                    # FastAPI REST API + WebSocket
│   └── pipeline/               # Orchestrator + Kafka client + GPU scheduler
│
├── infrastructure/             # SQL schemas, Kafka topics, Redis/Neo4j configs
├── models/                     # Model weights (tracked with DVC, not git)
├── data/                       # Training data (tracked with DVC, not git)
├── training/                   # Fine-tuning scripts per model
├── mlops/                      # DVC, MLflow, retraining pipelines
├── tests/                      # Unit, integration, security, performance tests
├── scripts/                    # CLI tools: enroll, audit verify, model download
├── docker-compose.yml          # Full production stack
├── Makefile                    # Developer commands
└── AGILE_PLAN.md               # Full agile project plan (80 stories, 4 phases)
```

---

## Security Architecture

Third-Eye is designed for adversarial environments:

| Threat | Mitigation |
|---|---|
| Printed photo attack | CDCN++ texture analysis + MiniFASNet depth cue |
| Phone/screen replay | Moire detection + temporal frame rate mismatch |
| Deepfake stream | GAN artifact detection (CDCN++) + temporal coherence check |
| Adversarial examples | Input randomization at inference (breaks gradient attacks) |
| Prompt injection (LLM) | Structured JSON context only — LLM never receives raw text |
| Data poisoning (enrollment) | Quality threshold + dual operator authorization |
| Model weight tampering | SHA-256 checksums verified at startup + hourly |
| Database compromise | AES-256 at rest, row-level security, no direct external DB access |
| Log tampering | Append-only hash-chained audit log — tampering is detectable |
| Insider threat | Dual authorization for enrollment and model promotion |
| Camera tampering | Perceptual hash delta detection → `CAMERA_TAMPER_SUSPECTED` event |

### Anti-spoofing ensemble (hard-AND gate)

```
Face crop → MiniFASNet-V2 (3ms) → CDCN++ (11ms) → Temporal check (8ms)
             All three must pass. Any failure = reject.
             Confidence < 0.85 = default-deny.
```

### Audit log design

Every enrollment, query, alert, model promotion, and login is recorded in a PostgreSQL table with a SHA-256 hash chain. Rows cannot be modified or deleted — even by the database administrator.

```sql
-- Tampering is cryptographically detectable
current_hash = sha256(log_id || event_time || actor_id || details || prev_hash)
```

---

## API Overview

```
POST /api/v1/auth/login          Get JWT access + refresh token
GET  /api/v1/cameras             List cameras with live health status
GET  /api/v1/identities          List enrolled persons
POST /api/v1/identities          Enroll a new identity [admin]
GET  /api/v1/identities/{id}/timeline  Historical timeline [analyst]
GET  /api/v1/events              Query event log (filterable, paginated)
POST /api/v1/queries             Submit natural-language query
GET  /api/v1/alerts              List active/recent alerts
PUT  /api/v1/alerts/{id}/acknowledge  Acknowledge alert [operator]
GET  /api/v1/admin/audit-log     View audit log [security_officer]
GET  /api/v1/admin/audit-log/verify  Verify hash chain integrity
GET  /api/v1/admin/system-health     GPU + service health
```

**RBAC roles:** `readonly` → `operator` → `analyst` → `admin` → `security_officer`

---

## VRAM Budget (RTX 3090 — 24 GB)

| Model | VRAM | Load |
|---|---|---|
| SCRFD-10GF face detection | 500 MB | Continuous |
| MiniFASNet-V2 + CDCN++ anti-spoofing | 550 MB | Continuous |
| AdaFace R100 face recognition | 800 MB | Continuous |
| YOLO26 object detection | 900 MB | Continuous |
| OSNet-x1.0 ReID | 200 MB | Continuous |
| X3D-M action (coarse) | 400 MB | Continuous |
| CLIP ViT-L/14 scene | 900 MB | Continuous |
| VideoMAE-B action (detail) | 1.2 GB | On-demand |
| Qwen3.5-9B-4bit (NLQ + scene captioning, one model) | 4.0 GB | Time-multiplexed (shared slot) |
| Frame buffers + CUDA workspace | 3.0 GB | Continuous |
| **Peak total** | **~15.5 GB** | |
| **Headroom** | **~8.5 GB** | |

---

## Fine-Tuning Priorities

| Model | Priority | Reason |
|---|---|---|
| Anti-spoofing (MiniFASNet + CDCN++) | **CRITICAL** | Pretrained generalizes poorly to site-specific lighting and attack types |
| OSNet ReID | **HIGH** | Cross-camera accuracy requires deployment-camera adaptation |
| Action Recognition (VideoMAE-B) | **HIGH** | Tailgating / loitering / rack access not in Kinetics-400 |
| Object Detection (YOLO26) | **HIGH** | Domain items (almirah, equipment) not in COCO — open-vocab now, fine-tune for production |
| Face Detection (SCRFD) | MEDIUM | Only needed with IR cameras or unusual lighting |
| Face Recognition (AdaFace) | LOW | MS1MV3 pretrained sufficient up to ~200 gallery entries |

---

## Development Roadmap

| Phase | Weeks | Goal |
|---|---|---|
| **Phase 1 — MVP** | 1–8 | 2-camera live detection, face recognition, current-state NL queries |
| **Phase 2 — Production** | 9–20 | All 14 layers, 8-camera support, full audit trail, fine-tuned models |
| **Phase 3 — Enterprise** | 21–32 | MLOps pipeline, forensic search, adversarial hardening, TensorRT |
| **Phase 4 — Advanced** | 33–48 | Behavioral baselines, predictive analytics, multi-node prep |

Epic E11 (model & architecture upgrades, June 2026) runs across Sprints 7–16:
frames-off-Kafka pipeline restructure and evaluation harness first, then measured
model swaps, multimodal liveness hardware, and Triton + TensorRT serving for the
H100 production profile.

See [AGILE_PLAN.md](AGILE_PLAN.md) for the complete backlog (89 user stories), sprint plan, acceptance criteria, risk register, and the §15 upgrade roadmap.

---

## MLOps

```
Operator feedback → data/feedback/
      ↓ (100+ corrections trigger review)
DVC pull latest dataset + new labels
      ↓
Training run → MLflow experiment logged
      ↓
Automated eval on held-out test set
      ↓
Metrics pass? → Promote to Staging
      ↓
Admin + Security Officer dual approval
      ↓
Blue-green deploy (24h shadow mode) → Production
```

Model weights are SHA-256 signed at promotion and verified at every startup.

---

## Running Tests

```bash
# All tests
make test

# Unit tests only
docker compose exec api pytest tests/unit/ -v

# Security tests (prompt injection, anti-spoofing)
docker compose exec api pytest tests/security/ -v

# Audit log verification
make audit-verify
```

---

## Configuration

All configuration is in `configs/`. Key files:

| File | Purpose |
|---|---|
| `configs/cameras/camera_registry.yaml` | Camera definitions, RTSP URLs, zone assignments |
| `configs/alerting/alert_rules.yaml` | Declarative event rules (YAML, no code changes needed) |
| `configs/models/antispoofing.yaml` | Anti-spoofing ensemble thresholds |
| `configs/models/tracker.yaml` | ByteTrack / StrongSORT / OSNet parameters |
| `configs/pipelines/pipeline_config.yaml` | FPS targets, batch sizes, memory flush intervals |

---

## Contributing

This is a single-developer research and production project. The architecture is intentionally modular — each of the 14 layers is a self-contained Python package.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/layer-X-improvement`
3. Write tests first (see `tests/unit/` for patterns)
4. Ensure `make lint` and `make test` pass
5. Open a PR with a clear description of the change

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) — SCRFD, ArcFace
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26, YOLO-World
- [Qwen](https://github.com/QwenLM) — Qwen3.5 multimodal LLM
- [Ollama](https://github.com/ollama/ollama) — Local LLM serving
- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity in PostgreSQL
- [TimescaleDB](https://github.com/timescale/timescaledb) — Time-series on PostgreSQL
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
