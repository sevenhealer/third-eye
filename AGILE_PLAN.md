# Third-Eye — Agile Project Plan
**Project:** Third-Eye Visual Intelligence Platform
**Date:** 2026-06-09
**Owner:** Rohan Chatterjee
**Methodology:** Scrum (2-week sprints)
**Hardware baseline:** RTX 3090 24 GB (dev/test) → NVIDIA H100 80 GB (production) · Linux · NVMe · Python 3.11+

---

## 1. Product Vision

> "Third-Eye provides security teams with a local-first visual intelligence platform that answers natural-language questions about their physical environment in real time, maintains an auditable long-term memory of all events, and continuously improves through operator feedback — with zero cloud dependency and adversarial resilience built in from day one."

**Value proposition:**
- No cloud: all inference, storage, and reasoning on-premises
- Natural-language interface: operators ask questions in plain English
- Adversarially hardened: built to resist spoofing, tampering, poisoning
- Continuously improving: operator feedback drives retraining

---

## 2. System Requirements

| ID | Requirement |
|----|-------------|
| SR-01 | System ingests video from ≥ 8 IP/USB cameras simultaneously |
| SR-02 | All ML inference runs locally on GPU — RTX 3090 (dev/test), H100 (production); no cloud API calls |
| SR-03 | System answers natural-language queries about current and historical state |
| SR-04 | All biometric data encrypted at rest (AES-256) |
| SR-05 | Audit log is append-only, hash-chained, and tamper-evident |
| SR-06 | System supports model retraining without downtime (blue-green deployment) |
| SR-07 | Archive stores raw video segments; forensic replay available for ≥ 30 days |
| SR-08 | System detects and alerts on security events within 5 seconds of occurrence |
| SR-09 | Identity enrollment requires dual authorization |
| SR-10 | System is deployable via `docker compose up` on a single host |

---

## 3. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | Detect human faces in every camera frame in real time |
| FR-02 | Reject spoofed faces (print, screen, deepfake, 3D mask) before recognition |
| FR-03 | Identify enrolled persons by face with ≤ 20ms latency |
| FR-04 | Trigger enrollment workflow for unrecognized persons |
| FR-05 | Detect and classify objects (COCO + custom domain classes) |
| FR-06 | Count objects per class per zone in real time |
| FR-07 | Assign persistent track IDs across frames and cameras |
| FR-08 | Classify human actions (walking, typing, tailgating, loitering, etc.) |
| FR-09 | Detect and track animals; classify by species |
| FR-10 | Generate natural-language scene descriptions per camera |
| FR-11 | Detect security events via rule engine and anomaly detection |
| FR-12 | Maintain per-identity temporal timelines and behavioral baselines |
| FR-13 | Store long-term memory in a queryable knowledge graph |
| FR-14 | Accept natural-language queries and return accurate, sourced answers |
| FR-15 | Deliver real-time alerts via webhook, email, or push notification |
| FR-16 | Support forensic search: face, object, time range, across archive |
| FR-17 | Provide WebSocket live feed and real-time event stream |
| FR-18 | Expose REST API with RBAC (readonly / operator / analyst / admin / security_officer) |
| FR-19 | Log every operator action, query, enrollment, and model change to audit trail |
| FR-20 | Retrain models from operator feedback with dual-approval promotion |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-01 | Performance | 8× 1080p30 cameras at ≥ 10 FPS through full pipeline |
| NFR-02 | Performance | Face-to-identity latency ≤ 20ms per detection |
| NFR-03 | Performance | NL query response ≤ 4 seconds (conversational), ≤ 15s (forensic) |
| NFR-04 | Performance | VRAM usage ≤ 22 GB at peak load (24 GB total) |
| NFR-05 | Reliability | Camera reconnect within 60 seconds of disconnect |
| NFR-06 | Reliability | System uptime ≥ 99.5% (excluding planned maintenance) |
| NFR-07 | Security | Anti-spoofing ACER ≤ 2% after domain fine-tuning |
| NFR-08 | Security | Face recognition TAR@FAR=1e-4 ≥ 92% on deployment gallery |
| NFR-09 | Security | All biometric embeddings AES-256 encrypted at rest |
| NFR-10 | Security | API authentication via JWT (15-min expiry); RBAC on all endpoints |
| NFR-11 | Auditability | Every enrollment, query, and model change recorded in tamper-evident audit log |
| NFR-12 | Maintainability | Each pipeline layer deployable and restartable independently |
| NFR-13 | Scalability | Architecture supports adding GPUs and nodes without core refactoring |
| NFR-14 | Compliance | PII (biometric data) stored with 90-day rolling retention by default; configurable |
| NFR-15 | Observability | Prometheus metrics + Grafana dashboards for all 14 layers |

---

## 5. Risk Register

| ID | Risk | Probability | Impact | Mitigation | Owner |
|----|------|-------------|--------|------------|-------|
| R-01 | VRAM exhaustion under peak load | MEDIUM | HIGH | GPU manager with strict VRAM budget; time-multiplexing the VLM with vision models; quantized VLM on 3090 | ML Engineer |
| R-02 | Anti-spoofing bypass by novel attack | MEDIUM | CRITICAL | Ensemble hard-AND; default-deny on uncertainty; quarterly red-team | Security Officer |
| R-03 | Face recognition false accept | LOW | CRITICAL | Strict 0.45 cosine threshold; dual-camera confirmation for high-security zones | ML Engineer |
| R-04 | Kafka consumer lag under burst | MEDIUM | HIGH | Per-topic consumer groups; backpressure signals; non-critical topic drop policy | DevOps |
| R-05 | NVMe saturation from archive writes | LOW | HIGH | Configurable retention per camera; tiered archive (NVMe → HDD) | DevOps |
| R-06 | LLM hallucination in NL answers | HIGH | MEDIUM | Structured JSON context only; schema validation; fact cross-validation | ML Engineer |
| R-07 | Model drift undetected | MEDIUM | HIGH | Hourly KL-divergence monitoring; monthly benchmark validation | MLOps |
| R-08 | Insider threat via unauthorized enrollment | LOW | CRITICAL | Dual authorization; daily enrollment audit report; anomaly detection on enrollment patterns | Security Officer |
| R-09 | Database corruption / data loss | LOW | CRITICAL | WAL replication; daily pg_dump to encrypted NVMe; TimescaleDB continuous backup | DevOps |
| R-10 | Prompt injection via LLM queries | MEDIUM | HIGH | Template-based queries only; LLM receives structured JSON not raw text; process isolation | Security Officer |
| R-11 | Cross-camera ReID failure | MEDIUM | MEDIUM | ReID fine-tuning on deployment cameras (OSNet → SOLIDER/CLIP-ReID per §15); spatial-temporal constraints | ML Engineer |
| R-13 | Mixed embedding spaces in gallery after model upgrade | MEDIUM | CRITICAL | Enforce `model_version` in gallery search; full re-enrollment on recognition model change (E11-S03) | ML Engineer |
| R-12 | Key person unavailable mid-project | LOW | HIGH | Document all architecture decisions; modular design allows parallel development | PM |

---

## 6. Epics

| Epic ID | Name | Layers |
|---------|------|--------|
| E-01 | Video Ingestion & Infrastructure | Layer 1, Kafka, Docker |
| E-02 | Face Intelligence | Layers 2, 3, 4 |
| E-03 | Object & Motion Intelligence | Layers 5, 6, 7 |
| E-04 | Behavioral Intelligence | Layers 8, 9, 10 |
| E-05 | Event & Alert Engine | Layer 11 |
| E-06 | Memory & Knowledge Graph | Layers 12, 13 |
| E-07 | Natural Language Interface | Layer 14, Forensics |
| E-08 | MLOps & Continuous Learning | DVC, MLflow, Retraining |
| E-09 | Security & Compliance | Audit log, Encryption, RBAC, Adversarial testing |
| E-10 | Observability & Operations | Prometheus, Grafana, Runbooks |
| E-11 | Model & Architecture Upgrades (2026-06) | All inference layers, serving, evaluation — see §15 |

---

## 7. Product Backlog

### Epic E-01: Video Ingestion & Infrastructure

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E01-S01 | As a DevOps engineer, I can run `docker compose up` and have all infrastructure services (PostgreSQL, Redis, Kafka, pgvector, TimescaleDB, Neo4j) healthy | 5 | P0 | 1 |
| E01-S02 | As a pipeline engineer, I can ingest a live RTSP stream from 1 camera using GStreamer + NVDEC with < 5ms decode latency | 5 | P0 | 1 |
| E01-S03 | As a pipeline engineer, I can run 8 simultaneous camera streams without exceeding CPU/GPU decode budget | 8 | P1 | 2 |
| E01-S04 | As an operator, I am alerted within 5 seconds when a camera disconnects, and the system reconnects automatically | 5 | P0 | 1 |
| E01-S05 | As a security officer, I can verify that archived H.264 segments are HMAC-signed and have not been tampered with | 5 | P1 | 2 |
| E01-S06 | As an operator, I can replay archived footage from any camera for any time range in the last 30 days | 8 | P1 | 3 |
| E01-S07 | As a developer, I have structured JSON logging and a Grafana dashboard showing camera health for all active streams | 3 | P1 | 1 |

### Epic E-02: Face Intelligence

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E02-S01 | As a pipeline engineer, I can detect all faces in a camera frame using SCRFD-10GF with quality scores in < 2ms per image (TensorRT FP16) | 5 | P0 | 1 |
| E02-S02 | As a security officer, a printed photo held in front of the camera is rejected by anti-spoofing before reaching recognition | 8 | P0 | 1 |
| E02-S03 | As a security officer, a phone-screen replay attack is rejected by anti-spoofing | 8 | P0 | 2 |
| E02-S04 | As a security officer, a deepfake video stream is rejected by the CDCN++ component of the anti-spoofing ensemble | 8 | P1 | 2 |
| E02-S05 | As an operator, an enrolled person is correctly identified in < 20ms with cosine similarity > 0.45 | 5 | P0 | 1 |
| E02-S06 | As an operator, an unrecognized person triggers an enrollment candidate in the operator UI after 30 seconds of observation | 5 | P0 | 1 |
| E02-S07 | As an admin, I can approve an enrollment candidate, assign a name, and have the identity recognized in all future frames | 5 | P0 | 1 |
| E02-S08 | As an admin, I can delete an enrolled identity with soft-delete and 30-day recovery window | 3 | P1 | 2 |
| E02-S09 | As a security officer, all enrollment events are logged to the audit trail with operator ID and timestamp | 3 | P0 | 1 |
| E02-S10 | As an ML engineer, I can run the anti-spoofing fine-tuning pipeline on domain-collected attack data and promote the result via dual approval | 8 | P1 | 2 |

### Epic E-03: Object & Motion Intelligence

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E03-S01 | As a pipeline engineer, YOLOv9-C detects standard COCO objects in real time at ≥ 10 FPS per camera | 5 | P0 | 1 |
| E03-S02 | As an admin, I can define custom object classes (server rack, monitor) and fine-tune YOLOv9-C to detect them | 8 | P1 | 2 |
| E03-S03 | As an operator, I can see real-time counts of each object class per zone on the dashboard | 3 | P1 | 1 |
| E03-S04 | As a pipeline engineer, ByteTrack assigns persistent track IDs across frames with IDF1 > 75% on single-camera test video | 5 | P0 | 1 |
| E03-S05 | As a pipeline engineer, OSNet ReID correctly re-identifies persons across camera pairs with mAP > 85% after fine-tuning | 8 | P1 | 2 |
| E03-S06 | As an operator, a person's track is maintained across all 8 cameras with a consistent UUID-based identity | 8 | P1 | 2 |
| E03-S07 | As an operator, I am alerted when a tracked object is removed from a defined zone | 5 | P1 | 2 |

### Epic E-04: Behavioral Intelligence

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E04-S01 | As a pipeline engineer, X3D-M classifies human actions at 2 FPS per track with < 5ms batch latency | 5 | P1 | 2 |
| E04-S02 | As a security officer, VideoMAE-B detects tailgating with > 85% accuracy after fine-tuning on domain clips | 8 | P1 | 2 |
| E04-S03 | As a security officer, loitering (person in zone > N seconds) triggers an alert via the action recognition pipeline | 5 | P1 | 2 |
| E04-S04 | As an operator, the system detects a cat entering a restricted zone and sends an animal intrusion alert | 3 | P2 | 2 |
| E04-S05 | As an analyst, CLIP ViT-L/14 classifies the scene type (office, server room, corridor) for each camera every second | 3 | P1 | 2 |
| E04-S06 | As an analyst, the VLM (Qwen3.5-9B dev / Qwen3.5-27B prod, supersedes LLaVA-1.5) generates a natural-language description of each camera scene every 30 seconds | 5 | P1 | 2 |
| E04-S07 | As a security officer, VLM scene captioning is isolated from prompt injection (structured JSON context only, no raw text) | 5 | P0 | 2 |

### Epic E-05: Event & Alert Engine

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E05-S01 | As an operator, I receive a `PERSON_ENTERED` event within 2 seconds when a person crosses a defined zone boundary | 5 | P0 | 1 |
| E05-S02 | As a security officer, I receive a `TAILGATING_DETECTED` alert within 5 seconds of the tailgating event | 5 | P1 | 2 |
| E05-S03 | As an admin, I can define alert rules in YAML (conditions, time window, severity, actions) without code changes | 5 | P1 | 2 |
| E05-S04 | As an operator, I receive a `RACK_ACCESS` event when a person approaches Server Rack 1 | 3 | P1 | 2 |
| E05-S05 | As an operator, anomaly detection fires `UNUSUAL_ACTIVITY` when zone occupancy deviates > 3 standard deviations from the 7-day baseline | 8 | P2 | 3 |
| E05-S06 | As an operator, alerts are delivered via webhook, email, and WebSocket push with < 5 second latency | 5 | P0 | 1 |
| E05-S07 | As an admin, I can schedule maintenance windows that suppress specific rule classes | 3 | P2 | 3 |

### Epic E-06: Memory & Knowledge Graph

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E06-S01 | As an analyst, the current location of every tracked identity is available from Redis with < 1ms read latency | 3 | P0 | 1 |
| E06-S02 | As an analyst, zone presence history is queryable from TimescaleDB for any 90-day window | 5 | P0 | 2 |
| E06-S03 | As an analyst, I can query Neo4j: "Which persons were in Zone A between 2 PM and 4 PM?" using Cypher | 8 | P1 | 2 |
| E06-S04 | As an ML engineer, the memory consolidation pipeline promotes events from Redis → PostgreSQL → Neo4j on schedule | 5 | P1 | 2 |
| E06-S05 | As an analyst, per-identity behavioral baselines are computed from 7-day history and used for deviation detection | 8 | P2 | 4 |
| E06-S06 | As an admin, data retention policy auto-deletes events older than 90 days per compliance configuration | 3 | P1 | 3 |
| E06-S07 | As an admin, named entities (Server Rack 1 → AI Cluster, Camera 4 → Data Center Entrance) are configurable in the knowledge graph | 5 | P1 | 2 |

### Epic E-07: Natural Language Interface

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E07-S01 | As an operator, I can ask "Who is currently in Room A?" and receive a correct answer in < 500ms | 8 | P0 | 2 |
| E07-S02 | As an analyst, I can ask "What is Rahul doing right now?" and receive a correct answer in < 2s | 5 | P1 | 2 |
| E07-S03 | As an analyst, I can ask "When did John last enter the building?" and receive the correct timestamp | 5 | P1 | 2 |
| E07-S04 | As an analyst, I can ask "Show every appearance of Person X in the last 30 days" and receive a timeline with video references | 8 | P1 | 2 |
| E07-S05 | As an operator, I can ask "How many monitors are currently present?" and receive the current count | 3 | P1 | 2 |
| E07-S06 | As an analyst, I can ask "Did anyone access Server Rack 1 today?" and receive a list of events | 5 | P1 | 2 |
| E07-S07 | As an analyst, I can ask "Was there any unusual activity between 2 PM and 4 PM?" and receive relevant events | 5 | P1 | 2 |
| E07-S08 | As an analyst, I can ask "Did the cat enter the room today?" and receive an accurate answer | 5 | P2 | 2 |
| E07-S09 | As an analyst, I can ask "What changed since yesterday?" and receive a summarized event diff | 8 | P2 | 3 |
| E07-S10 | As a security officer, prompt injection attempts via the NL query input are logged and neutralized | 5 | P0 | 2 |
| E07-S11 | As an analyst, I can perform a forensic face search across 30-day archive in < 15 seconds | 8 | P1 | 3 |
| E07-S12 | As a security officer, every NL query is logged with user ID, timestamp, and query text | 3 | P0 | 2 |

### Epic E-08: MLOps & Continuous Learning

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E08-S01 | As an ML engineer, all training datasets are versioned with DVC on local NVMe | 5 | P1 | 3 |
| E08-S02 | As an ML engineer, every training run is tracked in MLflow with hyperparameters, metrics, and dataset hash | 5 | P1 | 3 |
| E08-S03 | As a security officer, promoting a model to production requires approval from both admin and security officer in MLflow | 5 | P0 | 3 |
| E08-S04 | As an ML engineer, the drift detector alerts when model confidence distribution KL divergence exceeds 0.1 | 5 | P1 | 3 |
| E08-S05 | As an operator, I can flag a wrong recognition result and it is stored as labeled feedback for the next training run | 3 | P1 | 3 |
| E08-S06 | As an ML engineer, a new model version is deployed in shadow mode for 24h before full promotion | 5 | P1 | 3 |
| E08-S07 | As an ML engineer, production model weights are SHA-256 signed and verified at startup | 3 | P0 | 2 |

### Epic E-09: Security & Compliance

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E09-S01 | As a security officer, the audit log hash chain is verified at startup and on-demand | 5 | P0 | 2 |
| E09-S02 | As a security officer, face embeddings are AES-256 encrypted at rest in PostgreSQL | 5 | P0 | 2 |
| E09-S03 | As a security officer, API endpoints enforce RBAC; unauthorized access returns 403 with audit log entry | 5 | P0 | 1 |
| E09-S04 | As a security officer, all Docker services run with AppArmor and Seccomp profiles; no privileged containers | 5 | P1 | 3 |
| E09-S05 | As a security officer, Falco monitors container syscalls and alerts on anomalous behavior | 5 | P2 | 3 |
| E09-S06 | As a security officer, adversarial example tests (FGSM/PGD) do not bypass anti-spoofing at > 5% rate | 8 | P1 | 3 |
| E09-S07 | As a security officer, camera streams use RTSPS (TLS) with per-camera credentials | 3 | P0 | 1 |
| E09-S08 | As an admin, deleted identities are soft-deleted with 30-day recovery window; hard deletion is logged | 3 | P1 | 2 |

### Epic E-10: Observability & Operations

| Story ID | User Story | Points | Priority | Phase |
|----------|------------|--------|----------|-------|
| E10-S01 | As a DevOps engineer, Prometheus scrapes metrics from all 14 pipeline services | 3 | P1 | 2 |
| E10-S02 | As an operator, I can see live GPU VRAM usage, per-model inference latency, and Kafka consumer lag on Grafana | 5 | P1 | 2 |
| E10-S03 | As an operator, I can see a live camera grid with zone overlays and occupancy counts on the operations dashboard | 5 | P0 | 1 |
| E10-S04 | As a security officer, the security dashboard shows spoof attempts, unknown persons, and access violations by time | 5 | P1 | 2 |
| E10-S05 | As a DevOps engineer, the system recovers from PostgreSQL WAL backup within 30 minutes | 8 | P1 | 3 |
| E10-S06 | As a DevOps engineer, a complete operations runbook documents startup, shutdown, failure recovery, and backup procedures | 5 | P2 | 3 |

---

## 8. Story Point Summary

| Epic | Total Points | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|-------------|---------|---------|---------|---------|
| E-01 Video Ingestion | 34 | 23 | 11 | — | — |
| E-02 Face Intelligence | 58 | 34 | 24 | — | — |
| E-03 Object & Motion | 42 | 13 | 29 | — | — |
| E-04 Behavioral Intelligence | 34 | — | 34 | — | — |
| E-05 Event & Alert Engine | 34 | 10 | 18 | 6 | — |
| E-06 Memory & Knowledge | 37 | 3 | 26 | 3 | 5 |
| E-07 NL Interface | 63 | — | 42 | 13 | 8 |
| E-08 MLOps | 31 | — | 3 | 28 | — |
| E-09 Security & Compliance | 39 | 13 | 13 | 13 | — |
| E-10 Observability | 34 | 8 | 13 | 13 | — |
| **Total** | **406** | **104** | **213** | **76** | **13** |

Velocity assumption: 26 points/sprint (2 weeks), 1 engineer. Adjust per team size.

---

## 9. Dependencies

```
E01-S01 (Docker infra)         must complete before ALL other stories
E01-S02 (Camera ingest)        must complete before E02-S01, E03-S01
E02-S01 (Face detection)       must complete before E02-S02, E02-S05
E02-S02 (Anti-spoofing basic)  must complete before E02-S05, E02-S07
E02-S05 (Face recognition)     must complete before E07-S01, E07-S02
E03-S01 (Object detection)     must complete before E03-S03, E03-S04, E03-S07
E03-S04 (ByteTrack)            must complete before E03-S05, E05-S01
E06-S01 (Redis live state)     must complete before E07-S01
E06-S02 (TimescaleDB events)   must complete before E07-S03, E07-S04, E07-S06
E06-S03 (Neo4j graph)          must complete before E07-S03, E07-S09
E07-S01 (NL basic queries)     must complete before E07-S04..S09
E08-S01 (DVC versioning)       must complete before E08-S02, E08-S03
E09-S03 (RBAC API)             must complete before all API stories
E10-S01 (Prometheus scraping)  must complete before E10-S02, E10-S04
```

---

## 10. Milestones

| Milestone | Target Date | Definition of Done |
|-----------|-------------|-------------------|
| M-01: Infrastructure Ready | Week 2 | All Docker services healthy; Kafka topics created; DB schemas applied |
| M-02: Face Pipeline MVP | Week 4 | Face detected, anti-spoofed (basic), recognized, enrolled via UI |
| M-03: MVP Complete | Week 8 | "Who is in Room A?" query answered correctly; 2-camera live; alerts firing |
| M-04: Full Pipeline | Week 14 | All 14 layers operational; 8 cameras; NL queries working |
| M-05: Security Hardened | Week 20 | Anti-spoofing ACER < 2%; audit log verified; RBAC enforced; embeddings encrypted |
| M-06: MLOps Live | Week 24 | DVC + MLflow running; drift detection active; dual-approval model promotion |
| M-07: Forensic Search | Week 28 | 30-day archive searchable; face search < 15s; timeline export working |
| M-08: Production Release | Week 32 | All NFRs met; runbooks complete; load test passed; security audit passed |
| M-09: Behavioral Intelligence | Week 40 | Behavioral baselines per identity; deviation detection; pattern mining |
| M-10: Enterprise Complete | Week 48 | Multi-node design documented; predictive analytics; advanced animal detection |

---

## 11. Release Plan

### Phase 1 — MVP (Sprints 1–4, Weeks 1–8)

**Sprint 1 (Weeks 1–2): Foundation**
- E01-S01: Docker Compose infra (PostgreSQL, Redis, Kafka, pgvector, TimescaleDB)
- E01-S07: Structured logging, basic Grafana setup
- E09-S07: RTSPS camera credentials setup
- E09-S03: FastAPI skeleton with JWT auth and RBAC roles
- E10-S03: Basic camera grid dashboard

**Sprint 2 (Weeks 3–4): Face Pipeline Core**
- E01-S02: GStreamer + NVDEC single camera ingest
- E02-S01: SCRFD-10GF face detection (TensorRT FP16)
- E02-S02: MiniFASNet-V2 anti-spoofing (pretrained)
- E02-S05: AdaFace R100 face recognition + pgvector gallery
- E02-S09: Enrollment audit log entries
- E06-S01: Redis live state (current location per identity)

**Sprint 3 (Weeks 5–6): Objects + Events + Tracking**
- E03-S01: YOLOv9-C object detection
- E03-S03: Zone-level object counts to dashboard
- E03-S04: ByteTrack tracking
- E05-S01: PERSON_ENTERED / PERSON_EXITED events
- E05-S06: Webhook + WebSocket alert delivery
- E02-S06: Unknown person enrollment candidate workflow

**Sprint 4 (Weeks 7–8): MVP Completion**
- E02-S07: Admin enrollment approval flow
- E06-S01: Redis current-state fully populated
- E07-S01: "Who is in Room A?" NL query (Redis + identity lookup only)
- E10-S03: Live camera grid with zone overlays complete
- E01-S04: Camera disconnect watchdog and reconnect
- **Milestone M-03: MVP Complete**

**MVP Acceptance Criteria:** (see `live_testing/sprint4_live_tests.md` for
verification detail and what still needs human observation)
- [x] "Who is currently in Room A?" returns correct answer in < 500ms
- [ ] Unknown person triggers enrollment candidate in operator UI within 30 seconds — UI built, needs visual confirmation
- [ ] Printed photo rejected by anti-spoofing in 100% of basic test cases — intentionally deferred to Sprint 5
- [x] `PERSON_ENTERED` event fires within 2 seconds of zone entry
- [x] Alert delivered via webhook within 5 seconds
- [ ] 2 cameras at 10 FPS with VRAM < 12 GB — VRAM confirmed (~2.8GB); real 2-camera throughput needs the physical camera back online
- [x] Audit log records all enrollments and operator actions

---

### Phase 2 — Production (Sprints 5–10, Weeks 9–20)

**Sprint 5 (Weeks 9–10): Full Anti-Spoofing**
- E02-S03: CDCN++ + temporal consistency ensemble
- E02-S04: Deepfake detection via CDCN++
- E02-S10: Anti-spoofing fine-tuning pipeline + domain data collection
- E08-S07: Model weight SHA-256 signing

**Sprint 6 (Weeks 11–12): Advanced Tracking + Scale**
- E03-S05: OSNet ReID + StrongSORT integration
- E03-S06: Cross-camera tracking with UUID-based IDs
- E01-S03: Scale to 8 simultaneous cameras
- E03-S07: Object removal / addition events

**Sprint 7 (Weeks 13–14): Knowledge Graph + Memory + Pipeline Restructure**
- E06-S02: TimescaleDB zone presence history
- E06-S03: Neo4j deployment + schema + Cypher query interface
- E06-S04: Memory consolidation pipeline (Redis → PG → Neo4j)
- E06-S07: Named entity configuration (Server Rack 1 → AI Cluster, etc.)
- E11-S01: Frames off Kafka — decode + inference co-located on GPU node; only metadata/events/crops on Kafka (see §15, prerequisite for H100 scale)
- E11-S02: Evaluation harness v1 — labeled clip set from deployment cameras; per-model metrics (mAP, TAR@FAR, ACER, IDF1) logged to MLflow on every model change

**Sprint 8 (Weeks 15–16): Behavioral Intelligence**
- E04-S01: X3D-M always-on action detection
- E04-S02: VideoMAE-B tailgating detection (fine-tuned)
- E04-S03: Loitering alert via action recognition
- E05-S02: TAILGATING_DETECTED alert end-to-end
- E05-S03: YAML rule engine with custom rule support

**Sprint 9 (Weeks 17–18): NL Query Engine**
- E07-S02: "What is Rahul doing?" query
- E07-S03: "When did John last enter?" query
- E07-S05: "How many monitors?" query
- E07-S06: "Did anyone access Server Rack 1?" query
- E07-S07: "Was there unusual activity?" query
- E07-S10: Prompt injection defense
- E07-S12: NL query audit logging

**Sprint 10 (Weeks 19–20): Scene Understanding + Security Hardening**
- E04-S05: CLIP scene classification
- E04-S06: Qwen3.5 scene captioning (9B quantized on 3090; 27B on H100) — replaces LLaVA-1.5-7B and Mistral-7B for NLQ
- E04-S07: VLM prompt injection isolation
- E09-S01: Audit log hash chain verification
- E09-S02: Face embedding AES-256 encryption
- E09-S08: Soft-delete for identities
- E10-S01: Prometheus metrics for all services
- E10-S02: Full performance Grafana dashboard
- **Milestone M-05: Security Hardened**

**Production Acceptance Criteria:**
- [ ] All 20 example queries in requirements answered correctly
- [ ] Anti-spoofing ACER < 2% on domain fine-tuned test set
- [ ] Cross-camera tracking IDF1 > 80% on deployment test
- [ ] 8 cameras at 10 FPS with VRAM < 22 GB
- [ ] Audit log hash chain verified intact
- [ ] RBAC enforced for all 5 roles on all endpoints
- [ ] Face embeddings encrypted at rest

---

### Phase 3 — Enterprise (Sprints 11–16, Weeks 21–32)

**Sprint 11–12 (Weeks 21–24): MLOps Pipeline + Model Upgrades**
- E08-S01: DVC dataset versioning
- E08-S02: MLflow experiment tracking
- E08-S03: Dual-approval model promotion
- E08-S04: Drift detector (KL divergence hourly)
- E08-S05: Operator feedback → training loop
- E08-S06: Shadow mode deployment (24h blue-green)
- E11-S03: Face recognition upgrade — ArcFace R50 (buffalo_l) → AdaFace IR-101 (WebFace12M); full gallery re-enrollment; `model_version` enforced in gallery search (benchmark TopoFR R100 as alternative via E11-S02 harness)
- E11-S04: Object detection upgrade — YOLOv9-C → RF-DETR-M or D-FINE-L, gated on E11-S02 eval; add YOLO-World (open-vocabulary) for NLQ-driven detection of untrained classes
- E11-S05: ReID upgrade — OSNet → SOLIDER-ReID or CLIP-ReID, fine-tuned on deployment cameras
- **Milestone M-06: MLOps Live**

**Sprint 13–14 (Weeks 25–28): Security Hardening + Forensics**
- E09-S04: AppArmor + Seccomp Docker profiles
- E09-S05: Falco container monitoring
- E09-S06: Adversarial example tests (FGSM/PGD)
- E07-S11: Retroactive forensic face search (< 15s for 30-day archive)
- E01-S06: Video replay from archive
- E01-S05: Archive HMAC verification
- E07-S09: "What changed since yesterday?" query
- E06-S06: Data retention auto-delete policy
- E11-S06: Anti-spoofing upgrade — add CLIP/language-guided FAS member (FLIP-style) to the ensemble; benchmark on OULU-NPU + SiW-M
- E11-S07: Multimodal liveness hardware — IR/depth camera (e.g. RealSense) at enrollment station and high-security choke points; depth signal feeds ensemble hard-AND
- **Milestone M-07: Forensic Search**

**Sprint 15–16 (Weeks 29–32): Performance + Operations**
- E05-S05: Statistical anomaly detection (Z-score baseline)
- E05-S07: Maintenance window suppression
- E09-S07: Full RTSPS with certificate pinning
- E10-S04: Security Grafana dashboard
- E10-S05: PostgreSQL WAL disaster recovery test
- E10-S06: Operations runbook
- E11-S08: Inference serving — all vision models behind Triton Inference Server with TensorRT FP16 engines (3090) / FP8 (H100); per-model latency SLOs in Prometheus
- E11-S09: H100 production deployment profile — vLLM for Qwen3.5-27B, camera count and VRAM budget re-baselined for 80 GB
- Load test: 8 cameras + 10 concurrent NL queries
- **Milestone M-08: Production Release**

**Enterprise Acceptance Criteria:**
- [ ] Adversarial FGSM/PGD attacks do not bypass anti-spoofing > 5%
- [ ] Forensic face search across 30-day archive in < 15 seconds
- [ ] Model weights signed and verified at startup
- [ ] Dual-approval model promotion working end-to-end
- [ ] Disaster recovery: PostgreSQL restore from WAL < 30 minutes
- [ ] All Docker services running with AppArmor + Seccomp profiles
- [ ] Full operations runbook documented and tested

---

### Phase 4 — Advanced Intelligence (Sprints 17–24, Weeks 33–48)

**Sprint 17–20 (Weeks 33–40): Behavioral Baselines + Predictive Analytics**
- E06-S05: Per-identity behavioral baselines from 7-day history
- Deviation detection: person arrived outside baseline window
- PrefixSpan pattern mining on event sequences
- Zone occupancy forecasting (ARIMA or lightweight LSTM)
- Arrival time prediction per enrolled identity
- **Milestone M-09: Behavioral Intelligence**

**Sprint 21–22 (Weeks 41–44): Advanced Animal Detection**
- E04-S04: Animal species fine-tuning (iNaturalist-local)
- Animal behavioral analysis (movement patterns in zones)
- Habitat zone monitoring (define zones for animal movement tracking)

**Sprint 23–24 (Weeks 45–48): Multi-Node Preparation**
- Multi-GPU pipeline profiling and bottleneck analysis
- API gateway design for multi-node federation
- Distributed Kafka deployment design document
- Edge node architecture document (RTSP forward + lightweight inference)
- **Milestone M-10: Enterprise Complete**

---

## 12. Sprint Template

Each 2-week sprint follows this structure:

| Day | Activity |
|-----|----------|
| 1 (Mon) | Sprint planning: pull stories from backlog, confirm acceptance criteria, assign |
| 1–9 | Development, daily standup (15 min: what did I do, what will I do, any blockers) |
| 9 (Thu) | Sprint demo: show working software against acceptance criteria |
| 10 (Fri) | Sprint retrospective (15 min: what worked, what didn't, one improvement) |
| 10 (Fri) | Backlog refinement: estimate and prioritize stories for next sprint |

**Definition of Done (applies to every story):**
- [ ] Feature implemented and manually verified
- [ ] Unit test written and passing
- [ ] No new Prometheus alerts in red
- [ ] Audit log records the new action (if user-facing)
- [ ] Code reviewed (self-review minimum; peer review preferred)
- [ ] VRAM budget not exceeded (confirm with `nvidia-smi` in test)

---

## 13. Team Roles (Minimum Viable Team)

| Role | Responsibilities |
|------|-----------------|
| ML Engineer | Model integration, fine-tuning, VRAM optimization, evaluation |
| Backend Engineer | Pipeline services, Kafka consumers, FastAPI, data schemas |
| DevOps / MLOps | Docker, Kafka, databases, Prometheus, DVC, MLflow, CI |
| Security Officer | Audit log review, model promotion approval, adversarial testing, incident response |
| Product Owner | Backlog prioritization, acceptance criteria, stakeholder communication |

Single-person teams: wear all hats; use the Phase ordering to stay on the critical path.

---

## 14. Key Metrics to Track

| Metric | Target | Tracked In |
|--------|--------|-----------|
| Face recognition TAR@FAR=1e-4 | ≥ 92% | MLflow monthly eval |
| Anti-spoofing ACER | ≤ 2% | MLflow + red-team |
| Cross-camera ReID mAP | ≥ 85% | MLflow quarterly eval |
| NL query p50 latency | ≤ 2s | Grafana / Prometheus |
| NL forensic query p95 | ≤ 15s | Grafana / Prometheus |
| VRAM peak under full load | ≤ 22 GB | Prometheus |
| Kafka consumer lag (events.detected) | ≤ 2s | Grafana |
| Camera uptime | ≥ 99.5% | Prometheus |
| Audit log hash chain validity | 100% | Startup + hourly check |
| Model drift KL divergence | < 0.1 | Drift detector hourly |
| Object detection mAP50:95 (deployment eval set) | tracked per release | MLflow via E11-S02 harness |

---

## 15. Model & Architecture Upgrade Roadmap (Epic E11, added 2026-06-10)

Researched against the June 2026 state of the art. **Rule: no model swap ships without
a before/after run on the E11-S02 evaluation harness** — upgrades are measured, not vibes.

### Hardware strategy

| Environment | GPU | Role |
|---|---|---|
| Dev/test | RTX 3090 24 GB | Full pipeline at reduced scale; quantized VLM (Qwen3.5-9B 4-bit); TensorRT FP16 |
| Production | H100 80 GB | 8+ cameras full pipeline; Qwen3.5-27B via vLLM (FP8); TensorRT FP8 engines; headroom for shadow-mode A/B |

Everything must run on both — the 3090 profile is the H100 profile with smaller
VLM weights and lower camera count, not a different architecture.

### Model upgrade matrix

| Component | Current (shipped) | Target | Why |
|---|---|---|---|
| Face detection | SCRFD-10GF | **Keep** | Still best accuracy-per-FLOP for faces; not a bottleneck |
| Face recognition | ArcFace R50 (buffalo_l `w600k_r50`) — *plan said AdaFace R100; code never got it* | **AdaFace IR-101 (WebFace12M)**; benchmark TopoFR R100 | AdaFace's quality-adaptive margin is the strongest published approach for low-quality surveillance imagery (IJB-C hard sets); TopoFR leads clean benchmarks |
| Anti-spoofing | MiniFASNet-V2 + CDCN++ (RGB-only) | Keep ensemble; **add FLIP-style CLIP-guided FAS member + IR/depth at choke points** | RGB-only PAD has a hard ceiling; multimodal (RGB+depth/IR) reports 40–60% better spoof detection — this is a sensor problem more than a model problem |
| Object detection | YOLOv9-C | **RF-DETR-M or D-FINE-L** (+ **YOLO-World** for open-vocab) | Real-time DETRs now Pareto-dominate YOLO (RF-DETR-M: 54.7 mAP @ 4.5 ms on T4); NMS-free = simpler TensorRT graph; open-vocab lets NLQ find untrained classes |
| Tracking | ByteTrack / StrongSORT | **Keep**; re-evaluate BoT-SORT on harness | Tracker choice is not the current accuracy bottleneck; ReID features are |
| ReID | OSNet (2019) | **SOLIDER-ReID or CLIP-ReID**, fine-tuned on deployment cameras | Transformer/self-supervised human-centric pretraining substantially outperforms OSNet cross-camera |
| LLM (NLQ) | Mistral-7B-instruct-v0.3 | **Qwen3.5-9B (3090) / Qwen3.5-27B (H100)** | Qwen3.5 (Feb 2026, Apache-2.0) is natively multimodal — one model replaces both Mistral and LLaVA; 27B matches GPT-5-mini-class on reasoning; 262K context for long timelines |
| VLM (scenes) | LLaVA-1.5-7B | **Same Qwen3.5 instance** | Early-fusion multimodal: scene captioning and NLQ share one deployment (vLLM); halves VRAM vs two models. Note: there is no separate "Qwen3.5-VL" — Qwen3.5 *is* the VL model; track Qwen3.6 (27B dense / 35B-A3B) for the next eval cycle |
| Serving | Per-process ONNX Runtime | **Triton Inference Server + TensorRT** | Concurrent model execution, dynamic batching across 8 cameras, per-model metrics; standard path to multi-GPU/multi-node (§ Sprint 23–24) |

### Architectural commitments

1. **Frames never travel through Kafka in production** (E11-S01, Sprint 7). Decode and
   inference stay co-located on the GPU node; Kafka carries metadata, events, and face
   crops only. `camera.frames` becomes dev-only. This is the single change that makes
   the H100 (and later multi-node) scale-out work.
2. **One embedding space per gallery** (E11-S03, R-13). `face_gallery.model_version`
   becomes a hard filter in search SQL; switching recognition models requires
   re-enrollment of all identities. Never mix R50 and IR-101 embeddings.
3. **Eval harness before upgrades** (E11-S02, Sprint 7). Labeled clips from the real
   deployment cameras; every candidate model gets TAR@FAR / ACER / mAP / IDF1 logged to
   MLflow. The data flywheel (E08-S05 feedback loop + this harness) is the long-term
   moat — model picks above are a snapshot, the harness is what keeps the system
   state-of-the-art as releases continue.

### E11 story index

| Story | Sprint | Summary |
|---|---|---|
| E11-S01 | 7 | Frames off Kafka; metadata-only topics |
| E11-S02 | 7 | Evaluation harness v1 (deployment clip set + MLflow) |
| E11-S03 | 11–12 | AdaFace IR-101 migration + gallery re-enrollment + `model_version` enforcement |
| E11-S04 | 11–12 | RF-DETR/D-FINE swap + YOLO-World open-vocab |
| E11-S05 | 11–12 | SOLIDER/CLIP-ReID replacing OSNet |
| E11-S06 | 13–14 | CLIP-guided FAS ensemble member |
| E11-S07 | 13–14 | IR/depth liveness hardware at choke points |
| E11-S08 | 15–16 | Triton + TensorRT serving for all vision models |
| E11-S09 | 15–16 | H100 production profile (vLLM Qwen3.5-27B, re-baselined budgets) |
