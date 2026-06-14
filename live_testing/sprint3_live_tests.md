# Sprint 3 Live Tests — Objects + Events + Tracking

Scope (AGILE_PLAN): YOLO26 object detection (E03-S01), zone-level object
counts (E03-S03), ByteTrack tracking (E03-S04), PERSON_ENTERED /
PERSON_EXITED events (E05-S01), webhook + WebSocket alert delivery
(E05-S06), unknown-person enrollment candidate workflow (E02-S06).

Prerequisites: Sprint 2 fully passed (recognition live), infra containers
healthy, at least one identity enrolled ("Rohan").

> **BUILD STATUS: all six components are built.** For reference:
> `scripts/run_objects.py` (live YOLO runner), `src/event_detection/`
> (ZonePresenceMonitor + EventStore w/ identity_state + append-only
> IDENTITY_CORRECTED), `src/object_counting/counter.py`,
> `src/alerts/engine.py` + `delivery.py` (+ `/api/v1/alerts/ws` and
> `scripts/webhook_listener.py`), candidate capture in
> `src/face_recognition/candidates.py` with review endpoints under
> `/api/v1/identities/candidates`. Rules with predicates beyond
> `event_type`/`zone_type` equality (tailgating, loitering) load as
> INACTIVE until the temporal-reasoning sprint.

Hardware note (Linux 3090 on PCIe x1): YOLO26 at 1280 will be slower
than the face pipeline. If fps < 8, drop input size to 960 or 640 —
detection quality for room-scale (person-sized) objects is barely affected.

---

## READ FIRST — what to expect, what can go wrong, what to accept

This sprint validates the **foundation**: people, zones, events, alerts,
attribution, candidates. The intelligence built ON this (suspicious
behavior, theft correlation, small-asset protection) is later sprints —
see "Deferred — accept for now" below so you don't chase the wrong target.

### ✅ What to EXPECT to work (this is the pass bar)

- **People** detected and tracked reliably — a person is large in frame, so
  this is the solid part. Body track survives a full rotation (unlike the
  face-only track in Sprint 2).
- **PERSON_ENTERED / PERSON_EXITED** persisted, debounced (one each per real
  entry/exit, no blink events), each carrying an identity snapshot
  (`person`, `identity_state`, `similarity`).
- **Recognition** names enrolled people on the live feed; unknown people stay
  `unknown` and raise `UNKNOWN_PERSON_DETECTED`.
- **Restricted-zone alert** fires on entry to a zone seeded as `restricted`,
  once per cooldown; **webhook + WebSocket** both deliver it within ~1s.
- **OBJECT_ADDED / OBJECT_REMOVED** for clearly-visible, person-/box-sized
  objects (bring a bag/laptop into view, take it away).
- **Identity attribution:** events during an at-risk window are
  `provisional`; a relabel/demote writes an append-only `IDENTITY_CORRECTED`
  row; original event rows are never edited.
- **Unknown person → pending candidate → approve → recognized live.**

### ⚠ What can GO WRONG (real bugs — report these with the console output)

- **`inference: CPU` on the GPU box** / `libcublasLt.so.12` errors → CUDA
  install drift; fix per Sprint 2's CUDA section before continuing.
- **Events not appearing in the DB** → did you pass `--persist-events`? is
  Postgres up and the schema loaded (STEP 0)?
- **Alert never fires** → is the zone seeded with `zone_type='restricted'`?
  are you inside the cooldown window from a previous fire?
- **WebSocket closes immediately (code 4401)** → token must be passed as
  `?token=<access-token>` (browsers can't set WS headers).
- **Webhook never arrives** → listener running on the right port? value in
  `.env` `ALERT_WEBHOOK_URL`? (delivery failure is logged and must NOT stall
  the pipeline — if the feed freezes on a dead webhook, that's a bug.)
- **A name sticks to the wrong person after two people cross** for more than
  a few seconds of clear frontal views → report (the demotion guard should
  catch it).
- **fps that *grows* its lag over time** (vs a constant small delay) → a
  backlog bug; report. Constant small lag on the x1 link is normal.

### 🟡 DEFERRED — accept for now (NOT failures; later sprints/efforts)

- **Small items barely detected (~10%)** — shelf clutter, distant/tiny
  objects. This is the small-object problem; the fix (SAHI tiling + a closer
  dedicated camera + fine-tuning on a few valuables) is a separate inventory
  effort, not this sprint. Don't treat missed small items as a failure here.
- **Domain objects misclassified in closed-set** (almirah→fridge,
  iPad→laptop) — COCO has no such classes; open-vocab `--vocab` / fine-tuning
  address it later.
- **Krishna statue detected as `person`** — known COCO limit; the identity
  layer keeps it `unknown`, anti-spoofing rejects it in a later sprint.
- **A pendrive / tiny critical asset is NOT trackable on the wide camera** —
  by design no room CCTV catches a palmed 2cm object. Protecting it needs a
  tight asset-zone + close camera + non-camera sensors (USB logging, drawer
  contact) + forensic "who was near it" — a dedicated critical-asset track,
  later.
- **Suspicious-*behavior* detection** (rummaging, climbing, loitering nuance,
  tailgating) — needs action recognition + the anomaly engine = **Sprint 8**.
  The tailgating/loitering rules are written but load INACTIVE until then.
- **Anomaly / "abnormal for this scene at this hour"** — behavioral
  baselines = **Phase 4**.
- **Full theft correlation** ("unknown + restricted + object-removed + odd
  hour → THEFT") — the fusion engine is Sprint 8; this sprint only emits the
  individual building-block events.
- **Cross-camera identity continuity** — Sprint 6 (body ReID).
- **Multi-person track churn (single camera):** when people cross, occlude,
  or pass the frame edge, face-only tracking drops and re-creates track IDs,
  producing duplicate/spurious PERSON_ENTERED (same person entering "twice"
  with no exit) and phantom `unknown` entries. The event layer records this
  faithfully — the churn is in the tracker. Fixes: Sprint 6 body ReID
  (stable IDs) + identity-level presence (see docs/decisions/0007). For
  Sprint 3 foundation sign-off, test SINGLE-person flows.
- **Box disappears/reappears at the frame edge** — face partially out of
  frame → detection drops → track coasts/re-creates. Inherent to face boxes
  at boundaries; body tracking smooths it.
- **Lower fps than the face pipeline** on the PCIe x1 link — use
  `--imgsz 640`; expected, not a bug.

---

## STEP 0 — Infrastructure

```bash
docker compose up -d postgres redis
docker compose ps
```

**Expected:** both healthy. The TimescaleDB schema (system_events,
object_counts) is loaded into the same postgres container via
`infrastructure/timescaledb/schema.sql` — verify:

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "\dt" \
  | grep -E "system_events|object_counts|zones|zone_presence|alerts|enrollment_candidates"
```

**Expected:** all six tables listed.

**If `system_events` / `object_counts` are MISSING** (the time-series schema
only runs on first-ever container init, and an older version aborted on the
missing timescaledb extension), apply the corrected schema to the running DB:

```bash
sudo docker exec -i third-eye-postgres-1 psql -U thirdeye -d thirdeye \
  < infrastructure/timescaledb/schema.sql
```

You'll see `NOTICE: timescaledb unavailable — using plain tables` — that's
expected and fine (the pgvector image has no timescaledb; the tables work as
plain PostgreSQL). Re-run the verify; all six should now appear.

---

## STEP 1 — Unit tests for the object/event layer

```bash
.venv/bin/python -m pytest tests/unit/test_object_detector.py \
  tests/unit/test_tracker.py tests/unit/test_zone_events.py -q
```

**Expected:** all pass (tracker suite includes the Sprint-2-hardened
appearance/bank tests).

---

## STEP 2 — YOLO26 weights

YOLO26 (Ultralytics, Jan 2026) needs a current ultralytics:

```bash
.venv/bin/pip install -U ultralytics
.venv/bin/python -c "
from ultralytics import YOLO
m = YOLO('yolo26m.pt')          # auto-downloads
print('classes:', len(m.names), '| person id:', [k for k,v in m.names.items() if v=='person'])
"
```

**Expected:** `classes: 80 | person id: [0]`. (If `yolo26m.pt` fails to
download, your ultralytics is too old — upgrade it; as a fallback any
`yolo11m.pt`/`yolov9c.pt` works with `--model`.)

---

## STEP 3 — Live object detection + tracking

Detection has three modes, in increasing accuracy/effort. **Important:**
COCO's 80 classes do not include domain items — an *almirah* will read as
`fridge`, an *iPad* as `laptop`, and shelf items may be missed. That is a
dataset limit, not a bug; 3b and 3c fix it.

### 3a — YOLO26 closed-set (COCO 80)

```bash
.venv/bin/python scripts/run_objects.py --source <rtsp-url> --show --fps 15 \
  --camera-id cam0 --zone-id bedroom --model yolo26m.pt --imgsz 960
```

**Expected:**
- Banner: GPU (CUDA), model yolo26m, detection mode closed-set (COCO)
- Persons + COCO objects (chair, bed, bottle, laptop, cell phone) boxed;
  HUD/overlay as in the face runner
- ONE stable person track through a full body rotation (body boxes track
  far better than face boxes)
- Person/phone flicker should be markedly reduced vs the old yolov9c; if a
  person still flickers, try `--model yolo26l.pt --imgsz 1280`
- `OBJECT_ADDED` once when an object appears, `OBJECT_REMOVED` ~5 s after
  removal
- Misclassification of domain items (almirah→fridge, iPad→laptop) WILL
  remain here — proceed to 3b

**Pass (3a):** person + ≥3 COCO classes, stable body track, ADDED/REMOVED
fire once each.

### 3b — Open-vocabulary (detect your actual objects, no training)

List the things in *your* room as **plain, CLIP-friendly nouns**; YOLO-World
detects exactly those:

```bash
.venv/bin/python scripts/run_objects.py --source <rtsp-url> --show --fps 10 \
  --camera-id cam0 --zone-id bedroom --imgsz 1280 \
  --vocab "person,wardrobe,laptop,tablet,backpack,cardboard box,electric fan,water bottle,books,shoes,bed,chair"
```

**Prompt wording is everything here** — YOLO-World matches your text against
CLIP's vocabulary:
- Use words CLIP knows: `wardrobe`/`cupboard` (not `almirah`), `electric fan`
  or `pedestal fan` (not just `fan`), `cardboard box` (not `box`),
  `laptop`/`tablet` (not `macbook`/`ipad`).
- Confidence auto-drops to **0.05** in open-vocab mode (YOLO-World scores
  much lower than closed YOLO — the old 0.45 default hid almost everything).
  Still missing things? Lower further: `--conf 0.02`.
- `--imgsz 1280` materially helps small shelf items.

**Expected:**
- Banner: detection mode open-vocab (N classes), conf 0.05, vocab listed
- The wardrobe, fan, backpack, boxes, shelf items you named are now boxed
- Slower per frame than 3a (use it to validate classes, not for max fps)

**Honest limits of open-vocab:** close synonyms can swap (a laptop may read
as `tablet` and vice-versa — CLIP embeds them near each other), and odd
angles/clutter still get missed. This is correctness-for-free without
training, not perfection — for reliable, unambiguous labels on YOUR objects,
3c (fine-tuning) is the real fix.

**Pass (3b):** objects missed/misclassified in 3a (wardrobe, fan, backpack,
boxes) are now detected when listed in --vocab with the low conf. Some
synonym confusion (laptop/tablet) is acceptable here.

### 3c — Fine-tune on your own footage (the permanent fix)

The full loop you sketched — propose → review → correct → save → train:

```bash
# 1. capture & auto-label (open-vocab bootstraps the boxes)
.venv/bin/python scripts/capture_dataset.py --source <rtsp-url> \
  --vocab "person,wardrobe,laptop,tablet,backpack,cardboard box,electric fan" \
  --out datasets/room --every 1.0 --max 300

# 2. REVIEW/CORRECT the auto-labels (this is the step that buys accuracy)
.venv/bin/python scripts/review_labels.py --dataset datasets/room
#   sidebar lists classes with hotkeys. Per frame:
#     click a box to select it; press 1-9/0 to set its class (or click the
#     class row); drag empty area = new box; d = delete; n/p = nav.
#   CARRY-FORWARD (on by default, toggle 't'): each frame starts from your
#     CORRECTED previous frame, not the model's noisy per-frame guess — so on
#     a static camera you label the room once and only fix what moved (the
#     person). 'v' copies the previous frame again; 'a' resets a frame to the
#     model's original auto-labels.
#   Heavy corner-dragging at scale? Import datasets/room into Label Studio
#   or CVAT (local, free) — same YOLO format — then re-export and train.

# 3. train
.venv/bin/python scripts/train_detector.py --data datasets/room/data.yaml \
  --base yolo26m.pt --epochs 100 --name room_v1

# 4. run live on the trained model
.venv/bin/python scripts/run_objects.py --source <rtsp-url> --show \
  --model runs/detect/room_v1/weights/best.pt
```

**Expected:** after training, the domain model detects your specific
objects reliably at full closed-set speed (no open-vocab slowdown), with
far fewer misclassifications than COCO.

**Pass (3c):** trained model runs in run_objects and names your domain
objects correctly. (This step is optional for closing Sprint 3 — 3a+3b
satisfy the detection criteria; 3c is the production-quality follow-up.)

---

## STEP 4 — Zones + PERSON_ENTERED / PERSON_EXITED

The runner now self-registers its `--camera-id` and `--zone-id` at startup
(needed so `zone_presence`'s foreign keys are satisfied — otherwise the first
PERSON_ENTERED's presence insert FK-violates and silently aborts event
handling). So no manual seeding is required for a general zone. You only need
to seed a zone explicitly to make it **restricted** (for the alert test):

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
INSERT INTO zones (zone_id, display_name, zone_type) VALUES
  ('doorway',  'Doorway',  'restricted')
ON CONFLICT (zone_id) DO UPDATE SET zone_type = 'restricted';"
```

Run the face runner with event persistence:

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --show --fps 25 \
  --recognize --camera-id cam0 --zone-id bedroom --persist-events
```

Walk out of frame (>6 s), walk back in. Then check (times are stored in UTC —
the `AT TIME ZONE` converts to local; change the zone to yours):

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
SELECT event_type, zone_id, payload->>'person' AS person,
       payload->>'identity_state' AS identity_state,
       event_time AT TIME ZONE 'Asia/Kolkata' AS local_time
FROM system_events ORDER BY event_time DESC LIMIT 10;"
```

**Expected:**
- `PERSON_ENTERED` when your track is confirmed in the zone,
  `PERSON_EXITED` after you leave (grace period, no blink events)
- `person` = `Rohan` with `identity_state = verified` for clean tracks
- On re-entry you'll often see a fresh `PERSON_ENTERED` with `person=unknown`
  (recognition hasn't caught up at the instant of entry) followed seconds
  later by an `IDENTITY_CORRECTED` row (unknown → Rohan). The entry row stays
  unknown on purpose (append-only); the correction re-attributes it. Console
  prints `>> IDENTITY_CORRECTED track N: unknown -> Rohan`.
- An event fired while your track was at-risk (right after a gap return /
  crossing) carries `identity_state = provisional`
- `zone_presence` rows show your entry/exit times with `is_unknown=false`

**Pass:** one ENTERED + one EXITED per actual entry/exit, correctly named,
with identity_state present on every event.

---

## STEP 5 — Zone object counts

While STEP 3 or 4 is running:

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
SELECT bucket_time, zone_id, object_class, count
FROM object_counts ORDER BY bucket_time DESC LIMIT 12;"
```

**Expected:** periodic rows (e.g. 10 s buckets) per (zone, class):
`bedroom | person | 1`, `bedroom | bottle | 1`, counts changing when
objects enter/leave.

---

## STEP 6 — Events API

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -d "username=admin&password=<your-admin-pw>" \
  -H "Content-Type: application/x-www-form-urlencoded" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s "http://127.0.0.1:8000/api/v1/events?limit=5" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected:** the STEP 4 events, newest first; unauthenticated request
returns 401.

---

## STEP 7 — Alert rules → alerts rows

`doorway` is seeded as `restricted`. Re-run STEP 4's command with
`--zone-id doorway`, walk into frame.

```bash
curl -s "http://127.0.0.1:8000/api/v1/alerts?status=open" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected:**
- Rule `person_entered_restricted` (HIGH) fires → one `alerts` row with
  your person_id, zone `doorway`
- Cooldown honored: re-entering within 30 s does NOT create a second alert
- Acknowledge + resolve round-trip works:

```bash
curl -s -X PUT "http://127.0.0.1:8000/api/v1/alerts/<alert_id>/acknowledge" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
curl -s -X PUT "http://127.0.0.1:8000/api/v1/alerts/<alert_id>/resolve" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Pass:** alert fires once per cooldown window, ack/resolve transition the
status, audit_log records both actions.

---

## STEP 8 — Webhook + WebSocket delivery

Terminal A — webhook listener (script provided with the build):

```bash
.venv/bin/python scripts/webhook_listener.py --port 9000
# and set in .env:  ALERT_WEBHOOK_URL=http://127.0.0.1:9000/hook
```

Terminal B — WebSocket subscriber (needs `pip install websockets`; uses the
$TOKEN from STEP 6 — browsers can't set headers on WS, so auth is a query
param):

```bash
.venv/bin/python -c "
import asyncio, json, os, websockets
async def main():
    url = 'ws://127.0.0.1:8000/api/v1/alerts/ws?token=' + os.environ['TOKEN']
    async with websockets.connect(url) as ws:
        print('subscribed — waiting for alerts ...')
        while True: print(json.loads(await ws.recv()))
asyncio.run(main())"
```

Trigger the STEP 7 alert again (after cooldown).

**Expected:** both terminals receive the alert JSON within ~1 s of the
zone entry; webhook delivery failure (stop the listener) is logged and
retried, and never blocks the pipeline.

---

## STEP 9 — Identity attribution under uncertainty

The Sprint-2 finding, now enforced: events must never launder a guess
into a fact.

With `--persist-events --recognize` running, have a second (unenrolled)
person cross paths with you, then check:

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
SELECT event_type, payload->>'person' AS person,
       payload->>'identity_state' AS state, event_time
FROM system_events
WHERE event_type IN ('PERSON_ENTERED','PERSON_EXITED','IDENTITY_CORRECTED')
ORDER BY event_time DESC LIMIT 15;"
```

**Expected:**
- Events fired during the crossing window show `state = provisional`
- If a demotion/relabel happened, an `IDENTITY_CORRECTED` record exists
  covering that track + time range — and the original event rows are
  UNCHANGED (append-only; the audit chain stays tamper-evident)

**Pass:** no event claims `verified` identity during an at-risk window;
corrections appear as new records, never edits.

---

## STEP 10 — Unknown-person enrollment candidates

Have the unenrolled person stand in view ~10 s with their face visible.

```bash
curl -s "http://127.0.0.1:8000/api/v1/identities/candidates?status=pending" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected:** one pending candidate with `crop_count` ≥ 5, `crop_paths`
(MinIO keys of saved face photos), a 512-dim mean embedding, and — if any
faces are enrolled — a `suggested_person_id`/`suggested_name`/
`suggested_similarity` (the nearest enrolled identity, e.g. "looks like
Rohan 0.62"). View a saved crop:

```bash
sudo docker exec third-eye-minio-1 mc cat local/thirdeye-evidence/<crop_path>  # or browse http://localhost:9001
```

**(a) Label a NEW person** — create + enroll:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/identities/candidates/<id>/approve" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"display_name": "Guest One", "role": "visitor"}' | python3 -m json.tool
```

**(b) MERGE a re-sighting into an existing person** — when the suggestion
says it's someone already enrolled, add this candidate's embedding to their
gallery (improves recognition) instead of duplicating:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/identities/candidates/<id>/approve" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"person_id": "<existing-person-uuid>"}' | python3 -m json.tool
```

**Pass:** (a) creates the person + gallery rows (audit `CANDIDATE_APPROVED`)
and the live feed names them within one re-check cycle; (b) adds an embedding
to the existing person's gallery (audit `CANDIDATE_MERGED`), no duplicate
person; rejecting stores `rejected` and never touches `persons`; saved face
crops are viewable in MinIO; distinct unknowns are separate candidates while
the same unknown seen twice is merged.

---

## Cleanup (optional)

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
DELETE FROM alerts; DELETE FROM zone_presence; DELETE FROM system_events;
DELETE FROM object_counts; DELETE FROM enrollment_candidates;
DELETE FROM persons WHERE display_name = 'Guest One';"
```

---

## Sprint 3 Pass Criteria

- [ ] Object/tracker/zone-event unit tests pass
- [ ] YOLO26 closed-set live (3a): person + multiple COCO classes, stable
      IDs, body track survives full rotation, flicker reduced vs yolov9c
- [ ] Open-vocab (3b): domain objects (almirah, iPad, ...) named correctly
      via --vocab
- [ ] (Optional) Fine-tuned model (3c) runs in run_objects
- [ ] OBJECT_ADDED / OBJECT_REMOVED fire exactly once per appearance,
      with removal grace
- [ ] PERSON_ENTERED / PERSON_EXITED persisted with name + identity_state
- [ ] object_counts rows accumulate per (zone, class)
- [ ] Events API lists them; 401 without token
- [ ] Restricted-zone alert fires once per cooldown; ack/resolve works
      and is audit-logged
- [ ] Webhook AND WebSocket each deliver the alert ≤ 1 s; delivery failure
      never stalls the pipeline
- [ ] Provisional identity_state during at-risk windows; IDENTITY_CORRECTED
      is append-only
- [ ] Unknown person → pending candidate → approve → recognized live

## Expected misses — do NOT count as failures

See the full **"DEFERRED — accept for now"** list in the READ FIRST section
at the top of this guide. In short: small-item/inventory detection,
domain-object misclassification in closed-set, the statue-as-person,
tiny-critical-asset (pendrive) protection, suspicious-behavior/anomaly
detection, theft correlation, cross-camera identity, and lower fps on the
PCIe x1 link are all expected and handled by later sprints/efforts — not
failures of this foundation sprint.
