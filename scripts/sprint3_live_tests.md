# Sprint 3 Live Tests — Objects + Events + Tracking

Scope (AGILE_PLAN): YOLOv9-C object detection (E03-S01), zone-level object
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

Hardware note (Linux 3090 on PCIe x1): YOLOv9-C at 1280 will be slower
than the face pipeline. If fps < 8, drop input size to 960 or 640 —
detection quality for room-scale objects is barely affected.

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

---

## STEP 1 — Unit tests for the object/event layer

```bash
.venv/bin/python -m pytest tests/unit/test_object_detector.py \
  tests/unit/test_tracker.py tests/unit/test_zone_events.py -q
```

**Expected:** all pass (tracker suite includes the Sprint-2-hardened
appearance/bank tests).

---

## STEP 2 — YOLO weights

```bash
.venv/bin/pip install ultralytics
.venv/bin/python -c "
from ultralytics import YOLO
m = YOLO('yolov9c.pt')          # auto-downloads to ./yolov9c.pt (~50 MB)
print('classes:', len(m.names), '| person id:', [k for k,v in m.names.items() if v=='person'])
"
```

**Expected:** `classes: 80 | person id: [0]` (COCO classes).

---

## STEP 3 — Live object detection + tracking *(needs build #1)*

```bash
.venv/bin/python scripts/run_objects.py --source <rtsp-url> --show --fps 15 \
  --camera-id cam0 --zone-id bedroom
```

**Expected:**
- Banner: GPU (CUDA), model yolov9c, conf 0.45
- Same HUD/overlay conventions as the face runner (fps, infer ms, color
  states, `ID:n` labels) — but boxes now cover persons AND objects
  (chair, bed, bottle, laptop, cell phone, ...)
- You get ONE stable person track while moving around (ByteTrack on
  full-body boxes is much more forgiving than face boxes — full rotation
  should NOT switch your ID here)
- Static objects (bed, chair) hold stable IDs; the Krishna statue may be
  detected as `person` by COCO YOLO — that is a known YOLO limitation,
  note it and move on (face recognition is what tells persons apart)
- Console prints `OBJECT_ADDED` when a new object appears in frame
  (bring a bottle/laptop into view) and `OBJECT_REMOVED` ~5 s after you
  take it away (removal grace prevents blink-removals)

**Pass:** person + ≥3 object classes detected; your body track survives a
full rotation; ADDED/REMOVED fire exactly once per object appearance.

---

## STEP 4 — Zones + PERSON_ENTERED / PERSON_EXITED *(needs build #2)*

Seed the zones (one-time):

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
INSERT INTO zones (zone_id, display_name, zone_type) VALUES
  ('bedroom',  'Bedroom',  'general'),
  ('doorway',  'Doorway',  'restricted')
ON CONFLICT (zone_id) DO NOTHING;"
```

Run the face runner with event persistence:

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --show --fps 25 \
  --recognize --camera-id cam0 --zone-id bedroom --persist-events
```

Walk out of frame (>6 s), walk back in. Then check:

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
SELECT event_type, zone_id, payload->>'person' AS person,
       payload->>'identity_state' AS identity_state, event_time
FROM system_events ORDER BY event_time DESC LIMIT 10;"
```

**Expected:**
- `PERSON_ENTERED` when your track is confirmed in the zone,
  `PERSON_EXITED` after you leave (grace period, no blink events)
- `person` = `Rohan` with `identity_state = verified` for clean tracks
- An event fired while your track was at-risk (right after a gap return /
  crossing) carries `identity_state = provisional`
- `zone_presence` rows show your entry/exit times with `is_unknown=false`

**Pass:** one ENTERED + one EXITED per actual entry/exit, correctly named,
with identity_state present on every event.

---

## STEP 5 — Zone object counts *(needs build #3)*

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

## STEP 7 — Alert rules → alerts rows *(needs build #4)*

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

## STEP 8 — Webhook + WebSocket delivery *(needs build #5)*

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

## STEP 9 — Identity attribution under uncertainty *(needs build #2)*

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

## STEP 10 — Unknown-person enrollment candidates *(needs build #6)*

Have the unenrolled person stand in view ~10 s with their face visible.

```bash
curl -s "http://127.0.0.1:8000/api/v1/identities/candidates?status=pending" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected:** one pending candidate with crop count ≥ 5, a 512-dim mean
embedding, first/last seen timestamps. Approve it:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/identities/candidates/<id>/approve" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"display_name": "Guest One", "role": "visitor"}' | python3 -m json.tool
```

**Pass:** approval creates the person + gallery rows (audit-logged); the
live feed names them `Guest One` within one re-check cycle; rejecting a
candidate stores `rejected` and never touches `persons`. Duplicate
candidates for the same unknown face within one session are merged, not
multiplied.

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
- [ ] YOLOv9-C live: person + multiple object classes, stable IDs, body
      track survives full rotation
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

- Krishna statue detected as `person` by COCO YOLO (known model limit;
  identity layer keeps it `unknown`, anti-spoofing later rejects it)
- Object classes outside COCO's 80 (specific tools, custom items) are
  not detected — fine-tuning/forensic models come later
- Cross-camera identity continuity — Sprint 6
- Action recognition ("what are they doing") — later sprint
- fps lower than the face pipeline on the x1 link; use --det-size 640
