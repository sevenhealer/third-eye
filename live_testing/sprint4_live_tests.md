# Sprint 4 Live Tests — MVP Completion

Scope (AGILE_PLAN, Milestone M-03): Redis current-state fully populated
(E06-S01), "Who is in Room A?" NL query (E07-S01), live camera grid +
zone overlays + enrollment-candidate review in the operator UI (E10-S03),
camera disconnect watchdog and reconnect (E01-S04), admin enrollment
approval flow (E02-S07, carried from Sprint 3).

Prerequisites: Sprint 3 fully passed (objects/events/tracking/candidates
live). At least one identity enrolled. Infra containers healthy
(`docker compose ps` — postgres, redis, minio at minimum).

> **BUILD STATUS: all five components are built and backend-verified by
> me** (API responses, DB rows, audit log, VRAM). What still needs **your
> own eyes** before this sprint is actually done — not just wired — is
> called out explicitly in each STEP below. Self-administered checks
> (curl, headless-Chrome screenshots, synthetic DB rows) confirm the
> wiring is correct; they are not a substitute for watching the real
> thing, per the project's standing rule that a vision/ops system can
> only be validated by a human looking at it.

---

## READ FIRST — what to expect, what can go wrong, what to accept

### ✅ What to EXPECT to work (this is the pass bar)

- **Redis reflects live pipeline state**: while `run_live.py
  --persist-events` is running, `zone:{zone}:occupants` lists whoever is
  currently tracked in that zone, updated within roughly a frame's worth
  of latency on entry/exit.
- **"Who is in <zone>?"** (`POST /api/v1/queries/nl`) answers with the
  zone's *display name* and real enrolled names (not raw UUIDs), in
  under 500ms.
- **Dashboard at `/dashboard`** logs in, shows live camera snapshots
  (refreshing every 2s), a zones/occupancy table, a Pending Enrollments
  panel with real face-crop photos, and a live alert feed over
  WebSocket.
- **Approving/rejecting a candidate from the dashboard** actually changes
  its status in Postgres and writes an audit log row — not just a UI
  state change.
- **Killing the camera connection** triggers exponential-backoff
  reconnect attempts (2s, 4s, 8s, ... capped at 30s) and the feed resumes
  once the camera/network is back, instead of the pipeline dying
  silently.

### ⚠ What can GO WRONG (real bugs — report these with console output)

- **Zone query returns a UUID instead of a name, or "Room A" instead of
  your real zone** → `_extract_zone`/`_resolve_person_names` regressed;
  this was the actual Sprint-4 bug (the planner only recognized the
  AGILE_PLAN's example zone names, not this deployment's real ones).
- **Dashboard images never load / spinner forever** → check the browser
  console; candidate photos are fetched as authenticated `blob:` URLs
  (an `<img src>` can't carry a Bearer token), so a 401 here usually
  means the token expired (15 min) — re-login.
- **Camera never recovers after a real disconnect** → confirms
  `reopen_with_backoff` regressed; check `frame_producer_camera_lost_reconnecting`
  appears in the log and is followed by `camera_opened`, not a silent
  stop.
- **Candidate approve/reject button does nothing visible** → open
  the browser console for the fetch error; the route requires `admin`
  role — a `readonly`/`operator` login will 403.

### 🟡 DEFERRED — accept for now (NOT failures; later sprints)

- **NL query intent classification is keyword/regex-based**, not an LLM
  or trained classifier. It only understands a handful of question
  shapes (current-presence, object counts). Real NL understanding is
  **Sprint 9** (E07, full NL Query Engine).
- **Anti-spoofing is not exercised by this live test or by `run_live.py`
  at all** — this was already a known, intentional deferral from Sprint
  2 (see `live_testing/sprint2_live_tests.md`: "the live printed-photo
  rejection test runs in the full pipeline... exercised in the Sprint 5
  live tests"), not a Sprint-4 regression. The ensemble itself is real
  and unit-tested (`tests/unit/test_antispoofing.py`); it is simply not
  wired into the fast demo script. Full CDCN++ ensemble + live
  printed-photo rejection is **Sprint 5**.
- **True multi-camera test uses one physical camera twice** — only one
  RTSP camera exists on this network right now. See STEP 4 for what that
  does and doesn't prove.

---

## STEP 0 — Infrastructure

```bash
docker compose up -d postgres redis minio
docker compose ps
```

**Expected:** all three `healthy`.

---

## STEP 1 — Unit tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

**Expected:** `259 passed` (covers `test_query_planner.py`,
`test_frame_producer.py`'s new reconnect tests, and everything from
prior sprints — regressions here block the sprint).

---

## STEP 2 — Redis current-state (E06-S01)

**Library:** `redis.asyncio` (the redis-py project's native asyncio
client) via `src/memory/short_term.py`'s `ShortTermMemory` wrapper —
chosen over re-deriving "who's where right now" from Postgres on every
query because zone occupancy changes every frame and a KV store with
TTLs is the right tool for *ephemeral* state that should auto-expire if
the pipeline stops updating it (a crashed pipeline shouldn't leave stale
"still in the room" data forever). Durable history (for audit/analytics)
still goes to Postgres/Timescale separately — Redis is deliberately
short-term only.

Keys written by the live pipeline, with their TTL and *why* that TTL:
- `zone:{zone_id}:occupants` / `:count` — **60s TTL**. Refreshed every
  frame on entry/exit; if the pipeline dies, occupancy data goes stale
  and disappears within a minute rather than reporting people who left
  an hour ago.
- `identity:{person_id}:location/action/track_id` — **600s TTL**, since
  per-identity lookups are queried less frequently than zone occupancy.
- `camera:{camera_id}:status` — **120s TTL**, refreshed every 150 frames
  (~10s at 15fps) by the running script — long enough to survive the
  refresh gap, short enough that a truly-dead camera reports `offline`
  within 2 minutes instead of forever showing stale `online`.
- `name:{lower_name}:person_id` — no TTL churn concern, written once per
  name at startup for the NL query's name-resolution path.

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --recognize \
  --camera-id cam0 --zone-id bedroom --persist-events --gpu 0
```

While it's running, in another terminal:

```bash
.venv/bin/python -c "
import asyncio
from src.memory.short_term import get_short_term_memory
async def main():
    stm = get_short_term_memory()
    print('occupants:', await stm.get_zone_occupants('bedroom'))
asyncio.run(main())
"
```

**Expected:** your enrolled `person_id` (a UUID) appears in `occupants`
while you're in frame, and the list empties within ~60s of leaving
(`PERSON_EXITED` + TTL). This raw store is keyed by `person_id`, not
name — name resolution happens one layer up, at the NL query (STEP 3)
and dashboard (STEP 4). Don't expect a human-readable name straight out
of this call.

**👀 Needs your eyes:** walk in and out of the zone while watching the
command above re-run — confirm the occupant list updates promptly and
doesn't lag noticeably behind your actual movement.

**Confirmed (2026-06-23):** with the real camera, `occupants` returned
`['3ec97f0b-f0ce-4937-8ee2-d7655443f24b']`, verified against `persons`
to be exactly your enrolled record (`display_name = Rohan`). Doc text
above corrected — it previously said "name(s) appear," which was wrong;
this layer stores `person_id`, by design.

Exit timing is also faster than "TTL" implies: `ZonePresenceMonitor`
(`src/event_detection/zone_presence.py`) fires `PERSON_EXITED` after
`exit_grace_frames=45` consecutive missed frames (~3s at 15fps, a
debounce against brief occlusion, not the 60s key TTL), and
`run_live.py` actively calls `remove_zone_occupant()` right then —
it does not wait for the key to expire. The 60s TTL only matters if the
whole pipeline crashes before it can clean up. Live-confirmed: walked
out, log showed `PERSON_EXITED Rohan [verified] zone=bedroom track=1`,
and `occupants` was already `[]` well inside the 60s window.

---

## STEP 3 — "Who is in Room A?" NL query (E07-S01)

**Why keyword/regex, not an LLM call:** this is explicitly a Phase-1 MVP
placeholder (`src/nlq/query_planner.py`) — a small set of question
shapes (current-presence, object counts) matched by intent keywords,
with zone names resolved by checking your real `zones` table first and
falling back to a regex for the AGILE_PLAN's example phrasing
(`"room a"`, `"corridor"`, ...). This keeps the MVP's <500ms latency
target trivial to hit (no model call) and defers real natural-language
understanding to **Sprint 9**, where it belongs.

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=<your-password>" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:8000/api/v1/queries/nl \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query": "Who is currently in the bedroom?"}' | python3 -m json.tool
```

**Expected:** `answer` names your real enrolled identity (or "No one"),
references your real zone's display name, and the response lands well
under 500ms (`response_time_ms` in the payload).

**Confirmed (2026-06-22):** zone-name and person-name resolution against
this deployment's real `zones`/`persons` tables, not just the AGILE_PLAN's
example names — fixed two real bugs (zone extraction only recognized
"Room A"-style examples; occupants were returned as raw UUIDs) and
re-verified at 36–241ms response time against the live system.

---

## STEP 4 — Dashboard: camera grid, zones, candidates, alerts (E10-S03)

**Why a single static HTML file, no build tooling:** this is an internal
operator tool, not a customer-facing product — React/Vite/etc. would
add a build step and a node_modules tree for a page that's a login form,
three polling loops, and a WebSocket. FastAPI's `StaticFiles` serves it
directly at `/dashboard`; snapshots are served from `/stream` the same
way. Cameras and zones poll every 2s (the same cadence the dashboard's
own underlying data refreshes at — Redis zone-occupancy TTL is 60s, but
the *displayed* state should feel live); candidates poll every 4s since
new unknown people don't appear every couple seconds and there's no
reason to hit the DB-join-heavy `/candidates` endpoint as often. Alerts
use a WebSocket instead of polling because they're rare and
time-sensitive — polling would either lag or waste requests.

**Why candidate photos load via a backend proxy, not a direct MinIO
URL:** MinIO's S3 API is configured with the internal Docker hostname
(`http://minio:9000`), which the API container can resolve but a
browser on your LAN cannot. A presigned URL would embed that
unreachable hostname. `GET /api/v1/identities/candidates/{id}/crop/{i}`
instead streams the bytes through the already-reachable API server.

```bash
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://<box-ip>:8000/dashboard/` in a real browser, log in as
admin.

**👀 Needs your eyes — this is the actual pass bar for this step:**
- Do the camera tiles actually show a current, non-stale frame?
- Pending Enrollments: do the face-crop photos actually look like
  recognizable people (or whatever was in frame)? Does the
  "looks like X (similarity)" hint make sense for a re-sighting of
  someone enrolled, and stay absent/low for a true stranger?
- Try Approve / Merge / Reject on a real candidate from your own queue
  and confirm the card disappears and the right thing happened (new
  person enrolled, or merged into an existing one, or just dropped).
- Is the zones table and alert feed something you'd actually trust as
  an operator, or is anything confusing/wrong on first look?

**Backend-verified by me already (wiring only, not a substitute for the
above):** crop proxy returns real, valid JPEGs for actual stored
candidates; clicking Reject in the rendered page (driven via headless
Chrome) flips status in Postgres and writes a hash-chained audit log
row; approve/merge routes return the right shape. None of this confirms
the dashboard is actually *usable* — only that it's not lying about its
data.

---

## STEP 5 — Camera disconnect watchdog and reconnect (E01-S04)

**What was actually broken:** `CameraReader.reopen_with_backoff()`
(exponential backoff: starts at 2.0s, doubles each attempt, caps at
30.0s, gives up after `reconnect_max_seconds` — default 60s) existed
and was unit-tested in isolation, but **no caller anywhere in the
codebase ever invoked it** — a real camera drop just silently stopped
the pipeline. Fixed by wiring it into both `FrameProducer._run_sequential`
(file/simple sources) and `_grab_loop` (the `drop_stale=True` path
`run_live.py`/`run_objects.py` actually use for live cameras).

**Why exponential backoff at all** (vs. retrying instantly or once):
a camera that's truly gone (powered off, unplugged) shouldn't be hit
with a tight retry loop forever; a camera mid-reboot or a flaky Wi-Fi
link usually recovers within seconds, so starting fast (2s) and backing
off (up to 30s) balances "recover quickly from a blip" against "don't
spam a dead device."

To actually sever a live connection without physical camera access, a
local TCP byte-forwarding proxy stood in for the network link
(`rtsp_transport: "tcp"` is used end-to-end, so killing the proxy
process fully severs the stream — closer to a real disconnect than
killing the pipeline process itself):

```bash
# terminal A — proxy between you and the real camera
.venv/bin/python -c "
import asyncio, sys
listen_port, target_host, target_port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
async def pipe(r, w):
    try:
        while (data := await r.read(4096)):
            w.write(data); await w.drain()
    finally:
        w.close()
async def handle(reader, writer):
    tr, tw = await asyncio.open_connection(target_host, target_port)
    await asyncio.gather(pipe(reader, tw), pipe(tr, writer))
async def main():
    srv = await asyncio.start_server(handle, '127.0.0.1', listen_port)
    async with srv: await srv.serve_forever()
asyncio.run(main())
" 9554 10.7.7.152 554

# terminal B — point run_live.py at the proxy instead of the camera directly
.venv/bin/python scripts/run_live.py \
  --source "rtsp://sevenhealer:seven2026@127.0.0.1:9554/stream1" \
  --recognize --camera-id cam0 --zone-id bedroom --gpu 0
```

Kill terminal A's proxy mid-stream, watch terminal B's log, then restart
the proxy.

**Expected:** `frame_producer_camera_lost_reconnecting`, then repeated
`camera_reconnect_attempt` lines with growing `backoff_s` (2.0 → 4.0 →
8.0 → ...), then `camera_opened` and frames resuming the moment the
proxy comes back — no crash, no manual restart needed.

**Confirmed (2026-06-22):** live-verified exactly this way — detected the
drop, backed off 2s→4s→8s→16s, reconnected, resumed tracking correctly.

---

## STEP 6 — Two cameras at 10 FPS, VRAM < 12 GB (MVP acceptance criterion)

**Hardware note:** this box currently enumerates a single GPU
(`nvidia-smi -L` → RTX 3090, 24GB) — earlier sprints' multi-GPU notes
(3070/3060Ti) don't apply to whatever state the box is in right now. If
you've changed GPU hardware since Sprint 3, that's worth double-checking
isn't accidental.

**What I could and couldn't test:** the only RTSP camera on the network
(`10.7.7.152`) was unreachable during this session (`No route to host`)
— likely powered off. Rather than block on that, I substituted a local
video file (a real face snapshot looped into a 30-minute clip via
ffmpeg) as a stand-in source for two concurrent `run_live.py --recognize`
processes on the same GPU. This is a legitimate **backend/VRAM check**
but **not a real camera-throughput test** — a local file decodes far
faster than a real RTSP stream paces frames, so any "frames/sec"
number from this substitute is meaningless and was discarded.

What IS a valid result from that test, because it doesn't depend on the
input source: **combined VRAM usage stayed flat at ~2.8 GB** for two
simultaneous recognition pipelines (well under the 12GB target), and
**real per-frame inference time stayed at 33–39ms** under two-process
GPU contention — comfortably fast enough for two cameras at 10fps each
(20fps combined demand vs. ~25-30fps/process theoretical capacity even
while sharing the GPU).

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --fps 10 --show \
  --recognize --camera-id cam0 --zone-id bedroom --persist-events --gpu 0 &
.venv/bin/python scripts/run_live.py --source <rtsp-url> --fps 10 --show \
  --recognize --camera-id cam1 --zone-id corridor --gpu 0 &
watch -n1 nvidia-smi
```

**Confirmed live (2026-06-22), with the real camera, two windows
(`--show`) on the box's own display:** both feeds smooth simultaneously
at 10fps each, no visible contention; VRAM steady at **2.84 GB**; both
processes correctly recognized the enrolled user thousands of times
each. The other tracked object (static, low-similarity `unknown`) is the
Krishna-statue false positive already documented in Sprint 3 — a
non-human object the closed-set detector still boxes as a face-like
region; the identity layer correctly keeps it `unknown`, and true
non-human rejection is anti-spoofing's job in Sprint 5. Not a new bug.

**Image quality follow-up (2026-06-23):** during the 2026-06-22 run
above, the user reported the live image quality as "very bad." I have
no visual access, so I checked the stream objectively instead of
guessing: `ffprobe` on `stream1` (the profile we use) showed
2304x1296@15fps at only **~1.4-1.7 Mbps** — low for that resolution —
plus repeated "Non-monotonous DTS" warnings (timestamp jitter from the
camera itself). `stream2` is not a fix (it's 1280x720, lower-res, not
higher-quality). Re-checked visually with the user on a fresh
single-stream `--show` run today: looked fine this time. Likely
explanation, not yet confirmed: the bitrate/quality problem may be
**load- or motion-dependent** (worse under the two-concurrent-process
condition in the original test, or during higher-motion moments) rather
than constantly bad. Not re-tested under concurrent 2-camera load
specifically for quality (only for smoothness/VRAM, which passed). If
quality complaints recur, check `nvidia-smi` GPU decode load and
RTSP packet loss during the bad period, not just the encoder bitrate.

---

## Cleanup (optional)

```bash
sudo docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c "
DELETE FROM enrollment_candidates WHERE camera_id LIKE 'loadtest%';"
```

(Nothing else from this sprint's testing needs cleanup — the candidate
review test used a synthetic row that was deleted immediately after
verification, and its one audit log entry is intentionally
undeletable — `audit_log` has a Postgres `no_delete_audit` rule, by
design, since the whole point of an audit log is that it can't be
quietly edited after the fact.)

---

## Sprint 4 / MVP Pass Criteria

- [x] Unit tests pass (259/259)
- [x] "Who is currently in Room A?" returns correct answer in < 500ms —
      confirmed 36–241ms against real zones/identities
- [ ] Unknown person triggers enrollment candidate **visible in the
      operator UI** within 30 seconds — candidate creation/capture was
      Sprint 3; the UI surface is built this sprint — **needs your visual
      confirmation (STEP 4)**
- [ ] Printed photo rejected by anti-spoofing in 100% of basic test
      cases — **intentionally deferred to Sprint 5**, not a Sprint 4 gap
      (see sprint2 live-test doc)
- [x] `PERSON_ENTERED` fires within 2s of zone entry — confirmed in
      Sprint 3
- [x] Alert delivered via webhook within 5s — confirmed ~1s in Sprint 3
- [x] 2 cameras at 10 FPS with VRAM < 12 GB — confirmed live with the
      real camera, both feeds smooth, VRAM steady at 2.84GB
- [x] Audit log records all enrollments and operator actions —
      hash-chained, confirmed via the candidate-reject test
