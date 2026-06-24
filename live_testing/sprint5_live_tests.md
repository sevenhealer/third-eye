# Sprint 5 Live Tests — Anti-Spoofing (software-only)

Scope: the liveness/anti-spoofing layer that was deliberately deferred
from Sprint 2 ("the live printed-photo rejection test ... exercised in
the Sprint 5 live tests"). This sprint is the **software-only** slice the
user chose — no extra IR/depth hardware — built from three pieces:

1. **E08-S07 — model signing, verify-on-startup, fail-closed.** Every
   weight file is SHA-256 signed into a manifest; the API verifies them at
   every boot and refuses to start on a mismatch (tamper/corruption).
2. **Weight-free temporal liveness gate in the live path.** A per-track
   micro-movement check wired into `run_live.py` that flags a static
   presentation (printed photo / phone screen / statue) without needing
   any trained texture weights.
3. **E02-S10 — CDCN++ fine-tuning pipeline.** `collect → train → load`
   closed end to end, so the ensemble can be upgraded from temporal-only
   to texture + temporal once you've collected deployment data.

Prerequisites: Sprint 1–4 passed. Start from a clean slate with
**`make fresh`** (STEP 0). One physical RTSP camera + a printed photo (or
a phone showing a photo) of an **enrolled** person is all the extra kit
this sprint needs.

> **BUILD STATUS: all three pieces are built and backend/local-verified
> by me** — unit tests, a tamper round-trip, the temporal gate's
> static-vs-live score separation on synthetic frames, and the full
> collect→train→load→score loop on synthetic crops. What still needs
> **your own eyes / a real camera** is the actual spoof behaviour: does a
> real printed photo of an enrolled person score as static and (under
> enforcement) fail to be recognized, while you standing there normally
> stays live? That can only be confirmed in front of the camera, per the
> standing rule that a vision system is validated by a human watching it.

---

## READ FIRST — what to expect, what can go wrong, what to accept

### ✅ What to EXPECT to work (this is the pass bar)

- **Startup verification**: after `make sign-models` (run automatically by
  `make bootstrap`/`make run`), the API logs
  `startup_model_verification_complete` on boot. Corrupt a signed weight
  file and the next boot raises `ModelChecksumError` and refuses to start —
  it never silently runs on tampered weights.
- **Temporal liveness, observe mode (default)**: with `run_live.py`
  running, hold a **printed photo / phone screen** of an enrolled person
  in front of the camera. Within ~1 second its box turns **red** and is
  tagged `STATIC?`, and the console line for that track shows
  `STATIC? live=<low score>`. Your own live face stays green and is not
  tagged. Nothing is blocked yet in this mode — it's showing you the
  decision so you can judge the separation before enforcing.
- **Temporal liveness, enforce mode (`--enforce-liveness`)**: the same
  static photo is now **forced to `unknown`** — it is never recognized as
  the enrolled person, never written as that person entering, and never
  captured as an enrollment candidate. The box reads `SPOOF — BLOCKED`.
- **Data pipeline**: `collect_antispoofing_data.py` saves labelled crops;
  `train_antispoofing.py` trains a CDCN++ checkpoint from them and prints a
  rising `val_acc`; the checkpoint loads back through
  `CDCNWrapper.load_from_checkpoint` without error.

### ⚠ What can GO WRONG (real bugs — report these with console output)

- **Your real, moving face gets tagged `STATIC?`** → the temporal metric
  scores the *variance* of frame-to-frame face-crop luminance, so holding
  unusually still for ~1s can trip it. A brief flag while you freeze is
  expected; a persistent flag while you move/blink/talk normally is a real
  problem — report the `live=` score from the console. This exact
  sensitivity is **why enforcement is opt-in** and observe mode is the
  default (see below).
- **A printed photo scores as live (green, no tag)** → the core failure
  this sprint guards against. Note the `live=` score and the lighting /
  how steadily you held it — a photo waved around by hand has real motion
  and *will* (correctly, by this metric's logic) read as less static. The
  weight-free gate catches a **still** presentation; defeating a
  deliberately-jiggled photo is what the CDCN++ texture model (piece 3) is
  for once trained.
- **API refuses to start after a legitimate weights update** → if you
  re-download or replace a weight file, its hash no longer matches the
  manifest. That's the fail-closed working as designed — re-sign with
  `make sign-models` (or `python scripts/sign_models.py`) and restart.
  Report it only if re-signing doesn't clear it.
- **`train_antispoofing.py` reports a suspiciously perfect `val_acc=1.0`
  on a tiny dataset** → not a bug, but not a real result either: a few
  dozen crops overfit instantly. It means the pipeline runs, not that
  spoofing is solved. See the DEFERRED note.

### 🟡 DEFERRED — accept for now (NOT failures; later sprints)

- **CDCN++ texture model is not wired into `run_live.py`.** The live gate
  is temporal-only (weight-free) by design for this software-only slice.
  The trainer + loader exist and round-trip, but actually loading a
  trained CDCN++ checkpoint into the live ensemble and running the full
  hard-AND (MiniFAS + CDCN++ + temporal) is a follow-up that depends on
  you first collecting a real, varied live/spoof dataset. Treat piece 3 as
  the **scaffold** that makes that upgrade a config change, not new code.
- **No MiniFASNet-V2 weights either.** The ensemble's MiniFAS wrapper
  returns a fail-safe 0.0 (deny) with no weights — which is why the
  *ensemble* isn't the live gate yet (it would deny everything). The
  standalone temporal checker is.
- **In-house dataset will overfit.** CDCN++ needs thousands of varied
  crops (lighting, attack media, distances) to generalize. The trainer is
  deliberately a from-scratch scaffold; warm-starting from a public
  CASIA-SURF/OULU-NPU checkpoint is the realistic production path and is
  out of scope here.
- **Enforcement defaults OFF.** Until you've watched the real-vs-spoof
  score separation on your own camera and are comfortable with the false-
  reject risk on a genuinely-still person, `--enforce-liveness` stays opt-
  in. Turning it on by default is a deliberate decision to make *after*
  this live test, not before.

---

## STEP 0 — One-command clean slate

```bash
make fresh
```

Chains `stop → infra → reset → frontend-build → serve`. New this sprint:
`make bootstrap` (which `fresh`/`run` call) now also runs
**`make sign-models`** on first setup — it SHA-256-signs every present
weight file into `models/manifest.json`, which the API then verifies on
every boot. The manifest is per-clone (machine-specific paths) and
gitignored, regenerated on bootstrap.

**Login:** `admin` / `admin`.  **Dashboard:** http://localhost:8000/settings/

**Expected:** server boots to `third_eye_ready`. In the startup log you
should see `startup_model_verification_complete verified=N`, confirming
verify-on-startup ran and passed. `N` is however many weight files were
present when `sign-models` ran — 8 on this box (5 buffalo_l ONNX + 3 YOLO
`.pt`), but a fresh clone signs only the face models at first (the YOLO
`.pt` download lazily on the first object-detection run). Re-run
`make sign-models` after that to bring the YOLO weights under
verification too.

---

## STEP 1 — Unit tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

**Expected:** `274 passed` (272 from Sprint 4 plus 2 new model-registry
tests: repo-relative default manifest path, and `startup_verify()`
defaulting to it). The existing `test_model_registry.py` tamper-detection
and `test_antispoofing.py` ensemble tests are included.

---

## STEP 2 — Model signing: verify-on-startup, fail-closed (E08-S07)

**Why:** a security product must not run on a weight file that's been
swapped or corrupted. `scripts/sign_models.py` hashes each weight into a
manifest; `startup_verify()` (called from the API lifespan) re-hashes them
all at boot and raises `ModelChecksumError` on any mismatch instead of
continuing. The manifest path is resolved repo-relatively
(`DEFAULT_MANIFEST_PATH`) so it works natively and in the container — the
old hardcoded `/app/models/manifest.json` setting silently no-op'd
natively and has been removed.

Confirm the manifest exists and verification passes:

```bash
make sign-models          # idempotent; re-signs whatever is present
.venv/bin/python -c "from src.core.model_registry import startup_verify; startup_verify(); print('verify OK')"
```

**Tamper test** (proves fail-closed — do this on a throwaway copy, then
re-sign):

```bash
# append a byte to a signed weight, then re-verify → must raise
cp models/weights/models/buffalo_l/genderage.onnx /tmp/genderage.bak
printf '\x00' >> models/weights/models/buffalo_l/genderage.onnx
.venv/bin/python -c "from src.core.model_registry import startup_verify; startup_verify()" ; echo "exit=$?"
# restore + re-sign
mv /tmp/genderage.bak models/weights/models/buffalo_l/genderage.onnx
make sign-models
```

**Expected:** the tampered run prints a `ModelChecksumError` (SHA-256
mismatch) and a non-zero exit; after restore + `make sign-models`,
verification passes again. A full API boot against a tampered file would
likewise fail to reach `third_eye_ready`.

**👀 Needs your eyes:** nothing visual — this is a backend gate. The
tamper test above is self-administered and sufficient.

---

## STEP 3 — Temporal liveness gate, OBSERVE mode (default)

**What it is:** `run_live.py` now feeds each tracked face crop to a
weight-free `TemporalConsistencyChecker`. A live face never holds
perfectly still (blinks, micro-sway); a printed photo / static screen /
statue does, scoring near-zero variance. In the default (observe) mode the
gate **scores, draws, and logs** but does not block — exactly so you can
judge the real-vs-spoof separation on your own camera before enforcing.

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --recognize \
  --camera-id cam0 --zone-id bedroom --gpu 0 --show
```

The startup banner should read:
`anti-spoofing   : observe-only (scored + drawn, not blocking; use --enforce-liveness)`

Now, in front of the camera:
1. Stand there normally (move, blink, talk). Your box stays **green**.
2. Hold up a **printed photo or a phone screen** showing a face, as still
   as you can.

**Expected:** within ~1s the photo's box turns **red** with a `STATIC?`
tag, and its console line shows `STATIC? live=<score>` with a low score
(below the 0.1 live threshold). Your real face stays green/untagged.

**👀 Needs your eyes — this is the actual pass bar:**
- Does a still photo reliably go red, and your live face reliably stay
  green? Note the `live=` scores for each — that separation is what
  decides whether enforcement (STEP 4) is safe to turn on.
- How still do *you* have to hold to get falsely flagged? A brief flag
  while frozen is expected; report it if normal movement trips it.

**Local-verified by me (not a substitute for the above):** on synthetic
frames, a perfectly-static crop scored `0.0000` (→ STATIC) and an
irregularly-moving crop scored `0.1237` (→ LIVE), confirming the metric
separates the two and the threshold sits between them. Real-camera scores
are what your test gathers.

---

## STEP 4 — Temporal liveness gate, ENFORCE mode

Only after STEP 3 shows a clean separation, re-run with enforcement:

```bash
.venv/bin/python scripts/run_live.py --source <rtsp-url> --recognize \
  --camera-id cam0 --zone-id bedroom --gpu 0 --show \
  --enforce-liveness --persist-events
```

Banner should read:
`anti-spoofing   : ENFORCED — static presentations blocked`

Hold up the **printed photo of an enrolled person** (the attack: trying to
impersonate someone the system knows).

**Expected:**
- The photo's box reads **`SPOOF — BLOCKED`**.
- It is **never labelled with the enrolled name** — forced to `unknown`.
- No `PERSON_ENTERED` for that enrolled person is written, and no
  enrollment candidate is captured from the photo.
- Your own live face is still recognized and behaves exactly as in
  Sprint 4 (name, presence events, etc.).

**👀 Needs your eyes — the core sprint demo:** a printed photo of an
enrolled person must NOT get them logged as present. Confirm the name does
not appear on the photo's box and that no entry event for them shows in
the console / zone log while only the photo is in frame.

`--bypass-antispoofing` disables liveness entirely (dev mode, no scoring,
no annotation) if you need the old behaviour.

---

## STEP 5 — Anti-spoofing data pipeline: collect → train → load (E02-S10)

**What it is:** the scaffold to upgrade liveness from temporal-only to a
trained CDCN++ texture model. Not required for STEP 3/4 to pass — this is
the path to stronger anti-spoofing later.

Collect labelled crops (a few hundred each; vary lighting / distance /
attack media for anything that'll generalize):

```bash
# genuine faces — you, live, moving around
.venv/bin/python scripts/collect_antispoofing_data.py --label live  --auto --show --target 200
# attacks — printed photos, phone/tablet screens, masks
.venv/bin/python scripts/collect_antispoofing_data.py --label spoof --auto --show --target 200
```

Train CDCN++ on them:

```bash
.venv/bin/python scripts/train_antispoofing.py --epochs 30
# optional: --sign also adds the checkpoint to the startup manifest
```

**Expected:** the trainer reports `train N / val M crops`, prints a
per-epoch `train_loss` / `val_acc`, saves the best-by-val checkpoint to
`models/weights/cdcn_pp.pt`, and prints the one-liner to load it
(`CDCNWrapper.load_from_checkpoint(...)`). The checkpoint loads back
without an architecture error.

**👀 Needs your eyes:** mostly that the tooling runs cleanly on real
captured data. **Do not read a high `val_acc` on a small set as "spoofing
solved"** — it's confirming the pipeline, not the model's real-world
generalization (see DEFERRED).

**Local-verified by me:** the full `collect → train → save →
load_from_checkpoint → score` loop runs end to end on synthetic crops;
the saved checkpoint round-trips through the ensemble wrapper and returns
a score in [0, 1].

---

## Sprint 5 Pass Criteria

- [x] Unit tests pass (274/274) — confirmed locally
- [x] Model weights signed; API verifies on startup and fails closed on a
      tampered file — confirmed locally (tamper round-trip)
- [ ] **Printed photo of an enrolled person scores as static (observe) and
      is NOT recognized as that person (enforce)** — backend/synthetic
      verified; **needs your real-camera confirmation (STEP 3–4)**
- [ ] A genuinely-live, normally-moving face is NOT falsely blocked —
      **needs your real-camera confirmation (STEP 3)**
- [x] Anti-spoofing data pipeline collect → train → load round-trips —
      confirmed locally on synthetic crops; **real-data training is yours
      to run (STEP 5)**

## Deferred to a later sprint (explicitly, not gaps)

- Wiring a trained CDCN++ checkpoint into the live ensemble (full
  MiniFAS + CDCN++ + temporal hard-AND in `run_live.py`).
- MiniFASNet-V2 weights / a public-checkpoint warm start for CDCN++.
- Any depth/IR hardware-based liveness (the user chose software-only).
