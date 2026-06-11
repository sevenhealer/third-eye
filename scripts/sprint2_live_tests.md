# Sprint 2 — Live Test Guide (Face Pipeline)

Covers: SCRFD face detection, AdaFace/ArcFace recognition embeddings, pgvector
gallery, enrollment workflow, identities API, enrollment audit log.

Run all commands from the repo root: `/Users/iamrohanchatterjee/Documents/Code/third-eye`

Prerequisites from Sprint 1:
- `.env` configured (Postgres `testpass123`, admin login `admin / admin`)
- On Mac: `DEVICE=cpu` in `.env`. On the Linux RTX 3090 box: `DEVICE=cuda:0`.

> Anti-spoofing note: the MiniFASNet/CDCN ensemble is covered by unit tests in
> STEP 1. The live printed-photo rejection test runs in the full Kafka pipeline,
> not in `run_live.py` — it is exercised in the Sprint 5 live tests.

---

## Linux GPU setup — CUDA version alignment (one-time)

PyPI's default torch wheels are **CUDA 13** builds, but stable
`onnxruntime-gpu` is **CUDA 12**. With mismatched runtimes the CUDA provider
fails to load (`libcublasLt.so.12: cannot open shared object file`) and all
ONNX inference silently falls back to CPU — GPU idle, CPU pegged.

Install torch from the cu128 index BEFORE the editable install (or run
`make install-gpu`, which does both):

```bash
.venv/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[dev]"
```

If a CUDA-13 torch is already installed, remove it and its nvidia wheels
first (the cu12/cu13 cuDNN wheels overlap on disk):

```bash
.venv/bin/pip uninstall -y torch torchvision torchaudio \
  $(.venv/bin/pip list | grep -i nvidia | awk '{print $1}' | tr '\n' ' ')
```

Verify both stacks see the GPU:

```bash
.venv/bin/python -c "
import torch; print('torch CUDA:', torch.cuda.is_available(), torch.version.cuda)
from src.core.gpu_manager import preload_cuda_libraries; preload_cuda_libraries()
import onnxruntime as ort
s = ort.InferenceSession('models/weights/models/buffalo_l/genderage.onnx',
                         providers=['CUDAExecutionProvider'])
print('ORT providers:', s.get_providers())
"
```

**Expected:** `torch CUDA: True 12.8` and `CUDAExecutionProvider` first in the
ORT list. While the live feed runs, `nvidia-smi -l 1` should show activity.

> Also note: only `onnxruntime-gpu` may be installed on Linux — if plain
> `onnxruntime` (CPU) sneaks in, the two clobber the same module directory and
> CUDA disappears. Fix: uninstall **both**, reinstall `onnxruntime-gpu`.

---

## STEP 0 — Start infrastructure

```bash
docker-compose up postgres redis -d
docker-compose ps
```

**Expected:** `third-eye-postgres-1` and `third-eye-redis-1` both `Up (healthy)`.

---

## STEP 1 — Unit tests for the face pipeline

```bash
.venv/bin/python -m pytest tests/unit/test_face_detector.py \
  tests/unit/test_antispoofing.py tests/unit/test_face_recognition.py \
  tests/unit/test_gallery.py tests/unit/test_enrollment.py \
  tests/unit/test_face_pipeline.py -q
```

**Expected:** `48 passed`

---

## STEP 2 — Model weights present

If not done already (one-time, ~200 MB):

```bash
.venv/bin/python scripts/download_models.py
```

Verify:

```bash
ls models/weights/models/buffalo_l/
```

**Expected files:** `det_10g.onnx` (SCRFD detector), `w600k_r50.onnx`
(recognition, 512-dim embeddings), plus `1k3d68.onnx`, `2d106det.onnx`,
`genderage.onnx`.

---

## STEP 3 — Live face detection + tracking (webcam)

```bash
.venv/bin/python scripts/run_live.py --source 0 --show
```

> First run on macOS: grant camera access to Terminal/IDE under
> System Settings → Privacy & Security → Camera.

**Expected:**
- Banner shows `platform : Darwin arm64` and `inference : CPU + CoreML (Apple Neural Engine)`
  (Linux: `GPU (CUDA)`)
- A window opens with a green box and stable `ID:1` label around your face
- Terminal prints one line per tracked frame:
  ```
  [cam0 | frame    42]  track=  1  bbox=[ 512, 203, 781, 540]  conf=0.857
  ```
- Track ID stays the same while you stay in frame; a second person gets a new ID
- Press `q` (or Ctrl+C) to stop → summary line `Stopped — N frames, 1 unique person ID(s).`

---

## STEP 4 — Enroll yourself into the gallery

```bash
.venv/bin/python scripts/enroll.py --name "Rohan" --role engineer
```

Look at the camera; it collects 10 face crops (~1–2 seconds).

**Expected:**
```
  Crop  1/10 captured.
  ...
  Crop 10/10 captured.

Computed mean embedding from 10 crops.
person_id = <uuid>

Enrolled 'Rohan' successfully!
  person_id    : <uuid>
  embedding_id : <uuid>
```

(The script creates the `persons` row itself — no manual psql insert needed.)

---

## STEP 5 — Verify gallery rows in Postgres

```bash
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c \
  "SELECT p.display_name, p.role, g.quality_score, vector_dims(g.embedding) AS dims
   FROM persons p JOIN face_gallery g USING (person_id);"
```

**Expected:** One row: `Rohan | engineer | 0.85 | 512`

---

## STEP 6 — pgvector similarity search round-trip

Search the gallery with your own stored embedding (should match ≈ 1.0, decision
`accept`) and with a random vector (decision `reject`):

```bash
.venv/bin/python - <<'EOF'
import asyncio
import numpy as np
from src.face_recognition.gallery import FaceGallery

async def main():
    gallery = FaceGallery()
    from src.core.database import get_db_session
    from sqlalchemy import text
    async with get_db_session() as s:
        row = (await s.execute(text(
            "SELECT person_id, embedding::text AS vec FROM face_gallery LIMIT 1"
        ))).fetchone()
    own = np.array(row.vec.strip("[]").split(","), dtype="float32")

    print("-- search with own embedding (expect accept, ~1.0):")
    for m in await gallery.search(own, top_k=3):
        print(f"   {m.person_id}  sim={m.similarity:.3f}  {m.decision}")

    print("-- search with random vector (expect reject, <0.35):")
    rand = np.random.default_rng(7).normal(size=512).astype("float32")
    for m in await gallery.search(rand, top_k=3):
        print(f"   {m.person_id}  sim={m.similarity:.3f}  {m.decision}")

asyncio.run(main())
EOF
```

**Expected:** own embedding → `sim=1.000 accept`; random vector → low
similarity with `reject`.

---

## STEP 7 — Identities API (CRUD + RBAC)

Start the API if it isn't running:

```bash
.venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, log in and grab a token:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -F "username=admin" -F "password=admin" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

```bash
# Create an identity via API — expect HTTP 201 with person_id
curl -s -X POST http://127.0.0.1:8000/api/v1/identities \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"display_name": "API Test Person", "role": "visitor"}' | python3 -m json.tool

# List identities — expect both "Rohan" and "API Test Person"
curl -s http://127.0.0.1:8000/api/v1/identities \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# No token — expect 401
curl -sL -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/api/v1/identities
```

Soft-delete the API test person (use its `person_id` from the create response):

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X DELETE http://127.0.0.1:8000/api/v1/identities/<person_id> \
  -H "Authorization: Bearer $TOKEN"
```

**Expected:** `HTTP 204`, and the person disappears from the default list.

---

## STEP 8 — Enrollment audit log entries

```bash
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c \
  "SELECT action_type, actor_username, current_hash IS NOT NULL AS has_hash, event_time
   FROM audit_log
   WHERE action_type IN ('IDENTITY_ENROLLED','IDENTITY_SOFT_DELETED')
   ORDER BY log_id DESC LIMIT 5;"
```

**Expected:** `IDENTITY_ENROLLED` and `IDENTITY_SOFT_DELETED` rows from STEP 7,
all with `has_hash = t`.

---

## Cleanup (optional)

```bash
# Remove the test enrollment (cascades to face_gallery via FK)
docker exec third-eye-postgres-1 psql -U thirdeye -d thirdeye -c \
  "DELETE FROM persons WHERE display_name IN ('Rohan', 'API Test Person');"
```

---

## Sprint 2 Pass Criteria

- [ ] 48 face-pipeline unit tests pass
- [ ] buffalo_l weights present (`det_10g.onnx`, `w600k_r50.onnx`)
- [ ] Live webcam: face detected with stable track ID at ~10 FPS
- [ ] `enroll.py` collects 10 crops and writes `persons` + `face_gallery` rows
- [ ] Gallery row has a 512-dim embedding
- [ ] pgvector search: own embedding → `accept` (~1.0), random vector → `reject`
- [ ] API create identity returns 201; list shows enrolled people
- [ ] Unauthenticated identities request returns 401
- [ ] Soft delete returns 204 and hides the person from the list
- [ ] Audit log records `IDENTITY_ENROLLED` / `IDENTITY_SOFT_DELETED` with hashes
