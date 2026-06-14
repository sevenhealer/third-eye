#!/usr/bin/env python3
"""
Live camera feed — face detection and recognition.

Platform support:
  macOS M1/M2/M3/M4  — CPU + CoreML (Neural Engine), ARM64 native
  Linux RTX 3090      — CUDA GPU

Install (M4 Mac):
  brew install cmake                              # needed to compile insightface
  .venv/bin/pip install cython scikit-build-core
  .venv/bin/pip install insightface onnxruntime opencv-python numpy

Install (Linux CUDA):
  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy

Prerequisites:
  python scripts/download_models.py        # one-time, ~200 MB
  docker compose up -d postgres redis      # only for gallery lookups

Usage:
  python scripts/run_live.py --source 0                          # built-in camera
  python scripts/run_live.py --source 0 --show                   # display window
  python scripts/run_live.py --source rtsp://192.168.1.x/stream  # IP camera
  python scripts/run_live.py --source 0 --bypass-antispoofing    # skip liveness (dev)
  python scripts/run_live.py --source 0 --cpu                    # force CPU only
"""
import argparse
import asyncio
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ON_MAC = platform.system() == "Darwin"
ON_APPLE_SILICON = ON_MAC and platform.machine() == "arm64"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="third-eye live feed")
    p.add_argument("--source", default="0",
                   help="Camera index (0, 1, ...) or RTSP URL")
    p.add_argument("--camera-id", default="cam0")
    p.add_argument("--zone-id", default="entrance")
    p.add_argument("--fps", type=int, default=15,
                   help="Target processing FPS (default 15; 0 = uncapped). "
                        "Match the camera rate — fewer skipped frames means "
                        "less motion between frames and steadier track IDs.")
    p.add_argument("--det-size", type=int, default=0,
                   help="Face detector input size (default: 960 on GPU, 640 "
                        "on CPU). Larger finds small/distant faces on "
                        "high-res streams at some speed cost.")
    p.add_argument("--det-thresh", type=float, default=0.6,
                   help="Confidence needed to START a track (default 0.6). "
                        "The detector itself runs at a 0.30 floor: detections "
                        "in [0.30, det-thresh) only sustain already-started "
                        "tracks through confidence dips (ByteTrack), never "
                        "create new ones. Raise to suppress low-light false "
                        "faces, lower if real faces are never picked up.")
    p.add_argument("--reid-thresh", type=float, default=0.45,
                   help="Embedding similarity needed to give a returning face "
                        "its old ID back (default 0.45). Lower toward 0.38 if "
                        "you re-enter and get a new ID (watch for "
                        "reid_revival_missed log lines); don't go below "
                        "0.35 — that's the different-person boundary.")
    p.add_argument("--min-face", type=int, default=24,
                   help="Discard face boxes smaller than this many pixels on "
                        "their short side (default 24). Filters tiny low-light "
                        "phantom detections that also embed too poorly to track.")
    p.add_argument("--show", action="store_true",
                   help="Open a cv2 window (requires opencv-python, not headless)")
    p.add_argument("--recognize", action="store_true",
                   help="Match tracked faces against the pgvector gallery and "
                        "label them by name (needs postgres up and at least "
                        "one identity enrolled via scripts/enroll.py)")
    p.add_argument("--persist-events", action="store_true",
                   help="Write PERSON_ENTERED/EXITED + zone_presence + "
                        "IDENTITY_CORRECTED to the database, evaluate alert "
                        "rules (webhook/WebSocket delivery), and capture "
                        "unknown-person enrollment candidates. Use together "
                        "with --recognize for named events.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only inference")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index to use on multi-GPU machines "
                        "(default: device 0). Sets CUDA_VISIBLE_DEVICES, so "
                        "torch and onnxruntime both follow. List devices "
                        "with: nvidia-smi --query-gpu=index,name,"
                        "pcie.link.width.current --format=csv")
    p.add_argument("--bypass-antispoofing", action="store_true",
                   help="Skip liveness check — DEV MODE ONLY")
    return p.parse_args()


def _detect_inference(force_cpu: bool) -> tuple[int, str]:
    """
    Returns (ctx_id, description) for insightface.prepare().
    ctx_id  0 → CUDA GPU
    ctx_id -1 → CPU (or CoreML on Apple Silicon, handled inside onnxruntime)
    """
    if force_cpu:
        return -1, "CPU (forced)"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return 0, "GPU (CUDA)"
        if "CoreMLExecutionProvider" in providers and ON_APPLE_SILICON:
            # insightface uses ctx_id=-1 → CPUExecutionProvider by default,
            # but onnxruntime on Apple Silicon will also try CoreML for compatible ops
            return -1, "CPU + CoreML (Apple Neural Engine)"
    except ImportError:
        pass
    return -1, "CPU"


def _check_deps() -> None:
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")        # need full (not headless) for imshow on Mac
    try:
        import insightface  # noqa: F401
    except ImportError:
        missing.append("insightface")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing.append("onnxruntime-gpu" if not ON_MAC else "onnxruntime")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    if missing:
        print("ERROR: Missing dependencies:")
        for pkg in missing:
            print(f"  {pkg}")
        if ON_APPLE_SILICON:
            print("\nFor Apple Silicon (M1/M2/M3/M4):")
            print("  brew install cmake")
            print("  .venv/bin/pip install cython scikit-build-core")
            print(f"  .venv/bin/pip install {' '.join(missing)}")
        else:
            print(f"\n  .venv/bin/pip install {' '.join(missing)}")
        sys.exit(1)


def _check_weights(weights_dir: Path) -> None:
    buffalo = weights_dir / "models" / "buffalo_l"
    if not buffalo.exists() or not any(buffalo.iterdir()):
        print(f"ERROR: Weights not found at {buffalo}")
        print("Run:  python scripts/download_models.py")
        sys.exit(1)


def _set_minimal_env() -> None:
    # Settings reads .env itself; real env vars would shadow it, so only set
    # dummy fallbacks when no .env exists (keeps imports working without one).
    if (ROOT / ".env").exists():
        return
    defaults = {
        "APP_SECRET_KEY": "devkeydevkeydevkeydevkeydevkeydev",
        "JWT_SECRET_KEY": "devjwtdevjwtdevjwtdevjwtdevjwtdevjwtdevjwtdevjwt",
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
        "TIMESCALE_URL": "postgresql+asyncpg://u:p@localhost/ts",
        "NEO4J_PASSWORD": "secret",
        "POSTGRES_PASSWORD": "secret",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


async def main() -> None:
    args = parse_args()

    # must happen before torch/onnxruntime touch CUDA: the chosen card is
    # remapped to device 0 for this process, so ctx_id=0 stays correct
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    _check_deps()

    weights_dir = ROOT / "models" / "weights"
    _check_weights(weights_dir)
    _set_minimal_env()

    # must run before any ONNX session is created, or the CUDA provider
    # fails to find libcublasLt/libcudnn and falls back to CPU
    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    import cv2
    from insightface.app import FaceAnalysis

    from src.ingestion.camera import CameraReader
    from src.ingestion.frame_producer import FrameProducer

    ctx_id, inference_label = _detect_inference(force_cpu=args.cpu)
    if args.gpu is not None and ctx_id == 0:
        inference_label += f" — physical GPU {args.gpu}"

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    det_size = args.det_size or (960 if ctx_id == 0 else 640)

    print("Loading models ...", end=" ", flush=True)
    # detector floor sits well below --det-thresh so the tracker still sees
    # low-confidence detections: ByteTrack stage 2 uses them to hold existing
    # tracks through dips (statue/borderline faces, motion blur) while only
    # >= det-thresh detections may start a new track
    det_floor = min(0.30, args.det_thresh)

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size),
                det_thresh=det_floor)
    print("OK\n")

    print("=" * 62)
    print(f"  platform        : {platform.system()} {platform.machine()}")
    print(f"  source          : {source}")
    print(f"  camera-id       : {args.camera_id}")
    print(f"  zone-id         : {args.zone_id}")
    print(f"  fps target      : {args.fps}")
    print(f"  det size        : {det_size}x{det_size}")
    print(f"  det thresh      : {args.det_thresh} (start track) / {det_floor} (sustain floor)")
    print(f"  min face px     : {args.min_face}")
    print(f"  inference       : {inference_label}")
    print(f"  display window  : {'yes  (press q to quit)' if args.show else 'no'}")
    # liveness runs in FacePipeline (layer 3), not in this detect+track
    # script — statues, photos and screens WILL be boxed and tracked here
    print(f"  anti-spoofing   : "
          f"{'BYPASSED — dev mode' if args.bypass_antispoofing else 'n/a in this script (FacePipeline layer, later step)'}")
    print("=" * 62) 
    print("\nPress Ctrl+C to stop.\n")

    from src.object_detection.detector import ObjectDetection
    from src.tracking.tracker import ByteTracker, iou

    # iou_threshold 0.2: face boxes are small, so even one frame of brisk
    # movement costs a lot of relative overlap; 0.3 splits tracks on it.
    # max_age 90 (~6 s at 15 fps) keeps a lost track alive through feed
    # stalls and turn-aways; ByteTracker's appearance stage then revives the
    # old ID from the face embedding when IoU can't bridge the gap.
    # appearance_threshold 0.45 matches face_match_accept_threshold — below
    # that, wrongly merging two people is worse than a fresh ID.
    tracker = ByteTracker(max_age=90, min_hits=3, iou_threshold=0.2,
                          high_threshold=args.det_thresh,
                          low_threshold=det_floor,
                          appearance_threshold=args.reid_thresh)

    # Layer 4 (recognition): track embedding -> pgvector gallery -> name.
    # Tracking IDs stay anonymous and session-local; the gallery lookup is
    # what attaches a persistent identity to a track.
    gallery = None
    person_names: dict[str, str] = {}
    # tid -> (label, sim_ema, frame_last_checked, consecutive_misses, person_id)
    track_labels: dict[int, tuple[str, float, int, int, str | None]] = {}
    RECHECK_FRAMES = 30   # ~2s at 15fps: live-updating identity without per-frame DB load
    # Identity is assigned to the TRACK, so a name persists through turned-away
    # faces and blur. It can only be questioned after an event that could have
    # swapped the track's owner: a return from a gap, or two boxes crossing.
    at_risk: dict[int, bool] = {}
    if args.recognize:
        from sqlalchemy import text as sql_text

        from src.core.database import get_db_session
        from src.face_recognition.gallery import FaceGallery
        gallery = FaceGallery()
        async with get_db_session() as session:
            rows = (await session.execute(sql_text(
                "SELECT person_id, display_name FROM persons WHERE is_active = true"
            ))).fetchall()
        person_names = {str(r[0]): str(r[1]) for r in rows}
        print(f"Recognition: ON — {len(person_names)} enrolled identity(ies): "
              f"{', '.join(person_names.values()) or '—'}\n")

    # Layer 6 (events/alerts): presence events with identity snapshots,
    # alert rules, delivery, unknown-person candidates. See sprint3 guide.
    event_store = None
    presence = None
    alert_engine = None
    delivery = None
    candidates = None
    presence_rows: dict[int, int] = {}     # track_id -> zone_presence row id
    named_since: dict[int, int] = {}       # track_id -> ns timestamp first named
    entered_label: dict[int, str] = {}     # track_id -> person at PERSON_ENTERED (for late correction)
    if args.persist_events:
        from src.alerts.delivery import AlertDelivery
        from src.alerts.engine import AlertEngine
        from src.core.config import get_settings
        from src.event_detection.event_store import EventStore
        from src.event_detection.zone_presence import PresentTrack, ZonePresenceMonitor
        from src.face_recognition.candidates import CandidateCapture

        event_store = EventStore()
        zone_types = await event_store.load_zone_types()
        presence = ZonePresenceMonitor(
            camera_id=args.camera_id, zone_id=args.zone_id,
            enter_grace_frames=5,
            exit_grace_frames=max(int(args.fps * 3), 30),
        )
        alert_engine = AlertEngine(
            ROOT / "configs" / "alerting" / "alert_rules.yaml", zone_types
        )
        delivery = AlertDelivery(webhook_url=get_settings().alert_webhook_url)
        candidates = CandidateCapture(camera_id=args.camera_id)
        zone_type = zone_types.get(args.zone_id, "general (zone not in DB!)")
        print(f"Persistence: ON — zone '{args.zone_id}' type={zone_type}, "
              f"webhook={'set' if get_settings().alert_webhook_url else 'none'}\n")
        if not args.recognize:
            print("  (note: without --recognize every person is 'unknown')\n")

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        if ON_MAC:
            print("  macOS: check System Settings → Privacy & Security → Camera")
            print("         and grant access to Terminal (or your IDE).")
        sys.exit(1)

    win_name = "third-eye | live  (q to quit)"
    if args.show:
        # resizable window scaled to fit the monitor — high-res RTSP frames
        # (e.g. 2304x1296) otherwise overflow the desktop at 1:1
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    unique_ids: set[int] = set()
    last_frame_t = 0.0
    last_infer_ms = 0.0
    fps_ema = 0.0
    last_drawn: dict[int, int] = {}    # track_id -> frame when last emitted
    came_back: dict[int, tuple[int, int]] = {}   # track_id -> (frame returned, frames gone)

    if args.show:
        print("Overlay: GREEN tracked | YELLOW sustained by low-conf det "
              "| ORANGE returned after gap (revival/coast)")
        print("Label:   ID:n c=det-confidence | h hits, b stored views, "
              "px box short side, back+Nf frames it was gone\n")
    producer = FrameProducer(
        camera=camera,
        camera_id=args.camera_id,
        zone_id=args.zone_id,
        max_fps=args.fps,
        drop_stale=True,  # live source: always process the newest frame
    )

    async def resolve_label(t, force: bool = False, good_view: bool = True) -> str:
        """Track-level identity (production pattern): a name belongs to the
        track and persists through turned-away faces and blur — low
        similarity on a continuous track is absence of evidence, not
        evidence of absence. Demotion requires BOTH a risk event (gap
        return / box crossing, tracked in `at_risk`) AND 2 consecutive
        failed re-checks. Re-verification of a named track only runs on
        good views (high-conf, big-enough face) so garbage frames can't
        strip a name. A different person matching with `accept` always
        relabels (logged)."""
        nonlocal gallery
        if gallery is None or t.embedding is None:
            return ""
        label, sim, checked, misses, pid = track_labels.get(
            t.track_id, ("", 0.0, -999, 0, None)
        )
        named = label not in ("", "unknown", "?")
        due = label == "" or force or frame_count - checked >= RECHECK_FRAMES
        if due and (good_view or not named):
            try:
                matches = await gallery.search(t.embedding, top_k=1)
            except Exception as exc:
                print(f"  ! recognition disabled — gallery search failed: {exc}")
                gallery = None
                return ""
            m = matches[0] if matches else None
            if m is not None and m.decision == "accept":
                new_name = person_names.get(m.person_id, m.person_id[:8])
                if named and new_name != label:
                    print(f"  ! track {t.track_id} relabeled {label} -> {new_name} "
                          f"(sim {m.similarity:.2f})")
                    if event_store is not None:
                        await event_store.write_identity_correction(
                            camera_id=args.camera_id, track_id=t.track_id,
                            from_label=label, to_label=new_name,
                            since_ns=named_since.get(t.track_id, time.time_ns()),
                            zone_id=args.zone_id, similarity=m.similarity,
                        )
                if not named or new_name != label:
                    named_since[t.track_id] = time.time_ns()
                sim = m.similarity if (not named or new_name != label) \
                    else 0.7 * sim + 0.3 * m.similarity
                label, misses, pid = new_name, 0, m.person_id
                at_risk.pop(t.track_id, None)   # verified: track is clean again
            elif named:
                # two-tier demotion: failures only count on good views (the
                # gate above), so these are clear faces NOT matching the name.
                # At-risk tracks (gap/crossing) demote fast; low-risk tracks
                # demote slowly — the safety net for swaps the risk detectors
                # missed (sub-threshold crossings, detector flicker)
                misses += 1
                limit = 2 if at_risk.get(t.track_id) else 4
                if misses >= limit:
                    why = ("risk event" if at_risk.get(t.track_id)
                           else f"{limit} clear views failed to verify")
                    print(f"  ! track {t.track_id} demoted: {label} -> unknown ({why})")
                    if event_store is not None:
                        await event_store.write_identity_correction(
                            camera_id=args.camera_id, track_id=t.track_id,
                            from_label=label, to_label="unknown",
                            since_ns=named_since.get(t.track_id, time.time_ns()),
                            zone_id=args.zone_id,
                            similarity=m.similarity if m else None,
                        )
                    named_since.pop(t.track_id, None)
                    label, sim, misses, pid = (
                        "unknown", (m.similarity if m else 0.0), 0, None
                    )
                    at_risk.pop(t.track_id, None)
            else:
                label = "?" if (m is not None and m.decision == "ambiguous") else "unknown"
                sim, misses, pid = (m.similarity if m else 0.0), 0, None
            track_labels[t.track_id] = (label, sim, frame_count, misses, pid)
        return f"{label} ({sim:.2f})"

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count, last_frame_t, last_infer_ms, fps_ema
        frame_count += 1

        now = time.monotonic()
        # tell camera stalls apart from slow inference: if the gap dwarfs the
        # last inference time, the source delivered nothing during it
        gap = now - last_frame_t
        if last_frame_t and gap > 0.3:
            cause = (
                "processing (inference slow)"
                if last_infer_ms / 1000.0 > gap * 0.7
                else "camera/stream (no frames arrived)"
            )
            print(
                f"  ! feed stall {gap * 1000:.0f} ms "
                f"(last inference {last_infer_ms:.0f} ms) — {cause}"
            )
        if last_frame_t:
            inst_fps = 1.0 / max(gap, 1e-6)
            fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps
        last_frame_t = now

        t_inf = time.monotonic()
        raw_faces = app.get(frame)
        last_infer_ms = (time.monotonic() - t_inf) * 1000.0

        # Wrap insightface detections as ObjectDetection for ByteTracker
        dets: list[ObjectDetection] = []
        for face in raw_faces:
            bbox = face.bbox.astype("float32")
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < args.min_face:
                continue
            dets.append(
                ObjectDetection(
                    bbox=bbox,
                    class_id=0,
                    class_name="person",
                    confidence=float(face.det_score),
                    embedding=getattr(face, "normed_embedding", None),
                )
            )

        tracked = tracker.update(dets)
        for t in tracked:
            unique_ids.add(t.track_id)

        # crossing boxes can swap track IDs — flag both for re-verification
        if gallery is not None:
            for i in range(len(tracked)):
                for j in range(i + 1, len(tracked)):
                    if iou(tracked[i].bbox, tracked[j].bbox) > 0.2:
                        at_risk[tracked[i].track_id] = True
                        at_risk[tracked[j].track_id] = True

        for t in tracked:
            bbox = t.bbox.astype(int)

            # gap > 3 frames means the box was coasting or revived — flag it
            frames_gone = frame_count - last_drawn.get(t.track_id, frame_count)
            last_drawn[t.track_id] = frame_count
            if frames_gone > 3:
                came_back[t.track_id] = (frame_count, frames_gone)
            back_frame, back_gap = came_back.get(t.track_id, (-999, 0))
            recently_back = frame_count - back_frame < 20

            sustained = t.confidence < args.det_thresh
            short_px = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))

            # a return-from-gap could be a different person on the same ID —
            # flag the track and re-verify immediately
            if frames_gone > 3:
                at_risk[t.track_id] = True
            name_txt = await resolve_label(
                t,
                force=frames_gone > 3,
                good_view=(not sustained) and short_px >= 32,
            )

            console_extra = f"  hits={t.hits} bank={len(t.embedding_bank)} {short_px}px"
            if name_txt:
                console_extra += f"  [{name_txt}]"
            if frames_gone > 3:
                console_extra += f"  BACK after {frames_gone} frames"
            if sustained:
                console_extra += "  (low-conf sustain)"
            print(
                f"[{meta.camera_id} | frame {meta.frame_id:>5}]  "
                f"track={t.track_id:>3}  "
                f"bbox=[{bbox[0]:4},{bbox[1]:4},{bbox[2]:4},{bbox[3]:4}]  "
                f"conf={t.confidence:.3f}{console_extra}"
            )
            if args.show:
                if recently_back:
                    color = (0, 165, 255)    # orange: returned after a gap
                elif sustained:
                    color = (0, 220, 220)    # yellow: alive on low-conf dets
                else:
                    color = (0, 220, 0)      # green: normal high-conf track
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                if name_txt:
                    cv2.putText(frame, name_txt,
                                (bbox[0], max(bbox[1] - 46, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                label = f"ID:{t.track_id} c={t.confidence:.2f}"
                sub = f"h{t.hits} b{len(t.embedding_bank)} {short_px}px"
                if recently_back:
                    sub += f" back+{back_gap}f"
                cv2.putText(frame, label,
                            (bbox[0], max(bbox[1] - 26, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, sub,
                            (bbox[0], max(bbox[1] - 8, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if presence is not None:
            present_list = []
            for t in tracked:
                label, sim, _, misses, pid = track_labels.get(
                    t.track_id, ("", 0.0, -999, 0, None)
                )
                if label in ("", "?", "unknown"):
                    state, name = "unknown", "unknown"
                else:
                    state = ("provisional"
                             if at_risk.get(t.track_id) or misses > 0
                             else "verified")
                    name = label
                present_list.append(PresentTrack(
                    track_id=t.track_id, person_id=pid, person_name=name,
                    identity_state=state, similarity=sim,
                ))
                bbox = t.bbox
                short = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                candidates.observe(
                    t.track_id, t.embedding,
                    is_unknown=(label == "unknown"),
                    good_view=t.confidence >= args.det_thresh and short >= 32,
                    quality=t.confidence,
                )

            try:
                for ev in presence.update(present_list):
                    print(f"  >> {ev.event_type}  {ev.person_name} "
                          f"[{ev.identity_state}]  zone={ev.zone_id}  track={ev.track_id}")
                    await event_store.write_presence_event(ev)
                    if ev.event_type == "PERSON_ENTERED":
                        row_id = await event_store.open_presence(ev)
                        if row_id is not None:
                            presence_rows[ev.track_id] = row_id
                        entered_label[ev.track_id] = ev.person_name
                    elif ev.event_type == "PERSON_EXITED":
                        row_id = presence_rows.pop(ev.track_id, None)
                        if row_id is not None:
                            await event_store.close_presence(row_id, ev.timestamp_ns)
                        entered_label.pop(ev.track_id, None)
                    for trig in alert_engine.evaluate(
                        event_type=ev.event_type, camera_id=ev.camera_id,
                        zone_id=ev.zone_id, person_id=ev.person_id,
                        person_name=ev.person_name,
                        identity_state=ev.identity_state,
                    ):
                        alert_id = await event_store.write_alert(
                            event_type=trig.event_type, severity=trig.severity,
                            camera_id=trig.camera_id, zone_id=trig.zone_id,
                            person_id=trig.person_id,
                            description=trig.description, payload=trig.payload,
                        )
                        print(f"  !! ALERT [{trig.severity}] {trig.description} "
                              f"(id={alert_id})")
                        delivery.deliver({
                            "alert_id": alert_id, "rule": trig.rule_name,
                            "event_type": trig.event_type,
                            "severity": trig.severity,
                            "camera_id": trig.camera_id, "zone_id": trig.zone_id,
                            "person": trig.payload.get("person"),
                            "description": trig.description,
                        })

                # late resolution: a track that ENTERED as unknown (recognition
                # hadn't caught up at entry) but is now recognized — re-attribute
                # its entry with an append-only correction AND patch its open
                # presence row, so the forensic record reflects the truth.
                for pt in present_list:
                    recognized = (
                        pt.identity_state in ("verified", "provisional")
                        and pt.person_name not in ("unknown", "?")
                    )
                    if not recognized:
                        continue
                    if entered_label.get(pt.track_id) == "unknown":
                        await event_store.write_identity_correction(
                            camera_id=args.camera_id, track_id=pt.track_id,
                            from_label="unknown", to_label=pt.person_name,
                            since_ns=time.time_ns(), zone_id=args.zone_id,
                            similarity=pt.similarity,
                        )
                        row_id = presence_rows.get(pt.track_id)
                        if row_id is not None:
                            await event_store.resolve_presence_identity(row_id, pt.person_id)
                        print(f"  >> IDENTITY_CORRECTED track {pt.track_id}: "
                              f"unknown -> {pt.person_name}")
                        entered_label[pt.track_id] = pt.person_name   # once
                    elif (
                        pt.track_id in presence_rows
                        and pt.track_id not in entered_label
                    ):
                        # diagnostic: this track is recognized and has an open
                        # entry, but no record of how it entered — means the
                        # entering track had a DIFFERENT id (face-track ID
                        # instability). Logs so we can confirm the hypothesis.
                        print(f"  .. [debug] track {pt.track_id} = {pt.person_name} "
                              f"has open entry but no entered_label (ID changed mid-presence?)")

                if frame_count % 30 == 0:
                    for draft in candidates.due():
                        cid = await candidates.persist(draft)
                        print(f"  >> enrollment candidate {cid[:8]}… created/merged "
                              f"(track {draft.track_id}, "
                              f"{len(draft.quality_scores)} crops)")
            except Exception as exc:
                # event persistence must never stall the frame loop
                print(f"  ! event persistence error (frame continues): {exc}")

        if frame_count % 30 == 0 and not tracked:
            print(f"  — {frame_count} frames processed, {len(unique_ids)} unique ID(s) so far —")

        if args.show:
            hud = (
                f"fps {fps_ema:4.1f}  infer {last_infer_ms:3.0f}ms  "
                f"det {len(raw_faces)}  trk {len(tracked)}  "
                f"ids {len(unique_ids)}  frame {frame_count}"
            )
            # dark outline + light fill keeps the HUD readable on any scene
            cv2.putText(frame, hud, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, hud, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

            # downscale for display: cheaper to draw and fits the monitor
            h, w = frame.shape[:2]
            scale = min(1600 / w, 900 / h, 1.0)
            disp = (
                cv2.resize(frame, (int(w * scale), int(h * scale)))
                if scale < 1.0 else frame
            )
            cv2.imshow(win_name, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                producer.stop()

    try:
        await producer.run(process_frame)
    except KeyboardInterrupt:
        pass
    finally:
        if delivery is not None:
            await delivery.drain()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"\nStopped — {frame_count} frames, {len(unique_ids)} unique person ID(s).")


if __name__ == "__main__":
    asyncio.run(main())
