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
import signal
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
                   help="Disable the liveness check entirely (no scoring, no "
                        "annotation) — DEV MODE ONLY")
    p.add_argument("--ignore-regions", default=None,
                   help="JSON list of normalized [x1,y1,x2,y2] rectangles "
                        "(0-1). Detections centered inside any are dropped — "
                        "masks a burnt-in timestamp/OSD or a fixed false "
                        "positive (e.g. a statue) so it isn't tracked.")
    p.add_argument("--collect-antispoofing", action="store_true",
                   help="Auto-save face crops to data/antispoofing/{live,spoof}/, "
                        "auto-labelled by the liveness gate's own verdict, for "
                        "later CDCN++ training. Labels are reviewable/correctable "
                        "from the Anti-Spoofing page. Requires liveness on (i.e. "
                        "not --bypass-antispoofing).")
    p.add_argument("--enforce-liveness", action="store_true",
                   help="HARD-BLOCK static presentations (photos/screens/"
                        "statues): a track judged non-live is forced to "
                        "'unknown', never recognized as an enrolled identity, "
                        "never captured as an enrollment candidate. Without "
                        "this flag liveness is still computed, drawn and logged "
                        "(observe mode) but does not block — live-test the real "
                        "vs spoof scores first, then turn enforcement on.")
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
    # remapped to device 0 for this process, so ctx_id=0 stays correct.
    # CUDA_DEVICE_ORDER defaults to FASTEST_FIRST, which on this 4-GPU box
    # does NOT match nvidia-smi's index (PCI bus order) — e.g. --gpu 1
    # silently selected the 3070 (CUDA's #1) instead of nvidia-smi's #1
    # (the 3060). Pin PCI_BUS_ID so --gpu N always means nvidia-smi's GPU N.
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    # Weight-free temporal liveness (micro-movement variance) runs per-track
    # below. A printed photo / static screen / statue shows near-zero variance
    # and scores as non-live.
    if args.bypass_antispoofing:
        antispoof_label = "BYPASSED — dev mode (no liveness scoring)"
    elif args.enforce_liveness:
        antispoof_label = "ENFORCED — static presentations blocked"
    else:
        antispoof_label = "observe-only (scored + drawn, not blocking; use --enforce-liveness)"
    print(f"  anti-spoofing   : {antispoof_label}")
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

    # Weight-free liveness: per-track variance of the face crop's frame-to-frame
    # luminance. A live face never holds perfectly still (blinks, micro-sway);
    # a print/screen/statue does, so it scores near 0. Fed the FACE CROP (not
    # the whole frame) so a moving background can't fake liveness. None when
    # bypassed. live_state: track_id -> (is_live, score, decided).
    # Normalized [x1,y1,x2,y2] rects whose detections are dropped (OSD/timestamp
    # mask, fixed false positives). Parsed once; malformed JSON disables it.
    ignore_regions: list[tuple[float, float, float, float]] = []
    if args.ignore_regions:
        try:
            import json as _json
            ignore_regions = [tuple(float(v) for v in r) for r in _json.loads(args.ignore_regions)]
            print(f"Ignore regions: {len(ignore_regions)} mask(s) active")
        except (ValueError, TypeError) as exc:
            print(f"Bad --ignore-regions ({exc}) — ignoring")

    from src.antispoofing.ensemble import TemporalConsistencyChecker
    liveness = None if args.bypass_antispoofing else TemporalConsistencyChecker()
    MOTION_THRESHOLD = 0.015   # temporal observe-only hint floor (see below)

    # CDCN++ texture model is the RELIABLE liveness enforcer (the weight-free
    # temporal signal is confounded by crop size/noise and can't block — it
    # stays observe-only). When a trained checkpoint exists it auto-loads here
    # and drives both enforcement and the auto-collection labels.
    cdcn = None
    cdcn_ema: dict[int, float] = {}         # track_id -> asymmetric-EMA live prob [0,1]
    cdcn_live: dict[int, bool] = {}         # track_id -> current verdict (hysteresis)
    cdcn_spoof_latch: dict[int, bool] = {}  # track_id -> hard-latched once clearly spoof
    cdcn_last_score: dict[int, float] = {}  # track_id -> last raw score (scored every Nth frame)
    # Hysteresis biased toward spoof. A held-up photo brought close/clean scores
    # higher per-frame (its moiré/bezel cues vanish); a symmetric mean would let
    # that cross 0.5 and "become real". So: react fast to spoof evidence, rise to
    # live only slowly, require a STRONG real reading to clear a spoof, and hard-
    # latch a track that ever reads clearly spoof (a real face never sustains it).
    CDCN_SPOOF_ENTER = 0.45    # live -> spoof once EMA drops below this
    CDCN_LIVE_EXIT = 0.75      # spoof -> live only once EMA climbs above this (sustained)
    CDCN_STRONG_SPOOF = 0.30   # EMA <= this hard-latches spoof for the track's life
    CDCN_EVERY = 3             # run the net every Nth frame (EMA carries it; cuts feed lag)
    if not args.bypass_antispoofing:
        cdcn_path = ROOT / "models" / "weights" / "cdcn_pp.pt"
        if cdcn_path.exists():
            try:
                from src.antispoofing.ensemble import CDCNWrapper
                cdcn_device = "cpu" if args.cpu else "cuda"
                cdcn = CDCNWrapper.load_from_checkpoint(str(cdcn_path), device=cdcn_device)
                print(f"CDCN++ liveness model loaded ({cdcn_device}) — enforcement uses CDCN++")
            except Exception as exc:
                print(f"CDCN++ load failed ({exc}) — temporal observe-only")

    # live_state: track_id -> (is_live, score, decided, can_block). can_block is
    # True only when CDCN++ is the verdict source — the temporal gate never blocks.
    live_state: dict[int, tuple[bool, float, bool, bool]] = {}

    # Auto-collection: save good-view face crops auto-labelled by the gate's
    # verdict for CDCN++ training. Needs the liveness checker to label them.
    # Bounded on two axes so it can't grow without limit: a per-LABEL cap keeps
    # disk in check, and a per-TRACK cap stops a single persistent object (e.g.
    # a static statue, which would otherwise emit one near-identical spoof crop
    # every second forever) from dominating the dataset.
    collect_dirs: dict[str, Path] = {}
    last_collect: dict[int, int] = {}   # track_id -> last frame a crop was saved
    collect_per_track: dict[int, int] = {}  # track_id -> crops saved this run
    collect_per_label: dict[str, int] = {}  # label -> total crops in its folder
    last_saved_gray: dict[int, object] = {}  # track_id -> last saved crop (small gray)
    COLLECT_EVERY = 15                  # ~1 crop/sec/track at 15fps, avoid flooding
    MAX_PER_TRACK = 40                  # one object/person can't flood the set
    MAX_PER_LABEL = 400                 # hard disk/balance cap per label
    # A static object yields near-identical crops; only keep one that differs
    # enough from the last saved for its track (mean abs gray diff, 0-255).
    DEDUP_MIN_DIFF = 8.0
    if args.collect_antispoofing and liveness is not None:
        for lbl in ("live", "spoof"):
            d = ROOT / "data" / "antispoofing" / lbl
            d.mkdir(parents=True, exist_ok=True)
            collect_dirs[lbl] = d
            collect_per_label[lbl] = len(list(d.glob("*.jpg")))  # count what's already there
        print(f"Anti-spoofing collection: ON → {ROOT / 'data' / 'antispoofing'} "
              f"(caps: {MAX_PER_LABEL}/label, {MAX_PER_TRACK}/track; "
              f"have live={collect_per_label['live']} spoof={collect_per_label['spoof']})")
    elif args.collect_antispoofing:
        print("Anti-spoofing collection requested but liveness is bypassed — disabled.")

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
    stm = None
    presence_rows: dict[int, int] = {}     # track_id -> zone_presence row id
    named_since: dict[int, int] = {}       # track_id -> ns timestamp first named
    # track_id -> open presence row_id for entries recorded as unknown, still
    # awaiting identity resolution. Keyed by entry track, but resolution can
    # come from a DIFFERENT track id (face-track churn) — see reconciliation.
    pending_unknown: dict[int, int] = {}
    if args.persist_events:
        from src.alerts.delivery import AlertDelivery
        from src.alerts.engine import AlertEngine
        from src.core.config import get_settings
        from src.event_detection.event_store import EventStore
        from src.event_detection.zone_presence import PresentTrack, ZonePresenceMonitor
        from src.face_recognition.candidates import CandidateCapture
        from src.memory.short_term import get_short_term_memory

        event_store = EventStore()
        stm = get_short_term_memory()
        # self-register zone + camera so zone_presence FKs are satisfied
        # (otherwise the first PERSON_ENTERED's presence insert FK-violates and
        # aborts event handling — including the identity correction — each frame)
        await event_store.ensure_zone(args.zone_id)
        await event_store.ensure_camera(
            args.camera_id, stream_url=str(args.source), zone_id=args.zone_id
        )
        zone_types = await event_store.load_zone_types()
        presence = ZonePresenceMonitor(
            camera_id=args.camera_id, zone_id=args.zone_id,
            enter_grace_frames=5,
            exit_grace_frames=max(int(args.fps * 3), 30),
            # ~3s for recognition to resolve before declaring a genuine stranger
            unknown_grace_frames=max(int(args.fps * 3), 60),
        )
        alert_engine = AlertEngine(
            ROOT / "configs" / "alerting" / "alert_rules.yaml", zone_types
        )
        delivery = AlertDelivery(webhook_url=get_settings().alert_webhook_url)
        from src.storage import get_object_store
        store = get_object_store()
        if store.is_enabled:
            store.ensure_bucket()
        candidates = CandidateCapture(camera_id=args.camera_id, object_store=store)
        if store.is_enabled:
            print("  object storage: on — unknown face crops saved to MinIO")
        else:
            print("  object storage: OFF — candidates captured WITHOUT photos "
                  "(set S3_ENDPOINT_URL + `pip install boto3` to enable crops)")
        zone_type = zone_types.get(args.zone_id, "general (zone not in DB!)")
        print(f"Persistence: ON — zone '{args.zone_id}' type={zone_type}, "
              f"webhook={'set' if get_settings().alert_webhook_url else 'none'}")
        print("  build: identity-correction churn-robust + diagnostics\n")
        if not args.recognize:
            print("  (note: without --recognize every person is 'unknown')\n")

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        if ON_MAC:
            print("  macOS: check System Settings → Privacy & Security → Camera")
            print("         and grant access to Terminal (or your IDE).")
        if stm is not None:
            await stm.set_camera_status(args.camera_id, "offline")
        sys.exit(1)
    if stm is not None:
        await stm.set_camera_status(args.camera_id, "online")

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
        old_pid = pid
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
                if stm is not None:
                    # re-add every accept cycle (not just on change) to keep
                    # the zone-occupancy TTL alive for the whole presence
                    if old_pid and old_pid != pid:
                        await stm.remove_zone_occupant(args.zone_id, old_pid)
                    await stm.add_zone_occupant(args.zone_id, pid)
                    await stm.set_identity_location(pid, args.zone_id, track_id=str(t.track_id))
                    await stm.register_name_lookup(new_name, pid)
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
                    if stm is not None and old_pid:
                        await stm.remove_zone_occupant(args.zone_id, old_pid)
            else:
                label = "?" if (m is not None and m.decision == "ambiguous") else "unknown"
                sim, misses, pid = (m.similarity if m else 0.0), 0, None
            track_labels[t.track_id] = (label, sim, frame_count, misses, pid)
        return f"{label} ({sim:.2f})"

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count, last_frame_t, last_infer_ms, fps_ema
        frame_count += 1

        # refresh camera:{id}:status TTL periodically so it doesn't expire
        # (120s TTL) while the feed is healthy — every ~10s at 15fps
        if stm is not None and frame_count % 150 == 0:
            await stm.set_camera_status(args.camera_id, "online")

        # person_names is a startup-time snapshot; without this, a person
        # approved via the dashboard while this process keeps running would
        # show up as a raw person_id[:8] (see resolve_label) until restart —
        # same ~10s cadence as the camera-status heartbeat above.
        if gallery is not None and frame_count % 150 == 0:
            try:
                async with get_db_session() as session:
                    rows = (await session.execute(sql_text(
                        "SELECT person_id, display_name FROM persons WHERE is_active = true"
                    ))).fetchall()
                person_names.clear()
                person_names.update({str(r[0]): str(r[1]) for r in rows})
            except Exception as exc:
                print(f"  ! person_names refresh failed (keeping stale names): {exc}")

        now = time.monotonic()
        # tell camera stalls apart from slow inference: if the gap dwarfs the
        # last inference time, the source delivered nothing during it
        gap = now - last_frame_t
        # skip the first few frames: the first inference includes one-time
        # model/CUDA warmup (~2s) and isn't a real stall
        if last_frame_t and gap > 0.3 and frame_count > 3:
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
        fh, fw = frame.shape[:2]
        dets: list[ObjectDetection] = []
        for face in raw_faces:
            bbox = face.bbox.astype("float32")
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < args.min_face:
                continue
            # Ignore-region mask: drop a detection whose center falls inside a
            # masked rect (burnt-in OSD/timestamp that ticks, or a fixed
            # false-positive like a statue). Rects are normalized [x1,y1,x2,y2].
            cx = ((bbox[0] + bbox[2]) / 2) / fw
            cy = ((bbox[1] + bbox[3]) / 2) / fh
            if any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                   for rx1, ry1, rx2, ry2 in ignore_regions):
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

        render_info: list[dict] = []
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

            # Liveness verdict. CDCN++ (texture, trained) is the reliable
            # enforcer when loaded; the weight-free temporal motion is only an
            # observe-only hint (confounded by crop size, so it never blocks).
            is_live, live_score, decided, live_max, can_block = True, 1.0, False, 0.0, False
            if liveness is not None:
                tid = str(t.track_id)
                x1, y1 = max(bbox[0], 0), max(bbox[1], 0)
                x2, y2 = max(bbox[2], 0), max(bbox[3], 0)
                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size:
                    # Resize to a fixed canonical size: the face box changes
                    # dimensions frame to frame, but the checker diffs the
                    # current crop against the previous one and needs matching
                    # shapes. Normalizing also keeps the variance signal
                    # comparable across near vs far faces.
                    liveness.update(tid, cv2.resize(raw_crop, (112, 112)))
                # Temporal motion: observe-only hint (mean over ~5s; does not block).
                live_score = liveness.recent_mean_motion(tid)
                live_max = liveness.recent_max_motion(tid)
                decided = liveness.long_window_full(tid)
                is_live = (not decided) or live_score >= MOTION_THRESHOLD

                # CDCN++ overrides as the verdict when loaded — it enforces and
                # labels the collected crops. Scored every Nth frame (the EMA
                # carries it between), then turned into a verdict with spoof-
                # biased hysteresis + a hard spoof latch, so a photo can't drift
                # back to "real" once it has been caught.
                if cdcn is not None and raw_crop.size:
                    tk = t.track_id
                    if tk not in cdcn_last_score or frame_count % CDCN_EVERY == 0:
                        cdcn_last_score[tk] = cdcn.score(raw_crop)
                    s = cdcn_last_score[tk]
                    prev = cdcn_ema.get(tk)
                    if prev is None:
                        ema = s
                    elif s < prev:
                        ema = 0.5 * prev + 0.5 * s     # react fast to spoof evidence
                    else:
                        ema = 0.85 * prev + 0.15 * s   # rise to "live" only slowly
                    cdcn_ema[tk] = ema
                    live_score = ema
                    if ema <= CDCN_STRONG_SPOOF:
                        cdcn_spoof_latch[tk] = True
                    if cdcn_spoof_latch.get(tk):
                        is_live = False
                    else:
                        was_live = cdcn_live.get(tk, True)
                        if was_live and ema < CDCN_SPOOF_ENTER:
                            is_live = False
                        elif not was_live and ema > CDCN_LIVE_EXIT:
                            is_live = True
                        else:
                            is_live = was_live
                    cdcn_live[tk] = is_live
                    decided = True
                    can_block = True

                live_state[t.track_id] = (is_live, live_score, decided, can_block)

                # Auto-collection: once the gate has decided, save the crop
                # under its verdict's folder (throttled per track) so the
                # dataset is auto-labelled live/spoof for CDCN++ training. The
                # operator corrects mistakes later on the Anti-Spoofing page.
                if (collect_dirs and decided and raw_crop.size
                        and short_px >= 32
                        and frame_count - last_collect.get(t.track_id, -999) >= COLLECT_EVERY):
                    lbl = "live" if is_live else "spoof"
                    # Near-duplicate skip: a static object yields the same crop
                    # every tick; only keep one that differs from the last saved.
                    small_gray = cv2.cvtColor(cv2.resize(raw_crop, (64, 64)),
                                              cv2.COLOR_BGR2GRAY)
                    prev_gray = last_saved_gray.get(t.track_id)
                    too_similar = (
                        prev_gray is not None
                        and cv2.mean(cv2.absdiff(small_gray, prev_gray))[0] < DEDUP_MIN_DIFF
                    )
                    if (not too_similar
                            and collect_per_label.get(lbl, 0) < MAX_PER_LABEL
                            and collect_per_track.get(t.track_id, 0) < MAX_PER_TRACK):
                        fname = (collect_dirs[lbl]
                                 / f"{args.camera_id}_{frame_count:07d}_t{t.track_id}.jpg")
                        cv2.imwrite(str(fname), raw_crop)
                        last_collect[t.track_id] = frame_count
                        last_saved_gray[t.track_id] = small_gray
                        collect_per_track[t.track_id] = collect_per_track.get(t.track_id, 0) + 1
                        collect_per_label[lbl] = collect_per_label.get(lbl, 0) + 1
                        if collect_per_label[lbl] == MAX_PER_LABEL:
                            print(f"  [collect] '{lbl}' reached cap {MAX_PER_LABEL} — "
                                  f"no more '{lbl}' crops will be saved this run")

            # a return-from-gap could be a different person on the same ID —
            # flag the track and re-verify immediately
            if frames_gone > 3:
                at_risk[t.track_id] = True
            name_txt = await resolve_label(
                t,
                force=frames_gone > 3,
                good_view=(not sustained) and short_px >= 32,
            )
            # Enforcement: a confirmed spoof can't be an enrolled person — drop
            # the name so it never recognizes or persists as one. Only CDCN++
            # can block (can_block); the temporal hint never does.
            blocked = (args.enforce_liveness and can_block
                       and decided and not is_live)
            if blocked and name_txt:
                name_txt = ""
            render_info.append({
                "t": t, "bbox": bbox, "frames_gone": frames_gone,
                "recently_back": recently_back, "back_gap": back_gap,
                "sustained": sustained, "short_px": short_px,
                "name_txt": name_txt,
                "is_live": is_live, "live_score": live_score,
                "live_max": live_max if liveness is not None else 0.0,
                "decided": decided, "blocked": blocked, "can_block": can_block,
            })

        # Same-frame identity collision: two DIFFERENT simultaneously-tracked
        # bodies cannot legitimately be the same enrolled person. A crossing
        # or a noisy embedding can momentarily give two tracks an `accept`
        # match against the same gallery entry; the per-track misses-based
        # demotion in resolve_label is too slow to catch this (it only counts
        # FAILED matches). Enforce exclusivity immediately: the
        # higher-similarity claimant keeps the name, the other(s) are forced
        # back to unknown this frame (logged as an IDENTITY_CORRECTED, same
        # as any other demotion).
        claims: dict[str, list[int]] = {}
        for info in render_info:
            _, _, _, _, pid = track_labels.get(
                info["t"].track_id, ("", 0.0, -999, 0, None)
            )
            if pid is not None:
                claims.setdefault(pid, []).append(info["t"].track_id)
        spoofy_tracks = {
            tid for tid, lv in live_state.items()
            if lv[3] and lv[2] and not lv[0]   # can_block and decided and not is_live
        }
        for claimant_ids in claims.values():
            if len(claimant_ids) < 2:
                continue
            # A live face beats a spoof for a contested identity: a clean, frontal
            # held-up photo can out-score the real (often angled) person on
            # similarity, which would hand the name to the spoof and demote the
            # real person to unknown. Rank live-first, then by similarity.
            claimant_ids.sort(
                key=lambda tid: (1 if tid in spoofy_tracks else 0, -track_labels[tid][1]))
            keeper_id = claimant_ids[0]
            for loser_id in claimant_ids[1:]:
                label, sim, _, _, _ = track_labels[loser_id]
                print(f"  ! track {loser_id} demoted: {label} -> unknown "
                      f"(same identity also claimed by track {keeper_id} "
                      f"this frame)")
                if event_store is not None:
                    await event_store.write_identity_correction(
                        camera_id=args.camera_id, track_id=loser_id,
                        from_label=label, to_label="unknown",
                        since_ns=named_since.get(loser_id, time.time_ns()),
                        zone_id=args.zone_id, similarity=sim,
                    )
                named_since.pop(loser_id, None)
                track_labels[loser_id] = ("unknown", sim, frame_count, 0, None)
                at_risk[loser_id] = True
                for info in render_info:
                    if info["t"].track_id == loser_id:
                        info["name_txt"] = f"unknown ({sim:.2f})"

        for info in render_info:
            t, bbox = info["t"], info["bbox"]
            frames_gone, recently_back = info["frames_gone"], info["recently_back"]
            back_gap, sustained = info["back_gap"], info["sustained"]
            short_px, name_txt = info["short_px"], info["name_txt"]

            console_extra = f"  hits={t.hits} bank={len(t.embedding_bank)} {short_px}px"
            if name_txt:
                console_extra += f"  [{name_txt}]"
            if frames_gone > 3:
                console_extra += f"  BACK after {frames_gone} frames"
            if sustained:
                console_extra += "  (low-conf sustain)"
            if info.get("decided"):
                tag = ""
                if not info.get("is_live", True):
                    tag = f"  SPOOF?{'/BLOCKED' if info.get('blocked') else ''}"
                if info.get("can_block"):   # CDCN++ verdict
                    console_extra += f"{tag}  cdcn={info.get('live_score', 0.0):.3f}"
                else:                       # temporal observe-only hint
                    console_extra += (
                        f"  motion_mean={info.get('live_score', 0.0):.4f} "
                        f"max={info.get('live_max', 0.0):.4f}"
                    )
            print(
                f"[{meta.camera_id} | frame {meta.frame_id:>5}]  "
                f"track={t.track_id:>3}  "
                f"bbox=[{bbox[0]:4},{bbox[1]:4},{bbox[2]:4},{bbox[3]:4}]  "
                f"conf={t.confidence:.3f}{console_extra}"
            )
            if args.show or stm is not None:
                # Only flag a spoof visually on the reliable CDCN++ verdict;
                # the temporal hint is too noisy to colour a box on.
                static = (info.get("decided") and not info.get("is_live", True)
                          and info.get("can_block"))
                if static:
                    color = (0, 0, 255)      # red: CDCN++ says spoof
                elif recently_back:
                    color = (0, 165, 255)    # orange: returned after a gap
                elif sustained:
                    color = (0, 220, 220)    # yellow: alive on low-conf dets
                else:
                    color = (0, 220, 0)      # green: normal high-conf track
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                if static:
                    tag = "SPOOF — BLOCKED" if info.get("blocked") else "SPOOF?"
                    cv2.putText(frame, tag,
                                (bbox[0], min(bbox[3] + 18, frame.shape[0] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
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
                # Enforced static presentation: present as unknown with no
                # person_id, so a held-up photo of an enrolled person can't be
                # logged as that person entering.
                lv_live, _, lv_decided, lv_can_block = live_state.get(
                    t.track_id, (True, 1.0, False, False))
                blocked = (args.enforce_liveness and lv_can_block
                           and lv_decided and not lv_live)
                if blocked:
                    label, pid = "unknown", None
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
                # Only ENFORCE mode bars a spoof from becoming a candidate. In
                # observe mode we deliberately don't gate on the spoof verdict:
                # the model isn't perfect and a real person must never be
                # silently barred from enrollment. A spoof photo reaching Pending
                # Enrollments is acceptable — reject it there.
                good_view = (not blocked) and t.confidence >= args.det_thresh and short >= 32
                # for an unknown good view, hand the capture a JPEG face crop so
                # the candidate has a viewable photo (stored in MinIO)
                crop_jpeg = None
                if label == "unknown" and good_view and store.is_enabled:
                    b = bbox.astype(int)
                    x1, y1 = max(b[0], 0), max(b[1], 0)
                    x2, y2 = max(b[2], 0), max(b[3], 0)
                    face = frame[y1:y2, x1:x2]
                    if face.size:
                        ok_enc, buf = cv2.imencode(".jpg", face)
                        if ok_enc:
                            crop_jpeg = buf.tobytes()
                candidates.observe(
                    t.track_id, t.embedding,
                    is_unknown=(label == "unknown"),
                    good_view=good_view,
                    quality=t.confidence,
                    crop=crop_jpeg,
                )

            try:
                for ev in presence.update(present_list):
                    print(f"  >> {ev.event_type}  {ev.person_name} "
                          f"[{ev.identity_state}]  zone={ev.zone_id}  track={ev.track_id}")
                    await event_store.write_presence_event(ev)
                    if ev.event_type == "PERSON_ENTERED":
                        # park unknowns FIRST so a presence-insert failure can't
                        # stop the later identity correction
                        if ev.person_name == "unknown":
                            pending_unknown[ev.track_id] = -1
                        try:
                            row_id = await event_store.open_presence(ev)
                        except Exception as exc:
                            row_id = None
                            print(f"  ! open_presence failed (continuing): {exc}")
                        if row_id is not None:
                            presence_rows[ev.track_id] = row_id
                            if ev.track_id in pending_unknown:
                                pending_unknown[ev.track_id] = row_id
                        if ev.person_name == "unknown":
                            print(f"  .. [diag] pending unknown entry track={ev.track_id} "
                                  f"row={row_id}; awaiting identity")
                    elif ev.event_type == "PERSON_EXITED":
                        row_id = presence_rows.pop(ev.track_id, None)
                        if row_id is not None:
                            await event_store.close_presence(row_id, ev.timestamp_ns)
                        pending_unknown.pop(ev.track_id, None)
                        if stm is not None and ev.person_id:
                            await stm.remove_zone_occupant(ev.zone_id, ev.person_id)
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

                # late resolution: an entry recorded as unknown (recognition
                # hadn't caught up) is now recognized — re-attribute it with an
                # append-only correction AND patch the open presence row.
                # Same-track first (the entry's own track resolved). The
                # single-pending fallback handles face-track id churn (entry and
                # resolution under different ids) — but ONLY for a track that has
                # no entry of its own (`not in presence_rows`); otherwise an
                # already-present person (e.g. Rohan) would wrongly claim a newly
                # entered person's (Subhra's) pending entry. Multiple pending
                # unknowns are deferred to Sprint 6 identity-level presence.
                for pt in present_list:
                    recognized = (
                        pt.identity_state in ("verified", "provisional")
                        and pt.person_name not in ("unknown", "?")
                    )
                    if not recognized or not pending_unknown:
                        continue
                    if pt.track_id in pending_unknown:
                        old_tid, row_id = pt.track_id, pending_unknown.pop(pt.track_id)
                    elif len(pending_unknown) == 1 and pt.track_id not in presence_rows:
                        # churn: this recognized track has no entry of its own,
                        # and there's exactly one open unknown → same person
                        old_tid, row_id = pending_unknown.popitem()
                    else:
                        continue   # ambiguous / already-present — don't guess
                    await event_store.write_identity_correction(
                        camera_id=args.camera_id, track_id=pt.track_id,
                        from_label="unknown", to_label=pt.person_name,
                        since_ns=time.time_ns(), zone_id=args.zone_id,
                        similarity=pt.similarity,
                    )
                    if row_id and row_id > 0:
                        await event_store.resolve_presence_identity(row_id, pt.person_id)
                    arrow = (f"track {old_tid}->{pt.track_id}"
                             if old_tid != pt.track_id else f"track {pt.track_id}")
                    print(f"  >> IDENTITY_CORRECTED {arrow}: unknown -> {pt.person_name}")

                if frame_count % 30 == 0:
                    for draft in candidates.due():
                        cid = await candidates.persist(draft)
                        print(f"  >> enrollment candidate {cid[:8]}… created/merged "
                              f"(track {draft.track_id}, "
                              f"{len(draft.quality_scores)} crops)")
            except Exception as exc:
                # event persistence must never stall the frame loop — but make
                # the failure LOUD with a traceback so a silently-failing write
                # (e.g. a correction insert) is visible during testing
                import traceback
                print(f"  ! event persistence error (frame continues): {exc}")
                traceback.print_exc()

        # Drop liveness state for tracks no longer present so the per-track
        # buffers (and the stored prev-frame each holds) don't accumulate over a
        # long session.
        if liveness is not None and frame_count % 30 == 0:
            active = {t.track_id for t in tracked}
            for stale in [tid for tid in live_state if tid not in active]:
                liveness.clear(str(stale))
                live_state.pop(stale, None)
                last_collect.pop(stale, None)
                collect_per_track.pop(stale, None)
                last_saved_gray.pop(stale, None)
                cdcn_ema.pop(stale, None)
                cdcn_live.pop(stale, None)
                cdcn_spoof_latch.pop(stale, None)
                cdcn_last_score.pop(stale, None)

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

        # Live view (dashboard click-to-expand modal): publish a downscaled
        # JPEG every 2nd frame over Redis pub/sub for the MJPEG endpoint
        # (src/api/routers/cameras.py) to relay — a PUBLISH with nobody
        # subscribed is essentially free, so this runs unconditionally.
        # Camera-tile snapshot (the small thumbnails, polled every 2s by
        # the dashboard) reuses the same encode but only needs writing to
        # disk every ~10 frames. Atomic write (tmp + rename) so /stream
        # never serves a half-written file.
        if stm is not None and frame_count % 2 == 0:
            h, w = frame.shape[:2]
            scale = min(960 / w, 1.0)
            snap = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            ok_enc, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok_enc:
                jpeg_bytes = buf.tobytes()
                await stm.publish_frame(args.camera_id, jpeg_bytes)
                if frame_count % 10 == 0:
                    snap_dir = ROOT / "data" / "snapshots" / args.camera_id
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = snap_dir / "latest.jpg.tmp"
                    tmp_path.write_bytes(jpeg_bytes)
                    tmp_path.replace(snap_dir / "latest.jpg")

    # SIGINT (Ctrl+C) already stops cleanly via asyncio's default handling.
    # SIGTERM has no such default — a plain kill would skip the finally:
    # block below (drain alerts, release the camera) entirely. An
    # orchestrator that manages this process (see Settings/camera
    # supervisor) sends SIGTERM for a normal stop, so it needs the same
    # graceful path: producer.stop() just flips a flag the run loop already
    # checks every iteration, so run() returns normally instead of raising.
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, producer.stop)
    except (NotImplementedError, AttributeError):
        pass  # Windows: no add_signal_handler; SIGTERM isn't meaningful there anyway

    try:
        await producer.run(process_frame)
    except KeyboardInterrupt:
        pass
    finally:
        # A graceful stop (SIGTERM/Ctrl+C) still drops out of the frame
        # loop mid-track - anyone currently "entered" never gets a
        # PERSON_EXITED, so their zone_presence row was staying open
        # forever (exit_time NULL) even though the process that would
        # have closed it is gone. Found live: a zone changed via Settings
        # (which restarts the process) left the old zone's rows stuck
        # open. Close them explicitly here instead of abandoning them.
        if event_store is not None and presence_rows:
            now_ns = time.time_ns()
            for tid, row_id in list(presence_rows.items()):
                try:
                    await event_store.close_presence(row_id, now_ns)
                except Exception as exc:
                    print(f"  ! failed to close presence row {row_id} (track {tid}) on shutdown: {exc}")
                if stm is not None:
                    label_entry = track_labels.get(tid)
                    pid = label_entry[4] if label_entry else None
                    if pid:
                        try:
                            await stm.remove_zone_occupant(args.zone_id, pid)
                        except Exception:
                            pass
            print(f"  closed {len(presence_rows)} still-open zone_presence row(s) on shutdown")
        if delivery is not None:
            await delivery.drain()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"\nStopped — {frame_count} frames, {len(unique_ids)} unique person ID(s).")


if __name__ == "__main__":
    asyncio.run(main())
