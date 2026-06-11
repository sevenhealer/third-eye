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
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only inference")
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
    from src.tracking.tracker import ByteTracker

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

            console_extra = f"  hits={t.hits} bank={len(t.embedding_bank)} {short_px}px"
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
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"\nStopped — {frame_count} frames, {len(unique_ids)} unique person ID(s).")


if __name__ == "__main__":
    asyncio.run(main())
