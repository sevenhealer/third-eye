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
                   help="Face detection confidence threshold (default 0.6; "
                        "insightface default is 0.5). Raise to suppress "
                        "low-light false faces on furniture/dark corners, "
                        "lower if real faces in dim areas are missed.")
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
    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size),
                det_thresh=args.det_thresh)
    print("OK\n")

    print("=" * 62)
    print(f"  platform        : {platform.system()} {platform.machine()}")
    print(f"  source          : {source}")
    print(f"  camera-id       : {args.camera_id}")
    print(f"  zone-id         : {args.zone_id}")
    print(f"  fps target      : {args.fps}")
    print(f"  det size        : {det_size}x{det_size}")
    print(f"  det thresh      : {args.det_thresh}")
    print(f"  min face px     : {args.min_face}")
    print(f"  inference       : {inference_label}")
    print(f"  display window  : {'yes  (press q to quit)' if args.show else 'no'}")
    print(f"  anti-spoofing   : {'BYPASSED — dev mode' if args.bypass_antispoofing else 'enabled'}")
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
                          high_threshold=0.5, low_threshold=0.1,
                          appearance_threshold=0.45)

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
    producer = FrameProducer(
        camera=camera,
        camera_id=args.camera_id,
        zone_id=args.zone_id,
        max_fps=args.fps,
        drop_stale=True,  # live source: always process the newest frame
    )

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count
        frame_count += 1

        raw_faces = app.get(frame)

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
            print(
                f"[{meta.camera_id} | frame {meta.frame_id:>5}]  "
                f"track={t.track_id:>3}  "
                f"bbox=[{bbox[0]:4},{bbox[1]:4},{bbox[2]:4},{bbox[3]:4}]  "
                f"conf={t.confidence:.3f}"
            )
            if args.show:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 0), 2)
                cv2.putText(
                    frame, f"ID:{t.track_id}",
                    (bbox[0], max(bbox[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2,
                )

        if frame_count % 30 == 0 and not tracked:
            print(f"  — {frame_count} frames processed, {len(unique_ids)} unique ID(s) so far —")

        if args.show:
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
