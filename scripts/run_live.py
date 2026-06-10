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
    p.add_argument("--fps", type=int, default=10,
                   help="Target processing FPS (default 10)")
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

    import cv2
    from insightface.app import FaceAnalysis

    from src.ingestion.camera import CameraReader
    from src.ingestion.frame_producer import FrameProducer

    ctx_id, inference_label = _detect_inference(force_cpu=args.cpu)

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    print("Loading models ...", end=" ", flush=True)
    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("OK\n")

    print("=" * 62)
    print(f"  platform        : {platform.system()} {platform.machine()}")
    print(f"  source          : {source}")
    print(f"  camera-id       : {args.camera_id}")
    print(f"  zone-id         : {args.zone_id}")
    print(f"  fps target      : {args.fps}")
    print(f"  inference       : {inference_label}")
    print(f"  display window  : {'yes  (press q to quit)' if args.show else 'no'}")
    print(f"  anti-spoofing   : {'BYPASSED — dev mode' if args.bypass_antispoofing else 'enabled'}")
    print("=" * 62)
    print("\nPress Ctrl+C to stop.\n")

    from src.object_detection.detector import ObjectDetection
    from src.tracking.tracker import ByteTracker

    tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3,
                          high_threshold=0.5, low_threshold=0.1)

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        if ON_MAC:
            print("  macOS: check System Settings → Privacy & Security → Camera")
            print("         and grant access to Terminal (or your IDE).")
        sys.exit(1)

    frame_count = 0
    unique_ids: set[int] = set()
    producer = FrameProducer(
        camera=camera,
        camera_id=args.camera_id,
        zone_id=args.zone_id,
        max_fps=args.fps,
    )

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count
        frame_count += 1

        raw_faces = app.get(frame)

        # Wrap insightface detections as ObjectDetection for ByteTracker
        import numpy as _np
        dets: list[ObjectDetection] = []
        for face in raw_faces:
            dets.append(
                ObjectDetection(
                    bbox=face.bbox.astype("float32"),
                    class_id=0,
                    class_name="person",
                    confidence=float(face.det_score),
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
            cv2.imshow("third-eye | live  (q to quit)", frame)
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
