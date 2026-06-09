#!/usr/bin/env python3
"""
Live camera feed — face detection and recognition.

Works on macOS (FaceTime/USB camera, CPU inference) and Linux with RTX 3090
(CUDA inference). GPU vs CPU is detected automatically from ONNX Runtime.

Usage:
  python scripts/run_live.py --source 0           # default camera
  python scripts/run_live.py --source 0 --show    # display cv2 window
  python scripts/run_live.py --source rtsp://...  # IP camera
  python scripts/run_live.py --source 0 --bypass-antispoofing   # dev mode

Mac install:
  .venv/bin/pip install insightface onnxruntime opencv-python numpy

Linux (RTX 3090) install:
  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy

Prerequisites:
  python scripts/download_models.py        # download weights first (~200 MB)
  docker compose up -d postgres redis      # only needed for gallery lookups
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="third-eye live feed")
    p.add_argument("--source", default="0",
                   help="Camera index (0, 1, ...) or RTSP URL")
    p.add_argument("--camera-id", default="cam0")
    p.add_argument("--zone-id", default="entrance")
    p.add_argument("--fps", type=int, default=10,
                   help="Target processing FPS (default 10; reduce if CPU is slow)")
    p.add_argument("--show", action="store_true",
                   help="Open a cv2 window with bounding boxes (requires opencv-python, not headless)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference even if CUDA is available")
    p.add_argument("--bypass-antispoofing", action="store_true",
                   help="Skip liveness check — DEV MODE ONLY")
    return p.parse_args()


def _onnx_ctx_id(force_cpu: bool = False) -> int:
    """Return 0 (GPU/CUDA) or -1 (CPU) based on available ONNX providers."""
    if force_cpu:
        return -1
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return 0
    except ImportError:
        pass
    return -1


def _check_deps() -> None:
    import platform
    on_mac = platform.system() == "Darwin"

    missing_pkgs: list[str] = []
    missing_display: list[str] = []

    try:
        import cv2  # noqa: F401
    except ImportError:
        # On Mac, opencv-python (not headless) works for both capture and imshow
        missing_pkgs.append("opencv-python" if on_mac else "opencv-python-headless")
        missing_display.append("opencv-python" if on_mac else "opencv-python-headless")

    try:
        import insightface  # noqa: F401
    except ImportError:
        missing_pkgs.append("insightface")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        # onnxruntime-gpu only exists for Linux+CUDA
        missing_pkgs.append("onnxruntime" if on_mac else "onnxruntime-gpu")

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing_pkgs.append("numpy")

    if missing_pkgs:
        print("ERROR: Missing dependencies. Install with:")
        print(f"  .venv/bin/pip install {' '.join(missing_pkgs)}")
        sys.exit(1)


def _check_weights(weights_dir: Path) -> None:
    buffalo = weights_dir / "buffalo_l"
    if not buffalo.exists() or not any(buffalo.iterdir()):
        print(f"ERROR: Model weights not found at {buffalo}")
        print("Run first:  python scripts/download_models.py")
        sys.exit(1)


def _set_minimal_env() -> None:
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

    ctx_id = _onnx_ctx_id(force_cpu=args.cpu)
    inference_mode = "CPU" if ctx_id == -1 else "GPU (CUDA)"

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    print("Loading models ...", end=" ", flush=True)
    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("OK\n")

    print("=" * 62)
    print(f"  source          : {source}")
    print(f"  camera-id       : {args.camera_id}")
    print(f"  zone-id         : {args.zone_id}")
    print(f"  fps target      : {args.fps}")
    print(f"  inference       : {inference_mode}")
    print(f"  display window  : {'yes  (press q to quit)' if args.show else 'no'}")
    print(f"  anti-spoofing   : {'BYPASSED — dev mode' if args.bypass_antispoofing else 'enabled'}")
    print("=" * 62)
    print("\nPress Ctrl+C to stop.\n")

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        print("  macOS tip: check System Settings → Privacy → Camera and grant access to Terminal.")
        sys.exit(1)

    frame_count = 0
    total_faces = 0
    producer = FrameProducer(
        camera=camera,
        camera_id=args.camera_id,
        zone_id=args.zone_id,
        max_fps=args.fps,
    )

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count, total_faces
        frame_count += 1

        faces = app.get(frame)
        total_faces += len(faces)

        for face in faces:
            bbox = face.bbox.astype(int)
            det_score = float(face.det_score)
            print(
                f"[{meta.camera_id} | frame {meta.frame_id:>5}]  "
                f"bbox=[{bbox[0]:4},{bbox[1]:4},{bbox[2]:4},{bbox[3]:4}]  "
                f"conf={det_score:.3f}"
            )
            if args.show:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 0), 2)
                cv2.putText(
                    frame, f"{det_score:.2f}",
                    (bbox[0], max(bbox[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
                )

        if frame_count % 30 == 0 and not faces:
            print(f"  — {frame_count} frames, {total_faces} detections so far —")

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
        print(f"\nStopped — {frame_count} frames processed, {total_faces} face detections.")


if __name__ == "__main__":
    asyncio.run(main())
