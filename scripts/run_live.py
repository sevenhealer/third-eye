#!/usr/bin/env python3
"""
Live camera feed — face detection and recognition.

Usage:
  # USB webcam (index 0)
  python scripts/run_live.py --source 0

  # IP camera via RTSP
  python scripts/run_live.py --source rtsp://user:pass@192.168.1.100:554/stream1

  # Display bounding boxes in a cv2 window
  python scripts/run_live.py --source 0 --show

  # Skip liveness check while anti-spoofing weights aren't loaded yet (DEV ONLY)
  python scripts/run_live.py --source 0 --bypass-antispoofing

  # Custom camera ID and zone
  python scripts/run_live.py --source 0 --camera-id cam_entrance --zone-id building_entrance

Prerequisites:
  1. python scripts/download_models.py
  2. .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy
  3. docker compose up -d postgres redis kafka  (needed only for gallery lookups)
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
    p.add_argument("--camera-id", default="cam0",
                   help="Logical camera ID written into FrameMeta")
    p.add_argument("--zone-id", default="entrance",
                   help="Zone this camera covers (written into FrameMeta)")
    p.add_argument("--fps", type=int, default=10,
                   help="Target processing frames per second (default 10)")
    p.add_argument("--show", action="store_true",
                   help="Open a cv2 window with bounding boxes")
    p.add_argument("--bypass-antispoofing", action="store_true",
                   help="Skip liveness check — DEV MODE ONLY, never use in production")
    return p.parse_args()


def _check_deps() -> None:
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python-headless")
    try:
        import insightface  # noqa: F401
    except ImportError:
        missing.append("insightface")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing.append("onnxruntime-gpu")
    if missing:
        print("ERROR: Missing dependencies:")
        for pkg in missing:
            print(f"  {pkg}")
        print("\nInstall with:")
        print(f"  .venv/bin/pip install {' '.join(missing)}")
        sys.exit(1)


def _check_weights(weights_dir: Path) -> None:
    buffalo = weights_dir / "buffalo_l"
    if not buffalo.exists() or not any(buffalo.iterdir()):
        print(f"ERROR: Model weights not found at {buffalo}")
        print("Run:  python scripts/download_models.py")
        sys.exit(1)


def _set_minimal_env() -> None:
    """Provide dummy env vars so Settings() loads without a real .env file."""
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

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    # Load InsightFace bundle (SCRFD-10G detector + ArcFace recognizer)
    print("Loading models ...", end=" ", flush=True)
    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("OK\n")

    print("=" * 60)
    print(f"  source            : {source}")
    print(f"  camera-id         : {args.camera_id}")
    print(f"  zone-id           : {args.zone_id}")
    print(f"  fps target        : {args.fps}")
    print(f"  display window    : {'yes' if args.show else 'no'}")
    print(f"  anti-spoofing     : {'BYPASSED — dev mode' if args.bypass_antispoofing else 'enabled'}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
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
            # normed_embedding present when recognition model is loaded
            has_id = hasattr(face, "normed_embedding") and face.normed_embedding is not None
            label = "FACE" if not has_id else "IDENTIFIED"

            print(
                f"[{meta.camera_id} | frame {meta.frame_id:>5}]  "
                f"bbox=[{bbox[0]:4},{bbox[1]:4},{bbox[2]:4},{bbox[3]:4}]  "
                f"conf={det_score:.3f}  "
                f"{label}"
            )

            if args.show:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 0), 2)
                cv2.putText(
                    frame, f"{det_score:.2f}",
                    (bbox[0], bbox[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
                )

        if frame_count % 30 == 0:
            print(f"  — {frame_count} frames processed, {total_faces} face detections total —")

        if args.show:
            cv2.imshow("third-eye | live", frame)
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
        print(f"\nStopped — {frame_count} frames, {total_faces} face detections.")


if __name__ == "__main__":
    asyncio.run(main())
