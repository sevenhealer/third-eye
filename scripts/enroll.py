#!/usr/bin/env python3
"""
Enroll a new person into the face gallery from a live webcam.

Collects N face crops, computes the quality-weighted mean embedding,
and inserts it into the pgvector gallery via the third-eye API.

Usage:
  python scripts/enroll.py --name "John Doe"
  python scripts/enroll.py --name "Jane" --source 1 --crops 15
  python scripts/enroll.py --name "Rohan" --source rtsp://192.168.1.x:8554/stream

Prerequisites:
  - python scripts/download_models.py
  - docker compose up -d postgres redis
  - .venv/bin/pip install insightface opencv-python-headless onnxruntime-gpu numpy
"""
import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enroll a face into third-eye gallery")
    p.add_argument("--name", required=True, help="Person display name")
    p.add_argument("--source", default="0", help="Camera index or RTSP URL")
    p.add_argument("--crops", type=int, default=10,
                   help="Number of face crops to collect (default 10)")
    p.add_argument("--role", default="visitor", help="Person role in DB (default visitor)")
    p.add_argument("--new-person", action="store_true",
                   help="Create a separate person even if the name is already "
                        "enrolled (default: add this embedding to the existing person)")
    return p.parse_args()


# Pose-cell enrollment (FaceID-style): buckets over (yaw, pitch) fill
# automatically as the head moves through them — no keyboard interaction
# during capture, the frame loop never blocks, the preview stays live.
# insightface's landmark_3d_68 provides face.pose = [pitch, yaw, roll] (deg).
POSE_BUCKETS: list[tuple[str, str]] = [
    ("straight", "Look STRAIGHT at the camera"),
    ("left",     "Turn your head LEFT"),
    ("right",    "Turn your head RIGHT"),
    ("up",       "Tilt your head UP"),
    ("down",     "Tilt your head DOWN"),
]


def _pose_bucket(yaw: float, pitch: float) -> str | None:
    if abs(yaw) < 12 and abs(pitch) < 12:
        return "straight"
    if yaw <= -18:
        return "left"
    if yaw >= 18:
        return "right"
    if pitch >= 14:
        return "up"
    if pitch <= -14:
        return "down"
    return None   # between cells — too ambiguous to file anywhere


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

    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Run: .venv/bin/pip install insightface opencv-python-headless onnxruntime-gpu numpy")
        sys.exit(1)

    weights_dir = ROOT / "models" / "weights"
    if not (weights_dir / "models" / "buffalo_l").exists():
        print("ERROR: Run python scripts/download_models.py first")
        sys.exit(1)

    _set_minimal_env()

    # must run before any ONNX session is created, or the CUDA provider
    # fails to find libcublasLt/libcudnn and silently falls back to CPU
    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    try:
        import onnxruntime as ort
        ctx_id = 0 if "CUDAExecutionProvider" in ort.get_available_providers() else -1
    except ImportError:
        ctx_id = -1

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera '{source}'")
        if str(source).isdigit():
            print("No local webcam? Pass an RTSP stream: --source rtsp://...")
        sys.exit(1)

    # opencv-python-headless (typical on the Linux GPU box) has no GUI —
    # fall back to console-only progress instead of crashing on imshow
    win_name = "Enrollment — press q to abort"
    show_window = True
    try:
        cv2.namedWindow(win_name)
    except cv2.error:
        show_window = False
        print("(headless OpenCV build: no preview window, progress prints below; Ctrl+C to abort)")

    base, rem = divmod(args.crops, len(POSE_BUCKETS))
    need: dict[str, int] = {
        name: base + (1 if i < rem else 0)
        for i, (name, _) in enumerate(POSE_BUCKETS)
    }
    hints = dict(POSE_BUCKETS)

    print(f"\nEnrolling '{args.name}' — {args.crops} crops across "
          f"{len(POSE_BUCKETS)} head poses.")
    print("Move your head slowly: straight, left, right, up, down.")
    print("Crops are captured automatically when a pose is reached. "
          "Ctrl+C aborts.\n")

    embeddings: list[np.ndarray] = []
    deadline = time.monotonic() + 180.0
    last_capture = 0.0
    last_status = 0.0

    while any(need.values()):
        if time.monotonic() > deadline:
            print("Timed out after 3 min — check framing/lighting. Aborting.")
            cap.release()
            sys.exit(1)

        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            cap.release()
            sys.exit(1)

        faces = app.get(frame)
        # Use the largest detected face (closest to camera)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        ) if faces else None

        pending = [n for n, q in need.items() if q > 0]
        hint = hints[pending[0]]
        pose_txt = "no face"

        if (
            face is not None
            and face.normed_embedding is not None
            and getattr(face, "pose", None) is not None
        ):
            pitch, yaw = float(face.pose[0]), float(face.pose[1])
            pose_txt = f"yaw {yaw:+.0f}  pitch {pitch:+.0f}"
            bucket = _pose_bucket(yaw, pitch)
            if (
                bucket in pending
                and time.monotonic() - last_capture >= 0.4
            ):
                embeddings.append(face.normed_embedding.copy())
                need[bucket] -= 1
                last_capture = time.monotonic()
                print(f"  Crop {len(embeddings):>2}/{args.crops} captured "
                      f"({bucket}, yaw {yaw:+.0f}, pitch {pitch:+.0f}).")
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if show_window:
            cv2.putText(frame, f">>> {hint}", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
            cv2.putText(frame,
                        f"{len(embeddings)}/{args.crops}   {pose_txt}   "
                        f"pending: {', '.join(pending)}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Aborted.")
                sys.exit(0)
        elif time.monotonic() - last_status > 3.0:
            last_status = time.monotonic()
            print(f"  ... {hint}   ({pose_txt}; pending: {', '.join(pending)})")

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if len(embeddings) < args.crops:
        print(f"Only {len(embeddings)} crops collected (need {args.crops}). Aborting.")
        sys.exit(1)

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 1e-10:
        mean_emb /= norm

    print(f"\nComputed mean embedding from {len(embeddings)} crops.")
    print("Saving to gallery ...")

    from sqlalchemy import text

    from src.core.database import get_db_session
    from src.face_recognition.gallery import FaceGallery

    gallery = FaceGallery()
    try:
        async with get_db_session() as session:
            existing = (
                await session.execute(
                    text("SELECT person_id FROM persons WHERE display_name = :name LIMIT 1"),
                    {"name": args.name},
                )
            ).fetchone()

        if existing and not args.new_person:
            # re-enrollment: extra embeddings improve recognition of the SAME
            # person — a duplicate persons row would split their identity
            person_id = str(existing[0])
            print(f"'{args.name}' is already enrolled — adding this embedding "
                  f"to the same person ({person_id}).")
            print("(use --new-person if this is genuinely a different person)")
        else:
            person_id = str(uuid.uuid4())
            # persons row must exist first — face_gallery.person_id is a FK to it
            async with get_db_session() as session:
                await session.execute(
                    text(
                        "INSERT INTO persons (person_id, display_name, role) "
                        "VALUES (:pid, :name, :role)"
                    ),
                    {"pid": person_id, "name": args.name, "role": args.role},
                )
                await session.commit()
        print(f"person_id = {person_id}")
        eid = await gallery.add_embedding(
            person_id=person_id,
            embedding=mean_emb,
            quality_score=0.85,
            camera_id=str(source),
        )
        print(f"\nEnrolled '{args.name}' successfully!")
        print(f"  person_id    : {person_id}")
        print(f"  embedding_id : {eid}")
    except Exception as exc:
        print(f"\nERROR saving to gallery: {exc}")
        print("Is PostgreSQL running?  docker compose up -d postgres")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
