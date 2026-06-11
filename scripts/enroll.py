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
    return p.parse_args()


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

    print(f"\nEnrolling '{args.name}' — collecting {args.crops} crops.")
    print("Center your face in the frame. Press 'q' to abort.\n")

    embeddings: list[np.ndarray] = []

    while len(embeddings) < args.crops:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        faces = app.get(frame)
        # Use the largest detected face (closest to camera)
        if faces:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            if face.normed_embedding is not None:
                embeddings.append(face.normed_embedding.copy())
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Collected {len(embeddings)}/{args.crops}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                print(f"  Crop {len(embeddings):>2}/{args.crops} captured.")

        if show_window:
            cv2.imshow(win_name, frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Aborted.")
                sys.exit(0)
        else:
            time.sleep(0.1)   # same pacing waitKey(100) provided

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

    person_id = str(uuid.uuid4())
    print(f"\nComputed mean embedding from {len(embeddings)} crops.")
    print(f"person_id = {person_id}")
    print("Saving to gallery ...")

    from sqlalchemy import text

    from src.core.database import get_db_session
    from src.face_recognition.gallery import FaceGallery

    gallery = FaceGallery()
    try:
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
