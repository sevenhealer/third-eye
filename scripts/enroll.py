#!/usr/bin/env python3
"""
Enroll a new person into the face gallery from a live webcam.

Collects N face crops, computes the quality-weighted mean embedding,
and inserts it into the pgvector gallery via the third-eye API.

Usage:
  python scripts/enroll.py --name "John Doe"
  python scripts/enroll.py --name "Jane" --source 1 --crops 15

Prerequisites:
  - python scripts/download_models.py
  - docker compose up -d postgres redis
  - .venv/bin/pip install insightface opencv-python-headless onnxruntime-gpu numpy
"""
import argparse
import asyncio
import os
import sys
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
    if not (weights_dir / "buffalo_l").exists():
        print("ERROR: Run python scripts/download_models.py first")
        sys.exit(1)

    _set_minimal_env()

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=0, det_size=(640, 640))

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera '{source}'")
        sys.exit(1)

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

        cv2.imshow("Enrollment — press q to abort", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Aborted.")
            sys.exit(0)

    cap.release()
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

    from src.face_recognition.gallery import FaceGallery
    gallery = FaceGallery()
    try:
        eid = await gallery.add_embedding(
            person_id=person_id,
            embedding=mean_emb,
            quality_score=0.85,
            camera_id=str(source),
        )
        print(f"\nEnrolled '{args.name}' successfully!")
        print(f"  embedding_id : {eid}")
        print(f"\nRegister the display name (run in psql):")
        print(
            f"  INSERT INTO persons (person_id, display_name, role, enrolled_at)\n"
            f"  VALUES ('{person_id}', '{args.name}', '{args.role}', now());"
        )
    except Exception as exc:
        print(f"\nERROR saving to gallery: {exc}")
        print("Is PostgreSQL running?  docker compose up -d postgres")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
