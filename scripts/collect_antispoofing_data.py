#!/usr/bin/env python3
"""
Anti-spoofing domain data collection tool.

Captures labelled face crops from a live camera for fine-tuning anti-spoofing
models on deployment-specific conditions.

Output structure:
  data/antispoofing/live/{session}_{idx:05d}.jpg   — real faces
  data/antispoofing/spoof/{session}_{idx:05d}.jpg  — attack presentations

Usage:
  python scripts/collect_antispoofing_data.py --label live
  python scripts/collect_antispoofing_data.py --label spoof --source 1
  python scripts/collect_antispoofing_data.py --label live --auto --interval 0.5

Keys in interactive mode (requires --show):
  SPACE  — capture current frame
  q      — quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect anti-spoofing training data")
    p.add_argument("--label", required=True, choices=["live", "spoof"],
                   help="Ground-truth label for all captured frames in this session")
    p.add_argument("--source", default="0",
                   help="Camera index or RTSP URL (default: 0)")
    p.add_argument("--output-dir", default=str(ROOT / "data" / "antispoofing"),
                   help="Root output directory (default: data/antispoofing)")
    p.add_argument("--session", default=None,
                   help="Session name prefix (default: auto timestamp)")
    p.add_argument("--target", type=int, default=200,
                   help="Number of frames to capture (default: 200)")
    p.add_argument("--interval", type=float, default=0.1,
                   help="Seconds between auto-captures (default: 0.1)")
    p.add_argument("--auto", action="store_true",
                   help="Capture automatically at --interval instead of waiting for SPACE")
    p.add_argument("--show", action="store_true",
                   help="Display preview window (requires opencv-python, not headless)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only inference for face detection")
    return p.parse_args()


def _detect_inference(force_cpu: bool) -> int:
    if force_cpu:
        return -1
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return 0
    except ImportError:
        pass
    return -1


def main() -> None:
    args = parse_args()

    try:
        import cv2
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("  pip install opencv-python insightface onnxruntime")
        sys.exit(1)

    # Resolve session name
    session = args.session or time.strftime("%Y%m%d_%H%M%S")

    out_dir = Path(args.output_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files to continue numbering
    existing = len(list(out_dir.glob(f"{session}_*.jpg")))
    idx = existing

    # Load face detector
    weights_dir = ROOT / "models" / "weights"
    ctx_id = _detect_inference(args.cpu)
    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(320, 320))

    # Open camera
    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera: {source}")
        sys.exit(1)

    print(f"Collecting {args.target} '{args.label}' frames → {out_dir}")
    print("Press SPACE to capture, q to quit." if not args.auto
          else f"Auto-capturing every {args.interval}s. Press q to quit.")

    last_capture = 0.0

    while idx < args.target:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame read failed.")
            break

        faces = app.get(frame)
        display = frame.copy()

        # Draw bounding boxes
        for face in faces:
            b = face.bbox.astype(int)
            cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)

        cv2.putText(
            display,
            f"label={args.label}  captured={idx}/{args.target}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        if args.show:
            cv2.imshow("collect_antispoofing (q=quit, SPACE=capture)", display)

        now = time.monotonic()
        should_capture = (
            args.auto and (now - last_capture >= args.interval)
        )

        key = cv2.waitKey(1) & 0xFF if args.show else 0xFF

        if key == ord("q"):
            break

        if should_capture or (key == ord(" ")):
            if not faces:
                print(f"  frame {idx}: no face detected — skipping")
                continue

            face = faces[0]
            b = face.bbox.astype(int)
            # Clamp to frame bounds and add small padding
            pad = 16
            x1 = max(0, b[0] - pad)
            y1 = max(0, b[1] - pad)
            x2 = min(frame.shape[1], b[2] + pad)
            y2 = min(frame.shape[0], b[3] + pad)
            crop = frame[y1:y2, x1:x2]

            filename = out_dir / f"{session}_{idx:05d}.jpg"
            cv2.imwrite(str(filename), crop)
            idx += 1
            last_capture = now
            print(f"  [{idx:>4}/{args.target}] saved {filename.name}")

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nDone. {idx} '{args.label}' frames saved to {out_dir}")
    print(f"  Total '{args.label}' frames in dataset: "
          f"{len(list(out_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
