#!/usr/bin/env python3
"""
Capture a domain training dataset from a live camera (Sprint 3, training).

Grabs frames at an interval and, with --vocab, AUTO-LABELS them using
YOLO-World — turning your own footage into a YOLO-format dataset for
fine-tuning. Auto-labels are a starting point: review/correct them (Roboflow,
labelImg, CVAT) before training for "best-system" accuracy.

Output layout (Ultralytics-standard):
  <out>/images/train/*.jpg
  <out>/labels/train/*.txt        # "<class_id> cx cy w h" normalized
  <out>/data.yaml

Usage:
  # auto-labeled capture of your room objects:
  python scripts/capture_dataset.py --source rtsp://... \
      --vocab "person,almirah,ipad,water bottle,books,shoes" \
      --out datasets/room --every 1.0 --max 300

  # raw frames only (label them yourself later):
  python scripts/capture_dataset.py --source rtsp://... --out datasets/room --max 300
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="third-eye dataset capture + auto-label")
    p.add_argument("--source", default="0", help="Camera index or RTSP URL")
    p.add_argument("--out", required=True, help="Dataset output directory")
    p.add_argument("--vocab", default="",
                   help="Comma-separated classes for YOLO-World auto-labeling. "
                        "Omit to save raw frames only.")
    p.add_argument("--every", type=float, default=1.0,
                   help="Seconds between captured frames (default 1.0)")
    p.add_argument("--max", type=int, default=300,
                   help="Stop after this many frames (default 300)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Auto-label confidence floor (lower than live: a human "
                        "reviews these, recall matters more than precision)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv not installed")
        sys.exit(1)

    from src.ingestion.camera import CameraReader

    vocab = [c.strip() for c in args.vocab.split(",") if c.strip()] or None

    out = Path(args.out)
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if vocab:
        device = "cpu"
        if not args.cpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"
            except ImportError:
                pass
        from src.object_detection.detector import ModelNotLoadedError, YOLODetector
        detector = YOLODetector(conf_threshold=args.conf, imgsz=args.imgsz,
                                device=device, vocab=vocab)
        try:
            detector.load()
        except ModelNotLoadedError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        # write data.yaml so the dataset is immediately trainable
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(vocab))
        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/train\n"
            f"names:\n{names}\n"
        )

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        sys.exit(1)

    print(f"Capturing to {out}/  (mode: "
          f"{'auto-label ' + str(len(vocab)) + ' classes' if vocab else 'raw frames'})")
    print(f"every {args.every}s, up to {args.max} frames. Ctrl+C to stop.\n")

    saved = 0
    labeled_boxes = 0
    try:
        while saved < args.max:
            ok, frame = camera.read_frame()
            if not ok:
                print("Camera read failed.")
                break

            stamp = f"{int(time.time() * 1000)}"
            cv2.imwrite(str(img_dir / f"{stamp}.jpg"), frame)

            if detector is not None:
                h, w = frame.shape[:2]
                lines = []
                for det in detector.detect(frame):
                    x1, y1, x2, y2 = det.bbox
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    labeled_boxes += 1
                (lbl_dir / f"{stamp}.txt").write_text("\n".join(lines))

            saved += 1
            if saved % 10 == 0:
                print(f"  {saved}/{args.max} frames"
                      f"{f', {labeled_boxes} boxes' if detector else ''}")
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()

    print(f"\nDone — {saved} images in {img_dir}")
    if detector is not None:
        print(f"Auto-labeled {labeled_boxes} boxes into {lbl_dir}")
        print("NEXT: review/correct labels (Roboflow / labelImg / CVAT), then:")
        print(f"  python scripts/train_detector.py --data {out}/data.yaml")
    else:
        print("NEXT: label these images, then train with scripts/train_detector.py")


if __name__ == "__main__":
    main()
