#!/usr/bin/env python3
"""
Fine-tune YOLO26 on a domain dataset (Sprint 3, training) — the permanent
fix for misclassification: a model that knows YOUR almirah, iPad and shelves.

Usage:
  python scripts/train_detector.py --data datasets/room/data.yaml --epochs 100
  python scripts/train_detector.py --data datasets/room/data.yaml \
      --base yolo26m.pt --imgsz 960 --name room_v1

After training, point detection at the result:
  configs/models/object_detection.yaml -> custom_classes.weight_file, or
  scripts/run_objects.py --model runs/detect/room_v1/weights/best.pt
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="third-eye detector fine-tuning")
    p.add_argument("--data", required=True, help="Path to dataset data.yaml")
    p.add_argument("--base", default="yolo26m.pt",
                   help="Base checkpoint to fine-tune (default yolo26m.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=-1,
                   help="Batch size (-1 = auto-fit to VRAM)")
    p.add_argument("--name", default="thirdeye", help="Run name under runs/detect/")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        # CUDA's default device order (FASTEST_FIRST) doesn't match
        # nvidia-smi's PCI-bus-order index on this box — pin PCI_BUS_ID so
        # --gpu N always means nvidia-smi's GPU N (see run_live.py).
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset not found: {data_path}")
        print("Create one first: scripts/capture_dataset.py --out <dir> --vocab ...")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install -U ultralytics")
        sys.exit(1)

    device = "cpu"
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
        except ImportError:
            pass

    print(f"Fine-tuning {args.base} on {data_path} "
          f"({args.epochs} epochs, imgsz {args.imgsz}, device {device})\n")
    model = YOLO(args.base)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
    )
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Use them:  python scripts/run_objects.py --model {best} --source <rtsp>")


if __name__ == "__main__":
    main()
