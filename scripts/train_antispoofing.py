#!/usr/bin/env python3
"""
Fine-tune the CDCN++ liveness model on deployment-collected face crops (E02-S10).

Completes the software-only anti-spoofing pipeline:
    collect_antispoofing_data.py  →  train_antispoofing.py  →  CDCN++ checkpoint
                                                                 │
       the AntiSpoofingEnsemble activates CDCN++ once these weights exist
       (CDCNWrapper.load_from_checkpoint), upgrading liveness from the
       weight-free temporal gate alone to texture + temporal.

Reads the dataset written by scripts/collect_antispoofing_data.py:
    data/antispoofing/live/*.jpg    → label 1 (genuine face)
    data/antispoofing/spoof/*.jpg   → label 0 (photo / screen / mask)

and trains CDCNPlusPlus (src/antispoofing/cdcn.py) as a binary classifier,
saving the best-by-validation checkpoint in the {"model_state_dict": ...}
shape that load_cdcn_model expects.

Usage:
  python scripts/train_antispoofing.py
  python scripts/train_antispoofing.py --epochs 40 --batch-size 32
  python scripts/train_antispoofing.py --output models/weights/cdcn_pp.pt --sign

This is a from-scratch scaffold, not a turnkey production detector: CDCN++
needs thousands of varied live/spoof crops across lighting and attack media to
generalize. With a small in-house set it will overfit — treat a high val
accuracy here as "the pipeline works end to end," not "spoofing is solved."
Collect broadly, or warm-start from a public CASIA-SURF/OULU checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Default checkpoint location — under models/weights so it sits with the other
# weights and is covered by scripts/sign_models.py-style signing if desired.
DEFAULT_OUTPUT = ROOT / "models" / "weights" / "cdcn_pp.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune CDCN++ liveness model")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "antispoofing"),
                   help="Dataset root with live/ and spoof/ subdirs "
                        "(default: data/antispoofing)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help=f"Checkpoint output path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation (default: 0.2)")
    p.add_argument("--theta", type=float, default=0.7,
                   help="Central-difference mixing factor (default: 0.7)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--sign", action="store_true",
                   help="Sign the trained checkpoint into models/manifest.json "
                        "after saving, so startup verification covers it")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import cv2
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}")
        print("  torch and opencv-python are required to train.")
        sys.exit(1)

    from src.antispoofing.cdcn import INPUT_SIZE, CDCNPlusPlus

    data_dir = Path(args.data_dir)
    live_files = sorted((data_dir / "live").glob("*.jpg"))
    spoof_files = sorted((data_dir / "spoof").glob("*.jpg"))

    if not live_files or not spoof_files:
        print(f"ERROR: need both live/ and spoof/ crops under {data_dir}.")
        print(f"  found: live={len(live_files)}  spoof={len(spoof_files)}")
        print("  Collect first:")
        print("    python scripts/collect_antispoofing_data.py --label live  --auto --show")
        print("    python scripts/collect_antispoofing_data.py --label spoof --auto --show")
        sys.exit(1)

    samples = [(f, 1) for f in live_files] + [(f, 0) for f in spoof_files]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(samples)

    n_val = max(1, int(len(samples) * args.val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        print("ERROR: not enough samples to form a training split — collect more.")
        sys.exit(1)

    class CropDataset(Dataset):
        def __init__(self, items: list[tuple[Path, int]], train: bool) -> None:
            self.items = items
            self.train = train

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int):
            path, label = self.items[i]
            img = cv2.imread(str(path))
            if img is None:  # unreadable file — fall back to zeros, never crash
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.train:
                # Light augmentation only — the central-difference texture is
                # the signal, so don't distort it with heavy geometric warps.
                if rng.random() < 0.5:
                    img = img[:, ::-1, :]  # horizontal flip
                if rng.random() < 0.5:
                    scale = float(rng.uniform(0.85, 1.15))  # brightness jitter
                    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            return tensor, torch.tensor(float(label))

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)

    train_loader = DataLoader(
        CropDataset(train_samples, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False,
    )
    val_loader = DataLoader(
        CropDataset(val_samples, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    model = CDCNPlusPlus(theta=args.theta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    print(f"Training CDCN++ on {len(train_samples)} train / {len(val_samples)} val "
          f"crops (live={len(live_files)} spoof={len(spoof_files)}) on {device}")
    print("=" * 64)

    best_val_acc = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for crops, labels in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(crops)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * crops.size(0)
        train_loss = running / len(train_samples)

        # Validation accuracy at the 0.5 decision boundary.
        model.eval()
        correct = 0
        with torch.no_grad():
            for crops, labels in val_loader:
                crops, labels = crops.to(device), labels.to(device)
                preds = model(crops)
                correct += int(((preds >= 0.5).float() == labels).sum().item())
        val_acc = correct / len(val_samples)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "theta": args.theta,
                    "input_size": INPUT_SIZE,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                output,
            )
            flag = "  ← saved (best)"
        print(f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.3f}{flag}")

    print("=" * 64)
    print(f"✓ best val_acc={best_val_acc:.3f}  →  {output}")
    print("  Load it into the ensemble with:")
    print("    from src.antispoofing.ensemble import CDCNWrapper")
    print(f"    cdcn = CDCNWrapper.load_from_checkpoint('{output}')")

    if args.sign:
        from src.core.model_registry import DEFAULT_MANIFEST_PATH, ModelRegistry
        reg = ModelRegistry()
        if DEFAULT_MANIFEST_PATH.exists():
            reg.load_manifest(DEFAULT_MANIFEST_PATH)
        reg.sign("cdcn_pp", output)
        reg.save_manifest(DEFAULT_MANIFEST_PATH)
        print(f"✓ signed cdcn_pp into {DEFAULT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
