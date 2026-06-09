#!/usr/bin/env python3
"""
Anti-spoofing fine-tuning pipeline — CDCNPlusPlus on domain data.

Expected data layout (populated by scripts/collect_antispoofing_data.py):
  data/antispoofing/live/   — live face crops
  data/antispoofing/spoof/  — spoof presentations

Metrics reported:
  ACER (Attack Classification Error Rate) = (APCER + BPCER) / 2
    APCER — attack presentation classification error rate (false accept)
    BPCER — bona fide presentation classification error rate (false reject)

Usage:
  pip install -e ".[training]"
  python training/antispoofing/train.py
  python training/antispoofing/train.py --epochs 30 --lr 1e-4 --batch 32 --device cpu
  python training/antispoofing/train.py --resume models/weights/cdcn.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

LIVE_LABEL = 1
SPOOF_LABEL = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune CDCN++ anti-spoofing model")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "antispoofing"))
    p.add_argument("--output-dir", default=str(ROOT / "models" / "weights"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data reserved for validation (default 0.2)")
    p.add_argument("--theta", type=float, default=0.7,
                   help="CDC theta parameter (0=pure CDC, 1=standard conv, default 0.7)")
    p.add_argument("--device", default=None,
                   help="torch device string (default: auto-detect)")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--mlflow", action="store_true",
                   help="Log metrics to MLflow (requires mlflow package)")
    p.add_argument("--no-sign", action="store_true",
                   help="Skip SHA-256 signing of the output checkpoint")
    return p.parse_args()


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _build_transforms(train: bool):
    import torchvision.transforms as T
    if train:
        return T.Compose([
            T.Resize((288, 288)),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_dataset(data_dir: Path, val_split: float):
    """Return (train_dataset, val_dataset) from data_dir/{live,spoof}/."""
    from torch.utils.data import Dataset, random_split
    from PIL import Image

    class FaceDataset(Dataset):
        def __init__(self, items: list[tuple[Path, int]], transform) -> None:
            self.items = items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            path, label = self.items[idx]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    items: list[tuple[Path, int]] = []
    for label_name, label_val in [("live", LIVE_LABEL), ("spoof", SPOOF_LABEL)]:
        label_dir = data_dir / label_name
        if label_dir.exists():
            for p in sorted(label_dir.glob("*.jpg")):
                items.append((p, label_val))

    if not items:
        raise RuntimeError(
            f"No training images found in {data_dir}/{{live,spoof}}/\n"
            "Run: python scripts/collect_antispoofing_data.py --label live\n"
            "      python scripts/collect_antispoofing_data.py --label spoof"
        )

    train_t = _build_transforms(train=True)
    val_t = _build_transforms(train=False)

    n_val = max(1, int(len(items) * val_split))
    n_train = len(items) - n_val

    # Deterministic split (seed 42)
    import torch
    gen = torch.Generator().manual_seed(42)
    train_items, val_items = random_split(items, [n_train, n_val], generator=gen)

    return (
        FaceDataset(list(train_items), train_t),
        FaceDataset(list(val_items), val_t),
    )


def _compute_acer(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Compute ACER, APCER, and BPCER from numpy arrays."""
    import numpy as np
    y_pred = (y_pred_prob >= threshold).astype(int)
    spoof_mask = y_true == SPOOF_LABEL
    live_mask = y_true == LIVE_LABEL

    # APCER: spoofs classified as live (false accept rate)
    apcer = float(np.mean(y_pred[spoof_mask] == LIVE_LABEL)) if spoof_mask.any() else 0.0
    # BPCER: live classified as spoof (false reject rate)
    bpcer = float(np.mean(y_pred[live_mask] == SPOOF_LABEL)) if live_mask.any() else 0.0
    acer = (apcer + bpcer) / 2.0
    return {"acer": acer, "apcer": apcer, "bpcer": bpcer}


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: torch not found.  pip install -e '.[training]'")
        sys.exit(1)

    import numpy as np

    device = args.device or _auto_device()
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading data from {data_dir} …")
    train_ds, val_ds = _load_dataset(data_dir, args.val_split)
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.antispoofing.cdcn import CDCNPlusPlus
    model = CDCNPlusPlus(theta=args.theta).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Training setup ────────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("antispoofing_finetune")
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                "theta": args.theta, "device": device,
                "train_size": len(train_ds), "val_size": len(val_ds),
            })
        except ImportError:
            print("WARNING: mlflow not installed — skipping experiment tracking")

    best_acer = 1.0
    best_ckpt_path = output_dir / "cdcn_best.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.monotonic()
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(device))
                all_probs.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        metrics = _compute_acer(
            np.array(all_labels), np.array(all_probs),
        )
        elapsed = time.monotonic() - t0

        print(
            f"Epoch {epoch + 1:>3}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"ACER={metrics['acer']:.4f}  "
            f"APCER={metrics['apcer']:.4f}  "
            f"BPCER={metrics['bpcer']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if mlflow_run:
            import mlflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "acer": metrics["acer"],
                "apcer": metrics["apcer"],
                "bpcer": metrics["bpcer"],
            }, step=epoch)

        # Save best checkpoint
        if metrics["acer"] < best_acer:
            best_acer = metrics["acer"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "acer": best_acer,
                "theta": args.theta,
                "metrics": metrics,
            }, best_ckpt_path)
            print(f"  → new best checkpoint saved (ACER={best_acer:.4f})")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_ckpt = output_dir / "cdcn_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "acer": best_acer,
        "theta": args.theta,
    }, final_ckpt)
    print(f"\nFinal checkpoint: {final_ckpt}")
    print(f"Best checkpoint:  {best_ckpt_path}  (ACER={best_acer:.4f})")

    # ── SHA-256 signing ───────────────────────────────────────────────────────
    if not args.no_sign:
        manifest_path = output_dir / "manifest.json"
        from src.core.model_registry import ModelRegistry
        reg = ModelRegistry()
        if manifest_path.exists():
            reg.load_manifest(manifest_path)
        reg.sign("cdcn_best", best_ckpt_path)
        reg.sign("cdcn_final", final_ckpt)
        reg.save_manifest(manifest_path)
        print(f"SHA-256 manifest updated: {manifest_path}")

    if mlflow_run:
        import mlflow
        mlflow.log_artifact(str(best_ckpt_path))
        mlflow.log_metric("best_acer", best_acer)
        mlflow.end_run()

    # Write a simple training report
    report = {
        "best_acer": best_acer,
        "epochs_trained": args.epochs,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "theta": args.theta,
        "device": device,
    }
    (output_dir / "cdcn_train_report.json").write_text(json.dumps(report, indent=2))
    print(f"Training complete. Best ACER: {best_acer:.4f}")


if __name__ == "__main__":
    main()
