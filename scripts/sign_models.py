#!/usr/bin/env python3
"""
Sign all present model weight files into models/manifest.json (E08-S07).

Computes a SHA-256 for every weight file the platform loads and writes them to a
manifest that the API verifies at every startup (`startup_verify`). Any later
mismatch — tampering or corruption — makes the server fail closed.

Run this ONCE after downloading/training weights (the Makefile `bootstrap`
target calls it automatically after `download_models`). Re-running re-signs
whatever is present and overwrites the manifest, so it is safe to repeat after a
deliberate weights update.

Signs (only what exists — missing files are skipped with a note):
  - InsightFace buffalo_l pack:  models/weights/models/buffalo_l/*.onnx
  - YOLO detectors at repo root: yolo26m.pt, yolov8x-worldv2.pt, yolov9c.pt

Usage:
  python scripts/sign_models.py
  python scripts/sign_models.py --manifest /custom/manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.model_registry import DEFAULT_MANIFEST_PATH, ModelRegistry  # noqa: E402

# name → path. Names are stable identifiers used in the manifest and logs;
# paths are repo-relative so they resolve natively and in the container.
BUFFALO_DIR = ROOT / "models" / "weights" / "models" / "buffalo_l"
WEIGHTS: dict[str, Path] = {
    "buffalo_l/1k3d68": BUFFALO_DIR / "1k3d68.onnx",
    "buffalo_l/2d106det": BUFFALO_DIR / "2d106det.onnx",
    "buffalo_l/det_10g": BUFFALO_DIR / "det_10g.onnx",
    "buffalo_l/genderage": BUFFALO_DIR / "genderage.onnx",
    "buffalo_l/w600k_r50": BUFFALO_DIR / "w600k_r50.onnx",
    "yolo26m": ROOT / "yolo26m.pt",
    "yolov8x-worldv2": ROOT / "yolov8x-worldv2.pt",
    "yolov9c": ROOT / "yolov9c.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign model weights into a manifest")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Manifest output path (default: {DEFAULT_MANIFEST_PATH})",
    )
    args = parser.parse_args()

    registry = ModelRegistry()
    signed = 0
    skipped: list[str] = []

    for name, path in WEIGHTS.items():
        if not path.exists():
            skipped.append(name)
            continue
        entry = registry.sign(name, path)
        print(f"  signed {name:<24} {entry.sha256[:16]}…  ({path.stat().st_size // 1024} KB)")
        signed += 1

    if signed == 0:
        print("\nERROR: no weight files found to sign. Run scripts/download_models.py first.")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    registry.save_manifest(manifest_path)

    print(f"\n✓ signed {signed} model(s) → {manifest_path}")
    if skipped:
        print(f"  ({len(skipped)} not present, skipped: {', '.join(skipped)})")
    print("  The API verifies these at every startup and fails closed on mismatch.")


if __name__ == "__main__":
    main()
