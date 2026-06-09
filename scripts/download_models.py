#!/usr/bin/env python3
"""
Download pretrained model weights required by third-eye.

Downloaded:
  buffalo_l pack (via InsightFace model zoo):
    - det_10g.onnx       SCRFD-10GF face detector
    - w600k_r50.onnx     ArcFace R50 recognition (512-dim embeddings)
    - genderage.onnx     age/gender (unused in pipeline, bundled)

Anti-spoofing weights (MiniFASNet-V2 + CDCN++) are downloaded separately
when Phase 2 fine-tuning is set up.

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --root /custom/path
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Download third-eye model weights")
    parser.add_argument(
        "--root",
        default=str(ROOT / "models" / "weights"),
        help="Directory to download weights into (default: models/weights/)",
    )
    args = parser.parse_args()

    weights_dir = Path(args.root)
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        import insightface  # noqa: F401
    except ImportError:
        print(
            "\nERROR: insightface is not installed.\n"
            "Install ML dependencies first:\n\n"
            "  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless\n"
        )
        sys.exit(1)

    from insightface.app import FaceAnalysis

    print(f"Downloading buffalo_l pack to {weights_dir}/buffalo_l/ ...")
    print("(~200 MB — this may take a few minutes on first run)\n")

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    # ctx_id=0 → GPU; ctx_id=-1 → CPU only
    app.prepare(ctx_id=0, det_size=(640, 640))

    buffalo_dir = weights_dir / "buffalo_l"
    if buffalo_dir.exists():
        files = sorted(buffalo_dir.iterdir())
        print(f"\nDownloaded {len(files)} file(s) to {buffalo_dir}:")
        for f in files:
            print(f"  {f.name:<35} {f.stat().st_size // 1024:>6} KB")
    else:
        print("WARNING: buffalo_l directory not found after download — check insightface cache.")

    print("\nAll models ready. You can now run:")
    print("  python scripts/run_live.py --source 0")


if __name__ == "__main__":
    main()
