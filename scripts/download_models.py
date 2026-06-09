#!/usr/bin/env python3
"""
Download pretrained model weights for third-eye.

Downloads the InsightFace buffalo_l pack (~200 MB):
  - det_10g.onnx       SCRFD-10GF face detector
  - w600k_r50.onnx     ArcFace R50 recognition (512-dim embeddings)

GPU vs CPU is detected automatically from ONNX Runtime.

Mac install:
  .venv/bin/pip install insightface onnxruntime opencv-python numpy

Linux (RTX 3090) install:
  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --root /custom/path
  python scripts/download_models.py --cpu     # force CPU (no CUDA check)
"""
import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _onnx_ctx_id(force_cpu: bool = False) -> int:
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
    parser = argparse.ArgumentParser(description="Download third-eye model weights")
    parser.add_argument("--root", default=str(ROOT / "models" / "weights"),
                        help="Download directory (default: models/weights/)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode even if CUDA is available")
    args = parser.parse_args()

    try:
        import insightface  # noqa: F401
    except ImportError:
        on_mac = platform.system() == "Darwin"
        print("\nERROR: insightface is not installed.")
        if on_mac:
            print("Install with:\n  .venv/bin/pip install insightface onnxruntime opencv-python numpy")
        else:
            print("Install with:\n  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy")
        sys.exit(1)

    from insightface.app import FaceAnalysis

    weights_dir = Path(args.root)
    weights_dir.mkdir(parents=True, exist_ok=True)

    ctx_id = _onnx_ctx_id(force_cpu=args.cpu)
    mode = "CPU" if ctx_id == -1 else "GPU (CUDA)"
    print(f"Inference mode : {mode}")
    print(f"Download target: {weights_dir}/buffalo_l/")
    print("Downloading buffalo_l pack (~200 MB) ...\n")

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    buffalo_dir = weights_dir / "buffalo_l"
    if buffalo_dir.exists():
        files = sorted(buffalo_dir.iterdir())
        print(f"\nDownloaded {len(files)} file(s):")
        for f in files:
            print(f"  {f.name:<40} {f.stat().st_size // 1024:>6} KB")
    else:
        print("WARNING: buffalo_l directory not found — check insightface cache path.")

    print("\nDone. Run the live feed with:")
    if platform.system() == "Darwin":
        print("  python scripts/run_live.py --source 0 --show --bypass-antispoofing")
    else:
        print("  python scripts/run_live.py --source 0 --bypass-antispoofing")


if __name__ == "__main__":
    main()
