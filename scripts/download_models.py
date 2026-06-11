#!/usr/bin/env python3
"""
Download pretrained model weights for third-eye (~200 MB, one-time).

Downloads InsightFace buffalo_l pack:
  - det_10g.onnx       SCRFD-10GF face detector
  - w600k_r50.onnx     ArcFace R50 recognition (512-dim embeddings)

Install before running:
  macOS M-series:
    brew install cmake
    .venv/bin/pip install cython scikit-build-core
    .venv/bin/pip install insightface onnxruntime opencv-python numpy

  Linux (CUDA):
    .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --root /custom/path
  python scripts/download_models.py --cpu
"""
import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ON_MAC = platform.system() == "Darwin"
ON_APPLE_SILICON = ON_MAC and platform.machine() == "arm64"


def _detect_ctx_id(force_cpu: bool) -> tuple[int, str]:
    if force_cpu:
        return -1, "CPU (forced)"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return 0, "GPU (CUDA)"
        if "CoreMLExecutionProvider" in providers and ON_APPLE_SILICON:
            return -1, "CPU + CoreML (Apple Neural Engine)"
    except ImportError:
        pass
    return -1, "CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download third-eye model weights")
    parser.add_argument("--root", default=str(ROOT / "models" / "weights"),
                        help="Download directory (default: models/weights/)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (skip CUDA/CoreML check)")
    args = parser.parse_args()

    try:
        import insightface  # noqa: F401
    except ImportError:
        print("\nERROR: insightface is not installed.")
        if ON_APPLE_SILICON:
            print("For Apple Silicon (M1/M2/M3/M4):\n")
            print("  brew install cmake")
            print("  .venv/bin/pip install cython scikit-build-core")
            print("  .venv/bin/pip install insightface onnxruntime opencv-python numpy")
        else:
            print("  .venv/bin/pip install insightface onnxruntime-gpu opencv-python-headless numpy")
        sys.exit(1)

    from insightface.app import FaceAnalysis

    # must run before any ONNX session is created, or the CUDA provider
    # fails to find libcublasLt/libcudnn and silently falls back to CPU
    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    ctx_id, mode_label = _detect_ctx_id(force_cpu=args.cpu)
    weights_dir = Path(args.root)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Platform       : {platform.system()} {platform.machine()}")
    print(f"Inference mode : {mode_label}")
    print(f"Download target: {weights_dir}/models/buffalo_l/")
    print("Downloading buffalo_l pack (~200 MB) ...\n")

    app = FaceAnalysis(name="buffalo_l", root=str(weights_dir))
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # insightface unpacks under <root>/models/<pack-name>/
    buffalo_dir = weights_dir / "models" / "buffalo_l"
    if buffalo_dir.exists():
        files = sorted(buffalo_dir.iterdir())
        print(f"\nDownloaded {len(files)} file(s):")
        for f in files:
            print(f"  {f.name:<40} {f.stat().st_size // 1024:>6} KB")
    else:
        print("WARNING: buffalo_l directory not found — insightface may cache elsewhere.")

    print("\nDone. Run the live feed:")
    if ON_APPLE_SILICON:
        print("  python scripts/run_live.py --source 0 --show --bypass-antispoofing")
    else:
        print("  python scripts/run_live.py --source 0 --bypass-antispoofing")


if __name__ == "__main__":
    main()
