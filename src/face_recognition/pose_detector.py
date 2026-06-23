from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

_detector: Any = None


def get_pose_detector() -> Any:
    """Lazy-loaded insightface FaceAnalysis (buffalo_l) for the manual
    enrollment capture flow - same model run_live.py and scripts/enroll.py
    use, so face.normed_embedding lands in the same embedding space as the
    rest of the gallery. Not loaded at API startup: this is an occasional,
    operator-driven flow, not worth the GPU memory until first used."""
    global _detector
    if _detector is None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        from src.core.config import get_settings
        from src.core.gpu_manager import preload_cuda_libraries

        settings = get_settings()
        preload_cuda_libraries()
        ctx_id = 0 if "CUDAExecutionProvider" in ort.get_available_providers() else -1
        app = FaceAnalysis(name="buffalo_l", root=str(settings.weights_dir))
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _detector = app
        logger.info("manual_enroll_detector_loaded", ctx_id=ctx_id)
    return _detector
