from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import ModelLoadError
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import insightface as _insightface
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _insightface = None  # type: ignore[assignment]
    _INSIGHTFACE_AVAILABLE = False


@dataclasses.dataclass
class FaceDetection:
    """Single face detection result with quality metadata."""

    bbox: np.ndarray        # [x1, y1, x2, y2] float32
    score: float            # detector confidence [0, 1]
    kps: np.ndarray         # 5-point landmarks (5, 2) float32
    quality_score: float    # combined sharpness+size quality [0, 1]


def face_size(bbox: np.ndarray) -> int:
    """Minimum face dimension (width or height) in pixels."""
    return int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def laplacian_variance(img: np.ndarray) -> float:
    """
    Sharpness proxy via Laplacian variance.
    Returns 500.0 when cv2 is unavailable (assume acceptable quality).
    """
    if not _CV2_AVAILABLE:
        return 500.0
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(_cv2.Laplacian(gray, _cv2.CV_64F).var())


def compute_quality(crop: np.ndarray, bbox: np.ndarray) -> float:
    """
    Quality score [0, 1].
    Faces < 40 px on shortest side return 0.0.
    Sharpness is normalised: Laplacian variance 500 → score 1.0.
    """
    if face_size(bbox) < 40:
        return 0.0
    sharpness = laplacian_variance(crop)
    return min(sharpness / 500.0, 1.0)


# Faces below this quality are dropped before anti-spoofing
MIN_QUALITY_SCORE: float = 0.2


class SCRFDDetector:
    """
    SCRFD-10GF face detector via InsightFace.

    When insightface is unavailable (no weights / test env), detect() returns [].
    Quality filter runs before returning results — blurry or tiny faces are dropped.
    """

    def __init__(self, model_path: Path | None = None, det_thresh: float = 0.5) -> None:
        self.det_thresh = det_thresh
        self._model: Any = None
        if _INSIGHTFACE_AVAILABLE:
            self._load(model_path)

    def _load(self, model_path: Path | None) -> None:
        try:
            model = _insightface.model_zoo.get_model(
                "scrfd_10g_bnkps",
                root=str(model_path.parent) if model_path else None,
            )
            model.prepare(ctx_id=0, input_size=(640, 640), det_thresh=self.det_thresh)
            self._model = model
            logger.info("scrfd_loaded")
        except Exception as exc:
            raise ModelLoadError(f"SCRFD load failed: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect quality-filtered faces in a BGR frame (H×W×3 uint8)."""
        if self._model is None:
            return []

        bboxes, kpss = self._model.detect(frame)
        if bboxes is None or len(bboxes) == 0:
            return []

        h, w = frame.shape[:2]
        results: list[FaceDetection] = []
        for i, box_score in enumerate(bboxes):
            x1, y1, x2, y2, score = box_score
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            crop = frame[y1:y2, x1:x2]
            quality = compute_quality(crop, bbox)
            if quality < MIN_QUALITY_SCORE:
                logger.debug("face_quality_filtered", quality=quality)
                continue
            kps = kpss[i] if kpss is not None and i < len(kpss) else np.zeros((5, 2))
            results.append(FaceDetection(
                bbox=bbox,
                score=float(score),
                kps=kps.astype(np.float32),
                quality_score=quality,
            ))
        return results


_detector: SCRFDDetector | None = None


def get_detector() -> SCRFDDetector:
    global _detector
    if _detector is None:
        from src.core.config import get_settings
        settings = get_settings()
        path = settings.weights_dir / "scrfd_10g_bnkps.onnx"
        _detector = SCRFDDetector(model_path=path if path.exists() else None)
    return _detector
