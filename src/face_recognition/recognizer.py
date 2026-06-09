from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import ModelLoadError
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import insightface as _insightface
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _insightface = None  # type: ignore[assignment]
    _INSIGHTFACE_AVAILABLE = False

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


# AdaFace R100 produces 512-dimensional embeddings
EMBEDDING_DIM: int = 512

# Cosine similarity thresholds (from config defaults)
ACCEPT_THRESHOLD: float = 0.45
REJECT_THRESHOLD: float = 0.35


@dataclasses.dataclass
class RecognitionMatch:
    """Result of a gallery search for a single query embedding."""

    embedding_id: str
    person_id: str
    similarity: float           # cosine similarity [−1, 1]; higher = more similar
    decision: str               # "accept" | "ambiguous" | "reject"

    @classmethod
    def decide(cls, embedding_id: str, person_id: str, similarity: float) -> "RecognitionMatch":
        if similarity >= ACCEPT_THRESHOLD:
            decision = "accept"
        elif similarity >= REJECT_THRESHOLD:
            decision = "ambiguous"
        else:
            decision = "reject"
        return cls(
            embedding_id=embedding_id,
            person_id=person_id,
            similarity=similarity,
            decision=decision,
        )


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector to unit length. Safe for zero vectors."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class AdaFaceRecognizer:
    """
    AdaFace R100 face recognition model.

    Extracts 512-dim L2-normalized embeddings from aligned 112×112 face crops.
    When model weights are unavailable (dev/test env), raises ModelLoadError.

    The adaptive margin in AdaFace explicitly optimises for low-quality images
    (blur, partial occlusion) — the dominant condition in real surveillance.
    """

    INPUT_SIZE: tuple[int, int] = (112, 112)

    def __init__(self, model_path: Path | None = None) -> None:
        self._model: Any = None
        if _INSIGHTFACE_AVAILABLE and model_path and model_path.exists():
            self._load(model_path)

    def _load(self, model_path: Path) -> None:
        try:
            model = _insightface.model_zoo.get_model(str(model_path))
            model.prepare(ctx_id=0)
            self._model = model
            logger.info("adaface_loaded", path=str(model_path))
        except Exception as exc:
            raise ModelLoadError(f"AdaFace load failed: {exc}") from exc

    def _align(self, face_crop: np.ndarray, kps: np.ndarray | None) -> np.ndarray:
        """Align and resize face crop to INPUT_SIZE using 5-point keypoints."""
        if _CV2_AVAILABLE and kps is not None and kps.shape == (5, 2):
            from insightface.utils import face_align  # type: ignore[import]
            aligned = face_align.norm_crop(face_crop, landmark=kps, image_size=self.INPUT_SIZE[0])
            return aligned
        if _CV2_AVAILABLE:
            return _cv2.resize(face_crop, self.INPUT_SIZE)
        return face_crop

    def get_embedding(
        self, face_crop: np.ndarray, kps: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Extract L2-normalized 512-dim embedding.

        Raises:
            ModelLoadError: When model weights are not loaded.
        """
        if self._model is None:
            raise ModelLoadError("AdaFace model not loaded — weights missing or insightface unavailable")
        aligned = self._align(face_crop, kps)
        embedding = self._model.get_feat(aligned)
        return l2_normalize(embedding.flatten())


_recognizer: AdaFaceRecognizer | None = None


def get_recognizer() -> AdaFaceRecognizer:
    global _recognizer
    if _recognizer is None:
        from src.core.config import get_settings
        settings = get_settings()
        path = settings.weights_dir / "adaface_r100_ms1mv3.onnx"
        _recognizer = AdaFaceRecognizer(model_path=path if path.exists() else None)
    return _recognizer
