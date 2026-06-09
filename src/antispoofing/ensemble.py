from __future__ import annotations

import collections
import dataclasses
from typing import Any

import numpy as np

from src.core.exceptions import LowConfidenceError, SpoofingDetectedError
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# All models must score at or above this to accept a face as live
MIN_CONFIDENCE: float = 0.85


@dataclasses.dataclass
class SpoofingResult:
    """Output of the anti-spoofing ensemble."""

    is_live: bool
    confidence: float           # mean of model scores; < MIN_CONFIDENCE = denied
    minifas_score: float
    cdcn_score: float
    temporal_score: float
    rejected_by: str | None     # name of the model that triggered denial, or None


class MiniFASWrapper:
    """
    Wrapper for MiniFASNet-V2 — fast texture-based liveness screener (~3 ms/crop).

    When model weights are not loaded, score() returns 0.0 (fail-safe deny).
    In production, _model is an ONNX session or PyTorch module.
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

    def score(self, face_crop: np.ndarray) -> float:
        """
        Returns live probability [0, 1].
        0.0 → model not loaded (fail-safe).
        """
        if self._model is None:
            return 0.0
        return float(self._model(face_crop))


class CDCNWrapper:
    """
    Wrapper for CDCN++ — Central Difference Convolution for GAN/moire artifact
    detection (~11 ms/crop). Activated in Phase 2 when weights are available.

    Returns 0.0 (fail-safe deny) when not loaded.
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

    def score(self, face_crop: np.ndarray) -> float:
        """Returns live probability [0, 1]. 0.0 → model not loaded."""
        if self._model is None:
            return 0.0
        return float(self._model(face_crop))


class TemporalConsistencyChecker:
    """
    Liveness signal from frame-to-frame micro-movement variance.

    A printed photo or static screen replay shows near-zero variance.
    Score [0, 1]: low variance → low score → denied once buffer has >= WINDOW frames.
    Returns 0.5 (neutral) when fewer than 3 frames have been accumulated.
    """

    WINDOW: int = 10
    MINIMUM_FRAMES: int = 3

    def __init__(self) -> None:
        self._buffers: dict[str, collections.deque[float]] = {}
        self._prev_frames: dict[str, np.ndarray] = {}

    def update(self, track_id: str, frame: np.ndarray) -> None:
        """Push a new frame for the given track and record luminance diff."""
        buf = self._buffers.setdefault(track_id, collections.deque(maxlen=self.WINDOW))
        prev = self._prev_frames.get(track_id)
        if _CV2_AVAILABLE and prev is not None:
            gray_cur = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            gray_prev = prev
            diff = float(np.mean(np.abs(gray_cur.astype(np.float32) - gray_prev.astype(np.float32))))
            buf.append(diff / 255.0)
        else:
            buf.append(0.0)
        if _CV2_AVAILABLE:
            self._prev_frames[track_id] = (
                _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
            )

    def score(self, track_id: str) -> float:
        """
        Temporal liveness score [0, 1].
        Returns 0.5 (neutral) when < MINIMUM_FRAMES accumulated.
        Near-zero variance (still image) → score near 0.0.
        """
        buf = self._buffers.get(track_id)
        if not buf or len(buf) < self.MINIMUM_FRAMES:
            return 0.5
        variance = float(np.var(list(buf)))
        # Normalise: variance 1e-3 → fully live movement
        return min(variance / 1e-3, 1.0)

    def clear(self, track_id: str) -> None:
        self._buffers.pop(track_id, None)
        self._prev_frames.pop(track_id, None)


class AntiSpoofingEnsemble:
    """
    Hard-AND liveness ensemble: MiniFASNet-V2 + CDCN++ + TemporalConsistency.

    ALL active models must pass. Failing any single model → SpoofingDetectedError.
    Any model scoring in (0, MIN_CONFIDENCE) → LowConfidenceError (default-deny).
    A score of 0.0 means the model is not loaded; that also triggers denial.

    CDCN++ is wired but activates only when weights are present (Phase 2).
    Phase 1 uses MiniFASNet-V2 + temporal only.
    """

    TEMPORAL_LIVE_THRESHOLD: float = 0.1

    def __init__(
        self,
        minifas: MiniFASWrapper | None = None,
        cdcn: CDCNWrapper | None = None,
        temporal: TemporalConsistencyChecker | None = None,
        confidence_threshold: float = MIN_CONFIDENCE,
    ) -> None:
        self.minifas = minifas if minifas is not None else MiniFASWrapper()
        self.cdcn = cdcn if cdcn is not None else CDCNWrapper()
        self.temporal = temporal if temporal is not None else TemporalConsistencyChecker()
        self.confidence_threshold = confidence_threshold

    def _check_model(self, name: str, score: float) -> None:
        """Enforce confidence gate for a single model score."""
        if 0.0 < score < self.confidence_threshold:
            raise LowConfidenceError(
                f"{name} confidence {score:.3f} < {self.confidence_threshold} — default-deny"
            )
        if score <= 0.0:
            raise SpoofingDetectedError(f"{name} rejected (score={score:.3f}; model not loaded or spoof)")

    def check(
        self,
        face_crop: np.ndarray,
        track_id: str,
        raw_frame: np.ndarray | None = None,
    ) -> SpoofingResult:
        """
        Run the full liveness ensemble.

        Args:
            face_crop: Aligned face region (H×W×3 BGR).
            track_id:  Tracker-assigned ID for temporal state.
            raw_frame: Full camera frame (optional; improves temporal check).

        Returns:
            SpoofingResult with is_live=True if all checks pass.

        Raises:
            SpoofingDetectedError: Any model rejects the presentation.
            LowConfidenceError:    Any model score is in (0, threshold).
        """
        if raw_frame is not None:
            self.temporal.update(track_id, raw_frame)

        minifas_score = self.minifas.score(face_crop)
        cdcn_score = self.cdcn.score(face_crop)
        temporal_score = self.temporal.score(track_id)

        logger.debug(
            "antispoofing_scores",
            track_id=track_id,
            minifas=round(minifas_score, 4),
            cdcn=round(cdcn_score, 4),
            temporal=round(temporal_score, 4),
        )

        self._check_model("minifas", minifas_score)

        # CDCN++ is optional in Phase 1; only checked when its model is loaded
        if self.cdcn._model is not None:
            self._check_model("cdcn", cdcn_score)

        # Temporal: only act when enough frames accumulated
        buf = self.temporal._buffers.get(track_id, collections.deque())
        if len(buf) >= self.temporal.MINIMUM_FRAMES and temporal_score < self.TEMPORAL_LIVE_THRESHOLD:
            raise SpoofingDetectedError(
                f"Temporal consistency rejected (score={temporal_score:.3f}; static presentation)"
            )

        active_scores = [minifas_score]
        if self.cdcn._model is not None:
            active_scores.append(cdcn_score)
        ensemble_confidence = float(np.mean(active_scores))

        return SpoofingResult(
            is_live=True,
            confidence=ensemble_confidence,
            minifas_score=minifas_score,
            cdcn_score=cdcn_score,
            temporal_score=temporal_score,
            rejected_by=None,
        )


_ensemble: AntiSpoofingEnsemble | None = None


def get_ensemble() -> AntiSpoofingEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = AntiSpoofingEnsemble()
    return _ensemble
