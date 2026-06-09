from __future__ import annotations

import dataclasses
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np

from src.core.exceptions import EnrollmentError
from src.core.logging import get_logger
from src.face_recognition.recognizer import EMBEDDING_DIM, l2_normalize

if TYPE_CHECKING:
    from src.face_recognition.recognizer import AdaFaceRecognizer

logger = get_logger(__name__)


@dataclasses.dataclass
class CropRecord:
    """One face crop collected during the enrollment window."""

    crop: np.ndarray
    kps: np.ndarray | None
    quality_score: float
    captured_at: float  # time.monotonic()


@dataclasses.dataclass
class EnrollmentCandidate:
    """
    A candidate ready for operator review after the enrollment window completes.

    The embedding is the quality-weighted mean of all collected crops,
    L2-normalised to unit length.
    """

    track_id: str
    person_id: str          # "UNKNOWN_<uuid4>" assigned at first detection
    embedding: np.ndarray   # 512-dim L2-normalised
    crop_count: int
    mean_quality: float
    captured_at: float      # monotonic timestamp of first crop


class EnrollmentBuffer:
    """
    Per-track-ID crop accumulator for the enrollment workflow.

    Workflow:
      1. Unrecognised person detected with anti-spoofing pass.
      2. Assign UNKNOWN_<uuid4> and call add_crop() on each subsequent frame.
      3. Once min_crops reached, finalize() computes quality-weighted mean embedding.
      4. Resulting EnrollmentCandidate is published for operator review.
      5. Operator assigns name → gallery.add_embedding() → tracks retroactively linked.

    Quality weighting: a crop with quality 0.9 contributes 9× more than quality 0.1.
    This pulls the mean toward sharp, well-lit face crops.
    """

    def __init__(
        self,
        recognizer: "AdaFaceRecognizer",
        min_crops: int = 10,
        window_seconds: float = 30.0,
    ) -> None:
        self._recognizer = recognizer
        self.min_crops = min_crops
        self.window_seconds = window_seconds
        self._buffers: dict[str, list[CropRecord]] = {}
        self._person_ids: dict[str, str] = {}
        self._window_start: dict[str, float] = {}

    def start(self, track_id: str) -> str:
        """
        Begin enrollment for a track. Returns the assigned UNKNOWN_<uuid4> person_id.
        Calling start() on an existing track is idempotent.
        """
        if track_id not in self._person_ids:
            self._person_ids[track_id] = f"UNKNOWN_{uuid.uuid4().hex[:8]}"
            self._buffers[track_id] = []
            self._window_start[track_id] = time.monotonic()
            logger.info("enrollment_started", track_id=track_id, person_id=self._person_ids[track_id])
        return self._person_ids[track_id]

    def add_crop(
        self, track_id: str, crop: np.ndarray, quality_score: float, kps: np.ndarray | None = None
    ) -> None:
        """
        Add a face crop to the buffer.

        Silently discards crops outside the active window.
        Call start() before add_crop().
        """
        if track_id not in self._buffers:
            return
        elapsed = time.monotonic() - self._window_start[track_id]
        if elapsed > self.window_seconds:
            logger.debug("enrollment_crop_outside_window", track_id=track_id, elapsed=elapsed)
            return
        self._buffers[track_id].append(
            CropRecord(crop=crop, kps=kps, quality_score=quality_score, captured_at=time.monotonic())
        )

    def is_ready(self, track_id: str) -> bool:
        """True when enough crops have been collected to finalize."""
        return len(self._buffers.get(track_id, [])) >= self.min_crops

    def finalize(self, track_id: str) -> EnrollmentCandidate:
        """
        Compute the quality-weighted mean embedding and return a candidate.

        Raises:
            EnrollmentError: Fewer crops than min_crops, or recognizer unavailable.
        """
        records = self._buffers.get(track_id, [])
        if len(records) < self.min_crops:
            raise EnrollmentError(
                f"Insufficient crops for {track_id}: {len(records)} < {self.min_crops}"
            )

        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        for rec in records:
            try:
                emb = self._recognizer.get_embedding(rec.crop, rec.kps)
                embeddings.append(emb)
                weights.append(rec.quality_score)
            except Exception as exc:
                logger.warning("enrollment_crop_embedding_failed", track_id=track_id, error=str(exc))

        if not embeddings:
            raise EnrollmentError(f"All {len(records)} crops failed embedding extraction for {track_id}")

        weights_arr = np.array(weights, dtype=np.float64)
        weight_sum = weights_arr.sum()
        if weight_sum < 1e-10:
            weights_arr = np.ones_like(weights_arr)
            weight_sum = float(len(weights_arr))

        stack = np.stack(embeddings, axis=0)             # (N, 512)
        weighted_mean = (stack * weights_arr[:, None]).sum(axis=0) / weight_sum
        final_embedding = l2_normalize(weighted_mean)

        candidate = EnrollmentCandidate(
            track_id=track_id,
            person_id=self._person_ids[track_id],
            embedding=final_embedding,
            crop_count=len(records),
            mean_quality=float(weights_arr.mean()),
            captured_at=self._window_start[track_id],
        )
        self._clear(track_id)
        logger.info(
            "enrollment_finalized",
            track_id=track_id,
            person_id=candidate.person_id,
            crops=candidate.crop_count,
            mean_quality=round(candidate.mean_quality, 3),
        )
        return candidate

    def _clear(self, track_id: str) -> None:
        self._buffers.pop(track_id, None)
        self._person_ids.pop(track_id, None)
        self._window_start.pop(track_id, None)

    def abandon(self, track_id: str) -> None:
        """Discard an enrollment in progress (track lost before min_crops reached)."""
        self._clear(track_id)
        logger.debug("enrollment_abandoned", track_id=track_id)
