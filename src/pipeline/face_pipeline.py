from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np

from src.core.exceptions import LowConfidenceError, SpoofingDetectedError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.antispoofing.ensemble import AntiSpoofingEnsemble
    from src.face_detection.detector import SCRFDDetector
    from src.face_recognition.enrollment import EnrollmentBuffer
    from src.face_recognition.gallery import FaceGallery
    from src.face_recognition.recognizer import AdaFaceRecognizer, RecognitionMatch
    from src.memory.short_term import ShortTermMemory

logger = get_logger(__name__)


@dataclasses.dataclass
class FrameMeta:
    """Metadata attached to each camera frame entering the pipeline."""

    camera_id: str
    frame_id: str
    timestamp_ns: int
    zone_id: str = ""


@dataclasses.dataclass
class FacePipelineResult:
    """Per-face result from one pipeline invocation."""

    track_id: str
    person_id: str | None       # None → unknown / enrollment pending
    decision: str               # "accept" | "ambiguous" | "reject" | "spoof" | "low_confidence"
    similarity: float           # 0.0 for unknowns
    quality_score: float
    bbox: np.ndarray            # [x1, y1, x2, y2]
    liveness_confidence: float  # from anti-spoofing ensemble


class FacePipeline:
    """
    Layers 2→3→4 processor: detect → anti-spoof → recognise.

    Dependencies are injected so the pipeline is testable without real models,
    Kafka, or databases. In production, get_face_pipeline() provides a
    fully-wired instance.

    One FacePipeline instance handles all cameras concurrently; track_ids must
    be globally unique (e.g. "{camera_id}_{local_track_id}").
    """

    def __init__(
        self,
        detector: "SCRFDDetector",
        ensemble: "AntiSpoofingEnsemble",
        recognizer: "AdaFaceRecognizer",
        gallery: "FaceGallery",
        stm: "ShortTermMemory",
        enrollment_buffer: "EnrollmentBuffer",
    ) -> None:
        self._detector = detector
        self._ensemble = ensemble
        self._recognizer = recognizer
        self._gallery = gallery
        self._stm = stm
        self._enrollment = enrollment_buffer

        # track_id → last seen time (for enrollment timeout cleanup)
        self._track_last_seen: dict[str, float] = {}

    async def process_frame(
        self,
        frame: np.ndarray,
        meta: FrameMeta,
    ) -> list[FacePipelineResult]:
        """
        Run the full face pipeline on one frame.

        1. Detect faces (SCRFD quality filter).
        2. Anti-spoofing liveness check per face.
        3. Recognition: gallery search → identity or enrollment candidate.
        4. Short-term memory update.

        Returns one FacePipelineResult per accepted face.
        Spoofed or low-quality detections produce a result with decision="spoof".
        """
        detections = self._detector.detect(frame)
        results: list[FacePipelineResult] = []

        for i, det in enumerate(detections):
            track_id = f"{meta.camera_id}_{meta.frame_id}_{i}"
            self._track_last_seen[track_id] = time.monotonic()

            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            face_crop = frame[y1:y2, x1:x2]

            # Layer 3: Anti-spoofing
            try:
                spoof_result = self._ensemble.check(
                    face_crop=face_crop,
                    track_id=track_id,
                    raw_frame=frame,
                )
            except SpoofingDetectedError as exc:
                logger.warning("spoof_rejected", track_id=track_id, reason=str(exc))
                results.append(FacePipelineResult(
                    track_id=track_id,
                    person_id=None,
                    decision="spoof",
                    similarity=0.0,
                    quality_score=det.quality_score,
                    bbox=det.bbox,
                    liveness_confidence=0.0,
                ))
                continue
            except LowConfidenceError as exc:
                logger.warning("liveness_low_confidence", track_id=track_id, reason=str(exc))
                results.append(FacePipelineResult(
                    track_id=track_id,
                    person_id=None,
                    decision="low_confidence",
                    similarity=0.0,
                    quality_score=det.quality_score,
                    bbox=det.bbox,
                    liveness_confidence=0.0,
                ))
                continue

            # Layer 4: Recognition
            person_id: str | None = None
            decision = "reject"
            similarity = 0.0
            try:
                embedding = self._recognizer.get_embedding(face_crop, det.kps)
                matches = await self._gallery.search(embedding, top_k=3)
                if matches:
                    best = matches[0]
                    decision = best.decision
                    similarity = best.similarity
                    if best.decision == "accept":
                        person_id = best.person_id
            except Exception as exc:
                logger.warning("recognition_failed", track_id=track_id, error=str(exc))
                decision = "reject"

            # Enrollment: queue unknown faces that pass liveness
            if person_id is None and decision != "spoof":
                assigned_id = self._enrollment.start(track_id)
                self._enrollment.add_crop(track_id, face_crop, det.quality_score, det.kps)
                person_id = assigned_id if not self._enrollment.is_ready(track_id) else None

            # Short-term memory update for known identities
            if person_id and not person_id.startswith("UNKNOWN_"):
                await self._stm.set_identity_location(person_id, meta.zone_id, track_id)

            results.append(FacePipelineResult(
                track_id=track_id,
                person_id=person_id,
                decision=decision,
                similarity=similarity,
                quality_score=det.quality_score,
                bbox=det.bbox,
                liveness_confidence=spoof_result.confidence,
            ))

        return results


_pipeline: FacePipeline | None = None


def get_face_pipeline() -> FacePipeline:
    global _pipeline
    if _pipeline is None:
        from src.antispoofing.ensemble import get_ensemble
        from src.face_detection.detector import get_detector
        from src.face_recognition.enrollment import EnrollmentBuffer
        from src.face_recognition.gallery import get_gallery
        from src.face_recognition.recognizer import get_recognizer
        from src.memory.short_term import get_short_term_memory
        from src.core.config import get_settings
        settings = get_settings()
        _pipeline = FacePipeline(
            detector=get_detector(),
            ensemble=get_ensemble(),
            recognizer=get_recognizer(),
            gallery=get_gallery(),
            stm=get_short_term_memory(),
            enrollment_buffer=EnrollmentBuffer(
                recognizer=get_recognizer(),
                min_crops=settings.enrollment_min_crops,
                window_seconds=settings.enrollment_window_seconds,
            ),
        )
    return _pipeline
