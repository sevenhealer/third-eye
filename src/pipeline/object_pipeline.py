from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.object_detection.detector import ObjectDetection, YOLODetector
from src.pipeline.face_pipeline import FrameMeta
from src.tracking.tracker import ByteTracker, TrackedObject


@dataclass
class ObjectPipelineResult:
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray     # [x1, y1, x2, y2]
    confidence: float


class ObjectPipeline:
    """
    Layer 5 + 7: YOLODetector → ByteTracker.

    Produces per-frame list of confirmed tracked objects with stable track IDs.
    All dependencies injected — use mocks in tests.
    """

    def __init__(self, detector: YOLODetector, tracker: ByteTracker) -> None:
        self._detector = detector
        self._tracker = tracker

    def process_frame(
        self, frame: np.ndarray, meta: FrameMeta
    ) -> list[ObjectPipelineResult]:
        detections: list[ObjectDetection] = self._detector.detect(frame)
        tracked: list[TrackedObject] = self._tracker.update(detections)
        return [
            ObjectPipelineResult(
                track_id=t.track_id,
                class_id=t.class_id,
                class_name=t.class_name,
                bbox=t.bbox.copy(),
                confidence=t.confidence,
            )
            for t in tracked
        ]
