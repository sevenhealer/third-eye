from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.antispoofing.ensemble import SpoofingResult
from src.core.exceptions import LowConfidenceError, SpoofingDetectedError
from src.face_detection.detector import FaceDetection
from src.face_recognition.recognizer import RecognitionMatch
from src.pipeline.face_pipeline import FacePipeline, FrameMeta


# ── factories ──────────────────────────────────────────────────────────────────

def make_detection(x1=50, y1=50, x2=150, y2=150, score=0.92, quality=0.8) -> FaceDetection:
    return FaceDetection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        score=score,
        kps=np.zeros((5, 2), dtype=np.float32),
        quality_score=quality,
    )


def live_result(confidence: float = 0.91) -> SpoofingResult:
    return SpoofingResult(
        is_live=True,
        confidence=confidence,
        minifas_score=confidence,
        cdcn_score=0.0,
        temporal_score=0.8,
        rejected_by=None,
    )


def make_pipeline(
    detections=None,
    spoof_raises=None,
    gallery_matches=None,
    embedding=None,
) -> FacePipeline:
    detector = MagicMock()
    detector.detect.return_value = detections if detections is not None else []

    ensemble = MagicMock()
    if spoof_raises:
        ensemble.check.side_effect = spoof_raises
    else:
        ensemble.check.return_value = live_result()

    recognizer = MagicMock()
    if embedding is not None:
        recognizer.get_embedding.return_value = embedding
    else:
        recognizer.get_embedding.return_value = np.ones(512) / np.sqrt(512)

    gallery = AsyncMock()
    gallery.search.return_value = gallery_matches if gallery_matches is not None else []

    stm = AsyncMock()
    stm.set_identity_location = AsyncMock()

    enrollment = MagicMock()
    enrollment.start.return_value = "UNKNOWN_abc123"
    enrollment.is_ready.return_value = False

    return FacePipeline(
        detector=detector,
        ensemble=ensemble,
        recognizer=recognizer,
        gallery=gallery,
        stm=stm,
        enrollment_buffer=enrollment,
    )


frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
meta = FrameMeta(camera_id="cam1", frame_id="f001", timestamp_ns=0, zone_id="room_a")


# ── tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_detections_returns_empty():
    pipeline = make_pipeline(detections=[])
    results = await pipeline.process_frame(frame, meta)
    assert results == []


@pytest.mark.asyncio
async def test_spoof_rejection_produces_spoof_decision():
    pipeline = make_pipeline(
        detections=[make_detection()],
        spoof_raises=SpoofingDetectedError("printed photo"),
    )
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 1
    assert results[0].decision == "spoof"
    assert results[0].person_id is None


@pytest.mark.asyncio
async def test_low_confidence_produces_low_confidence_decision():
    pipeline = make_pipeline(
        detections=[make_detection()],
        spoof_raises=LowConfidenceError("score below threshold"),
    )
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 1
    assert results[0].decision == "low_confidence"


@pytest.mark.asyncio
async def test_known_person_accepted_from_gallery():
    match = RecognitionMatch(
        embedding_id="eid1",
        person_id="person-uuid-1",
        similarity=0.55,
        decision="accept",
    )
    pipeline = make_pipeline(
        detections=[make_detection()],
        gallery_matches=[match],
    )
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 1
    assert results[0].decision == "accept"
    assert results[0].person_id == "person-uuid-1"
    assert results[0].similarity == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_unknown_person_triggers_enrollment():
    pipeline = make_pipeline(
        detections=[make_detection()],
        gallery_matches=[],   # no gallery hit → unknown
    )
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 1
    result = results[0]
    # Enrollment started, person_id assigned as UNKNOWN_...
    pipeline._enrollment.start.assert_called_once()
    pipeline._enrollment.add_crop.assert_called_once()
    assert result.person_id == "UNKNOWN_abc123"


@pytest.mark.asyncio
async def test_recognition_failure_defaults_to_reject():
    pipeline = make_pipeline(detections=[make_detection()])
    pipeline._recognizer.get_embedding.side_effect = RuntimeError("model error")
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 1
    assert results[0].decision == "reject"


@pytest.mark.asyncio
async def test_known_person_updates_short_term_memory():
    match = RecognitionMatch(
        embedding_id="eid1",
        person_id="person-uuid-known",
        similarity=0.60,
        decision="accept",
    )
    pipeline = make_pipeline(
        detections=[make_detection()],
        gallery_matches=[match],
    )
    await pipeline.process_frame(frame, meta)
    pipeline._stm.set_identity_location.assert_called_once_with(
        "person-uuid-known", "room_a", pipeline._stm.set_identity_location.call_args[0][2]
    )


@pytest.mark.asyncio
async def test_multiple_faces_produce_multiple_results():
    pipeline = make_pipeline(
        detections=[make_detection(50, 50, 150, 150), make_detection(200, 50, 300, 150)],
        gallery_matches=[],
    )
    results = await pipeline.process_frame(frame, meta)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_liveness_confidence_propagated_to_result():
    pipeline = make_pipeline(detections=[make_detection()])
    pipeline._ensemble.check.return_value = live_result(confidence=0.94)
    results = await pipeline.process_frame(frame, meta)
    assert results[0].liveness_confidence == pytest.approx(0.94)
