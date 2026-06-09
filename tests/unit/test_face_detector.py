from unittest.mock import MagicMock

import numpy as np
import pytest

from src.face_detection.detector import (
    MIN_QUALITY_SCORE,
    FaceDetection,
    SCRFDDetector,
    compute_quality,
    face_size,
    laplacian_variance,
)


def test_detect_returns_empty_when_no_model():
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector._model = None
    detector.det_thresh = 0.5
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_face_size_uses_minimum_dimension():
    # 100×60 face → size = 60
    assert face_size(np.array([10.0, 20.0, 110.0, 80.0])) == 60


def test_compute_quality_small_face_returns_zero():
    # 30×30 face → below 40px threshold
    crop = np.ones((30, 30, 3), dtype=np.uint8) * 128
    bbox = np.array([0.0, 0.0, 30.0, 30.0])
    assert compute_quality(crop, bbox) == 0.0


def test_compute_quality_large_face_returns_positive():
    # Checkerboard crop has high Laplacian variance → quality > 0 with or without cv2
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[::2, :] = 255
    bbox = np.array([0.0, 0.0, 100.0, 100.0])
    quality = compute_quality(crop, bbox)
    assert quality > 0.0


def test_quality_filter_removes_small_faces():
    """Mock model returning a 20×20 face bbox → filtered before returning."""
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.det_thresh = 0.5
    mock_model = MagicMock()
    # Bbox: x1=10, y1=10, x2=30, y2=30 → 20px face → quality=0.0 → dropped
    mock_model.detect.return_value = (
        np.array([[10.0, 10.0, 30.0, 30.0, 0.95]]),
        np.zeros((1, 5, 2)),
    )
    detector._model = mock_model
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_quality_filter_passes_large_sharp_faces():
    """Mock model returning a 100×100 face bbox → passes quality filter."""
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.det_thresh = 0.5
    mock_model = MagicMock()
    # Bbox: 100×100 face → size=100 ≥ 40; laplacian fallback → quality > MIN_QUALITY_SCORE
    mock_model.detect.return_value = (
        np.array([[50.0, 50.0, 150.0, 150.0, 0.91]]),
        np.zeros((1, 5, 2)),
    )
    detector._model = mock_model
    # Checkerboard frame → high Laplacian variance → quality passes filter
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[::2, :] = 200
    results = detector.detect(frame)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, FaceDetection)
    assert r.score == pytest.approx(0.91)
    assert r.quality_score > 0.0


def test_detect_clips_bbox_to_frame_bounds():
    """Bbox extending outside frame is clipped to frame dimensions."""
    detector = SCRFDDetector.__new__(SCRFDDetector)
    detector.det_thresh = 0.5
    mock_model = MagicMock()
    # Bbox goes beyond 640×480 frame
    mock_model.detect.return_value = (
        np.array([[-10.0, -5.0, 700.0, 500.0, 0.88]]),
        np.zeros((1, 5, 2)),
    )
    detector._model = mock_model
    # Checkerboard frame → face crop has non-zero Laplacian → quality passes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[::2, :] = 180
    results = detector.detect(frame)
    assert len(results) == 1
    x1, y1, x2, y2 = results[0].bbox
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 640 and y2 <= 480
