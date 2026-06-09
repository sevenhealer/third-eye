from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.camera import CameraReader


def make_cv2(opened: bool = True, read_ok: bool = True) -> MagicMock:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (read_ok, frame if read_ok else None)
    cap.get.return_value = 1920.0
    cap.getBackendName.return_value = "V4L2"

    cv2 = MagicMock()
    cv2.VideoCapture.return_value = cap
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.CAP_PROP_FPS = 5
    return cv2


# ── is_opened ─────────────────────────────────────────────────────────────────

def test_is_opened_false_before_open():
    reader = CameraReader(source=0)
    assert reader.is_opened() is False


def test_is_opened_true_after_successful_open():
    mock_cv2 = make_cv2(opened=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        reader.open()
    assert reader.is_opened() is True


# ── open ──────────────────────────────────────────────────────────────────────

def test_open_returns_false_when_cv2_unavailable():
    reader = CameraReader(source=0)
    with patch("src.ingestion.camera._CV2_AVAILABLE", False):
        result = reader.open()
    assert result is False


def test_open_returns_false_when_cap_not_opened():
    mock_cv2 = make_cv2(opened=False)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        result = reader.open()
    assert result is False
    assert reader.is_opened() is False


def test_open_returns_true_on_success():
    mock_cv2 = make_cv2(opened=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        result = reader.open()
    assert result is True


# ── read_frame ────────────────────────────────────────────────────────────────

def test_read_frame_without_open_returns_false_none():
    reader = CameraReader(source=0)
    ok, frame = reader.read_frame()
    assert ok is False
    assert frame is None


def test_read_frame_success_returns_frame():
    mock_cv2 = make_cv2(opened=True, read_ok=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        reader.open()
        ok, frame = reader.read_frame()
    assert ok is True
    assert isinstance(frame, np.ndarray)


def test_read_frame_failure_returns_false_none():
    mock_cv2 = make_cv2(opened=True, read_ok=False)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        reader.open()
        ok, frame = reader.read_frame()
    assert ok is False
    assert frame is None


# ── release ───────────────────────────────────────────────────────────────────

def test_release_sets_cap_none():
    mock_cv2 = make_cv2(opened=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        reader.open()
        reader.release()
    assert reader._cap is None
    assert reader.is_opened() is False


def test_release_idempotent_when_not_opened():
    reader = CameraReader(source=0)
    reader.release()  # should not raise
    assert reader._cap is None


# ── props ─────────────────────────────────────────────────────────────────────

def test_props_returns_opened_false_when_not_opened():
    reader = CameraReader(source="rtsp://test")
    p = reader.props()
    assert p["opened"] is False


def test_props_returns_dimensions_when_opened():
    mock_cv2 = make_cv2(opened=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        reader = CameraReader(source=0)
        reader.open()
        p = reader.props()
    assert "width" in p
    assert "height" in p
    assert "fps" in p


# ── context manager ───────────────────────────────────────────────────────────

def test_context_manager_opens_and_releases():
    mock_cv2 = make_cv2(opened=True)
    with patch("src.ingestion.camera._cv2", mock_cv2), \
         patch("src.ingestion.camera._CV2_AVAILABLE", True):
        with CameraReader(source=0) as reader:
            assert reader.is_opened() is True
        assert reader.is_opened() is False
