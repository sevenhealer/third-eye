import numpy as np
import pytest

from src.object_detection.detector import ObjectDetection
from src.tracking.tracker import (
    ByteTracker,
    KalmanBoxTracker,
    hungarian_match,
    iou,
)


def _det(x1=100, y1=100, x2=200, y2=200, cls_id=0, conf=0.9, name="person"):
    return ObjectDetection(
        bbox=np.array([x1, y1, x2, y2], dtype="float32"),
        class_id=cls_id,
        class_name=name,
        confidence=conf,
    )


def fresh_tracker(**kw) -> ByteTracker:
    defaults: dict = {"max_age": 5, "min_hits": 3}
    defaults.update(kw)
    t = ByteTracker(**defaults)
    t.reset()
    return t


# ── iou ──────────────────────────────────────────────────────────────────────

def test_iou_perfect_overlap():
    box = np.array([0, 0, 100, 100], dtype="float32")
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = np.array([0, 0, 10, 10], dtype="float32")
    b = np.array([20, 20, 30, 30], dtype="float32")
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = np.array([0, 0, 10, 10], dtype="float32")
    b = np.array([5, 0, 15, 10], dtype="float32")
    # inter=50, area_a=100, area_b=100, union=150
    assert iou(a, b) == pytest.approx(50 / 150)


# ── hungarian_match ──────────────────────────────────────────────────────────

def test_hungarian_match_perfect():
    cost = np.array([[0.9, 0.1], [0.1, 0.8]], dtype="float32")
    matches, unmatched_r, unmatched_c = hungarian_match(cost, threshold=0.3)
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert unmatched_r == []
    assert unmatched_c == []


def test_hungarian_match_below_threshold_rejected():
    cost = np.array([[0.1]], dtype="float32")
    matches, unmatched_r, unmatched_c = hungarian_match(cost, threshold=0.3)
    assert matches == []
    assert 0 in unmatched_r
    assert 0 in unmatched_c


def test_hungarian_match_empty_cost():
    matches, unmatched_r, unmatched_c = hungarian_match(
        np.zeros((0, 0)), threshold=0.3
    )
    assert matches == []


# ── Kalman filter ─────────────────────────────────────────────────────────────

def test_kalman_predict_returns_array():
    kf = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype="float32"))
    pred = kf.predict()
    assert pred.shape == (4,)


def test_kalman_update_moves_toward_measurement():
    kf = KalmanBoxTracker(np.array([0, 0, 100, 100], dtype="float32"))
    kf.predict()
    kf.update(np.array([200, 200, 300, 300], dtype="float32"))
    x_c = (kf.bbox[0] + kf.bbox[2]) / 2
    # After one update towards center=(250,250), predicted center shifts from 50 toward 250
    assert x_c > 50


# ── ByteTracker ───────────────────────────────────────────────────────────────

def test_no_detections_no_tracks():
    tracker = fresh_tracker()
    result = tracker.update([])
    assert result == []


def test_single_high_conf_detection_creates_tentative():
    tracker = fresh_tracker(min_hits=3, high_threshold=0.6)
    tracker.update([_det(conf=0.9)])
    # After 1 hit, still tentative — confirmed tracks list is empty
    assert len(tracker.update([])) == 0
    assert tracker.active_track_count == 1


def test_track_confirmed_after_min_hits():
    tracker = fresh_tracker(min_hits=3, high_threshold=0.6)
    det = _det(conf=0.9)
    for _ in range(3):
        result = tracker.update([det])
    assert len(result) == 1
    assert result[0].state == "confirmed"


def test_two_nearby_detections_get_different_ids():
    tracker = fresh_tracker(min_hits=1, high_threshold=0.6)
    det1 = _det(0, 0, 50, 50, conf=0.9)
    det2 = _det(300, 300, 350, 350, conf=0.9)
    result = tracker.update([det1, det2])
    ids = {t.track_id for t in result}
    assert len(ids) == 2


def test_coasting_track_not_emitted():
    """A confirmed track that missed this frame must not report a stale bbox."""
    tracker = fresh_tracker(min_hits=1, max_age=5, high_threshold=0.6)
    det = _det(conf=0.9)
    assert len(tracker.update([det])) == 1   # confirmed and matched
    assert tracker.update([]) == []          # coasting: suppressed, not deleted
    assert tracker.active_track_count == 1
    # same id comes back once the detection reappears
    result = tracker.update([det])
    assert len(result) == 1
    assert result[0].time_since_update == 0


def test_track_deleted_after_max_age():
    tracker = fresh_tracker(min_hits=1, max_age=2, high_threshold=0.6)
    tracker.update([_det(conf=0.9)])   # hit 1 → confirmed (min_hits=1)
    tracker.update([])                 # miss 1
    tracker.update([])                 # miss 2
    result = tracker.update([])        # miss 3 > max_age=2 → deleted
    assert result == []
    assert tracker.active_track_count == 0


def test_reset_clears_all_state():
    tracker = fresh_tracker(min_hits=1, high_threshold=0.6)
    for _ in range(3):
        tracker.update([_det(conf=0.9)])
    tracker.reset()
    assert tracker.active_track_count == 0
    assert ByteTracker._next_id == 1
