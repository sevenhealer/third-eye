import time

import numpy as np

from src.face_recognition.manual_enroll import ManualEnrollCapture, pose_bucket


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=64).astype("float32")


def test_pose_bucket_straight():
    assert pose_bucket(0, 0) == "straight"
    assert pose_bucket(5, -5) == "straight"


def test_pose_bucket_left_right():
    assert pose_bucket(-25, 0) == "left"
    assert pose_bucket(25, 0) == "right"


def test_pose_bucket_up_down():
    assert pose_bucket(0, 20) == "up"
    assert pose_bucket(0, -20) == "down"


def test_pose_bucket_ambiguous_between_cells():
    assert pose_bucket(15, 0) is None


def test_create_distributes_need_across_buckets():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=10)
    session = cap.get(sid)
    assert sum(session.need.values()) == 10
    assert set(session.need.keys()) == {"straight", "left", "right", "up", "down"}


def test_create_handles_remainder():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=7)
    session = cap.get(sid)
    assert sum(session.need.values()) == 7


def test_try_capture_fills_matching_bucket():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=5)
    bucket, captured = cap.try_capture(sid, yaw=0, pitch=0, embedding=_emb(0))
    assert bucket == "straight"
    assert captured is True
    assert cap.get(sid).need["straight"] == (5 // 5) - 1


def test_try_capture_debounces_rapid_frames():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=10)
    _, first = cap.try_capture(sid, yaw=0, pitch=0, embedding=_emb(0))
    _, second = cap.try_capture(sid, yaw=0, pitch=0, embedding=_emb(1))
    assert first is True
    assert second is False   # too soon after the first capture


def test_try_capture_skips_full_bucket():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=5)   # 1 slot per bucket
    cap.try_capture(sid, yaw=0, pitch=0, embedding=_emb(0))
    session = cap.get(sid)
    session.last_capture = 0.0   # bypass debounce to isolate the full-bucket check
    bucket, captured = cap.try_capture(sid, yaw=0, pitch=0, embedding=_emb(1))
    assert bucket == "straight"
    assert captured is False
    assert len(session.embeddings) == 1


def test_pending_empty_when_all_buckets_filled():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=5)
    session = cap.get(sid)
    for name in list(session.need):
        session.need[name] = 0
    assert session.pending == []


def test_discard_removes_session():
    cap = ManualEnrollCapture()
    sid = cap.create(crops_total=5)
    cap.discard(sid)
    assert cap.get(sid) is None
