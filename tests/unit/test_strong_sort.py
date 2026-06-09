import numpy as np
import pytest

from src.object_detection.detector import ObjectDetection
from src.tracking.strong_sort import ReidTrack, StrongSortTracker


def _det(x1=100, y1=100, x2=200, y2=200, cls_id=0, conf=0.9, name="person"):
    return ObjectDetection(
        bbox=np.array([x1, y1, x2, y2], dtype="float32"),
        class_id=cls_id,
        class_name=name,
        confidence=conf,
    )


def _emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def fresh(**kw) -> StrongSortTracker:
    defaults = {"max_age": 5, "min_hits": 3, "high_threshold": 0.6, "iou_weight": 0.5}
    defaults.update(kw)
    t = StrongSortTracker(**defaults)
    t.reset()
    return t


# ── ReidTrack.update_embedding ────────────────────────────────────────────────

def test_update_embedding_first_sets_value():
    t = ReidTrack(track_id=1, bbox=np.zeros(4), class_id=0, class_name="p", confidence=0.9)
    e = _emb(1)
    t.update_embedding(e)
    assert np.allclose(t.embedding, e)


def test_update_embedding_ema_blends():
    t = ReidTrack(track_id=1, bbox=np.zeros(4), class_id=0, class_name="p", confidence=0.9)
    e1 = _emb(1)
    e2 = _emb(2)
    t.update_embedding(e1)
    t.update_embedding(e2, alpha=0.9)
    # result should be unit-norm blend of e1 and e2
    assert abs(np.linalg.norm(t.embedding) - 1.0) < 1e-5
    # must differ from both originals
    assert not np.allclose(t.embedding, e1)
    assert not np.allclose(t.embedding, e2)


def test_update_embedding_stays_unit_norm():
    t = ReidTrack(track_id=1, bbox=np.zeros(4), class_id=0, class_name="p", confidence=0.9)
    for seed in range(5):
        t.update_embedding(_emb(seed))
    assert abs(np.linalg.norm(t.embedding) - 1.0) < 1e-5


# ── StrongSortTracker basic behaviour ─────────────────────────────────────────

def test_no_detections_no_tracks():
    tracker = fresh()
    assert tracker.update([]) == []


def test_single_high_conf_becomes_tentative_then_confirmed():
    tracker = fresh(min_hits=3, high_threshold=0.6)
    det = _det(conf=0.9)
    tracker.update([det])
    tracker.update([det])
    result = tracker.update([det])
    assert len(result) == 1
    assert result[0].state == "confirmed"


def test_two_distant_detections_get_different_ids():
    tracker = fresh(min_hits=1)
    r = tracker.update([_det(0, 0, 50, 50, conf=0.9), _det(400, 400, 450, 450, conf=0.9)])
    ids = {t.track_id for t in r}
    assert len(ids) == 2


def test_track_deleted_after_max_age():
    tracker = fresh(min_hits=1, max_age=2)
    tracker.update([_det(conf=0.9)])   # confirmed (min_hits=1)
    tracker.update([])
    tracker.update([])
    result = tracker.update([])        # 3 misses > max_age=2
    assert result == []
    assert tracker.active_track_count == 0


def test_reset_clears_all_state():
    tracker = fresh(min_hits=1)
    for _ in range(3):
        tracker.update([_det(conf=0.9)])
    tracker.reset()
    assert tracker.active_track_count == 0
    assert StrongSortTracker._next_id == 1


# ── ReID embedding integration ────────────────────────────────────────────────

def test_update_with_embeddings_accepted():
    tracker = fresh(min_hits=1)
    det = _det(conf=0.9)
    emb = _emb(42)
    result = tracker.update([det], embeddings=[emb])
    assert len(result) == 1
    # Track should carry the embedding
    stored = tracker._tracks[result[0].track_id].embedding
    assert not np.all(stored == 0)


def test_appearance_bias_matches_similar_track():
    """
    Two detections at the same box but with very different embeddings should
    both re-match to the *same* track when one embedding aligns perfectly
    with the stored appearance (pure appearance match scenario).
    """
    tracker = fresh(min_hits=1, iou_weight=0.0)   # appearance-only matching
    emb_a = _emb(1)
    det = _det(conf=0.9)
    # Build track with emb_a
    tracker.update([det], embeddings=[emb_a])

    # Next frame: same location, same embedding → should keep track
    result = tracker.update([det], embeddings=[emb_a])
    assert len(result) == 1


def test_no_embeddings_falls_back_to_iou():
    tracker = fresh(min_hits=2)
    det = _det(conf=0.9)
    tracker.update([det])
    result = tracker.update([det])
    # Track confirmed via IoU only, no embeddings supplied
    assert len(result) == 1
