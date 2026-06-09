import numpy as np
import pytest

from src.tracking.cross_camera import CrossCameraCoordinator
from src.tracking.strong_sort import ReidTrack


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _track(tid: int, emb: np.ndarray, cls_name: str = "person") -> ReidTrack:
    return ReidTrack(
        track_id=tid,
        bbox=np.array([0, 0, 50, 50], dtype="float32"),
        class_id=0,
        class_name=cls_name,
        confidence=0.9,
        embedding=emb.copy(),
        hits=3,
        state="confirmed",
    )


# ── get_tracker ───────────────────────────────────────────────────────────────

def test_get_tracker_creates_new_per_camera():
    coord = CrossCameraCoordinator()
    t1 = coord.get_tracker("cam0")
    t2 = coord.get_tracker("cam1")
    assert t1 is not t2


def test_get_tracker_returns_same_instance():
    coord = CrossCameraCoordinator()
    assert coord.get_tracker("cam0") is coord.get_tracker("cam0")


# ── update_camera — UUID assignment ──────────────────────────────────────────

def test_new_track_gets_uuid():
    coord = CrossCameraCoordinator()
    results = coord.update_camera("cam0", [_track(1, _emb(1))])
    assert len(results) == 1
    gid = results[0].global_id
    assert len(gid) == 36   # UUID4 string length


def test_same_track_same_uuid_across_frames():
    coord = CrossCameraCoordinator()
    emb = _emb(1)
    r1 = coord.update_camera("cam0", [_track(1, emb)])
    r2 = coord.update_camera("cam0", [_track(1, emb)])
    assert r1[0].global_id == r2[0].global_id


def test_different_local_tracks_get_different_uuids_when_dissimilar():
    coord = CrossCameraCoordinator(reid_threshold=0.99)  # very strict threshold
    r1 = coord.update_camera("cam0", [_track(1, _emb(1))])
    r2 = coord.update_camera("cam0", [_track(2, _emb(99))])
    assert r1[0].global_id != r2[0].global_id


def test_similar_tracks_across_cameras_share_uuid():
    """Same person seen on cam0 and cam1 should get the same UUID."""
    emb = _emb(7)
    coord = CrossCameraCoordinator(reid_threshold=0.0)  # always match any non-zero embedding
    r0 = coord.update_camera("cam0", [_track(1, emb)])
    r1 = coord.update_camera("cam1", [_track(1, emb)])
    assert r0[0].global_id == r1[0].global_id


def test_zero_embedding_always_gets_fresh_uuid():
    zero = np.zeros(512, dtype=np.float32)
    coord = CrossCameraCoordinator(reid_threshold=0.0)
    r1 = coord.update_camera("cam0", [_track(1, zero)])
    r2 = coord.update_camera("cam1", [_track(1, zero)])
    # Zero embeddings have no similarity; each should be a fresh UUID
    assert r1[0].global_id != r2[0].global_id


# ── queries ───────────────────────────────────────────────────────────────────

def test_cameras_for_global_id():
    coord = CrossCameraCoordinator(reid_threshold=0.0)
    emb = _emb(3)
    r0 = coord.update_camera("cam0", [_track(1, emb)])
    r1 = coord.update_camera("cam1", [_track(2, emb)])
    # Both cameras matched to the same global id
    gid = r0[0].global_id
    cameras = coord.cameras_for_global_id(gid)
    assert "cam0" in cameras
    assert "cam1" in cameras


def test_global_id_for_local_returns_none_for_unknown():
    coord = CrossCameraCoordinator()
    assert coord.global_id_for_local("cam0", 999) is None


def test_active_global_ids_reflects_seen_tracks():
    coord = CrossCameraCoordinator(reid_threshold=0.99)
    coord.update_camera("cam0", [_track(1, _emb(1)), _track(2, _emb(2))])
    assert len(coord.active_global_ids()) == 2


# ── GlobalTrack fields ────────────────────────────────────────────────────────

def test_global_track_has_correct_fields():
    emb = _emb(5)
    coord = CrossCameraCoordinator()
    results = coord.update_camera("cam0", [_track(3, emb, cls_name="car")])
    gt = results[0]
    assert gt.camera_id == "cam0"
    assert gt.local_track_id == 3
    assert gt.class_name == "car"
    assert gt.embedding.shape == (512,)
