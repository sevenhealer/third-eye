import numpy as np
import pytest

from src.tracking.strong_sort import ReidTrack
from src.pipeline.zone_events import ZoneEventDetector, ZoneObjectEvent


def _track(tid: int, cls_name: str = "person") -> ReidTrack:
    return ReidTrack(
        track_id=tid,
        bbox=np.array([0, 0, 50, 50], dtype="float32"),
        class_id=0,
        class_name=cls_name,
        confidence=0.9,
        hits=3,
        state="confirmed",
    )


def _events(
    det: ZoneEventDetector,
    tracks: list[ReidTrack],
    cam: str = "cam0",
    zone: str = "zone_a",
    ts: int = 1_000_000,
) -> list[ZoneObjectEvent]:
    return det.update(cam, zone, tracks, timestamp_ns=ts)


# ── OBJECT_ADDED ──────────────────────────────────────────────────────────────

def test_new_track_fires_added():
    det = ZoneEventDetector()
    events = _events(det, [_track(1)])
    assert len(events) == 1
    assert events[0].event_type == "OBJECT_ADDED"
    assert events[0].track_id == 1


def test_existing_track_no_event():
    det = ZoneEventDetector()
    _events(det, [_track(1)])
    events = _events(det, [_track(1)])
    assert events == []


def test_multiple_new_tracks_fire_multiple_added():
    det = ZoneEventDetector()
    events = _events(det, [_track(1), _track(2)])
    assert len(events) == 2
    assert all(e.event_type == "OBJECT_ADDED" for e in events)
    assert {e.track_id for e in events} == {1, 2}


# ── OBJECT_REMOVED with grace period ─────────────────────────────────────────

def test_track_missing_within_grace_no_removed():
    det = ZoneEventDetector(removal_grace_frames=3)
    _events(det, [_track(1)])
    for _ in range(3):                  # exactly at grace boundary, not over
        events = _events(det, [])
        assert events == []


def test_track_missing_past_grace_fires_removed():
    det = ZoneEventDetector(removal_grace_frames=3)
    _events(det, [_track(1)])
    for _ in range(4):                  # 4 > grace=3
        events = _events(det, [])
    assert len(events) == 1
    assert events[0].event_type == "OBJECT_REMOVED"
    assert events[0].track_id == 1


def test_track_reappears_within_grace_no_removed():
    det = ZoneEventDetector(removal_grace_frames=5)
    _events(det, [_track(1)])
    _events(det, [])                    # miss 1
    _events(det, [])                    # miss 2
    events = _events(det, [_track(1)]) # re-appears before grace expires
    # Re-appearance generates ADDED again (track left and came back)
    assert not any(e.event_type == "OBJECT_REMOVED" for e in events)


def test_class_name_preserved_in_removed_event():
    det = ZoneEventDetector(removal_grace_frames=1)
    _events(det, [_track(5, cls_name="car")])
    _events(det, [])                   # miss 1
    events = _events(det, [])          # miss 2 > grace=1
    assert events[0].class_name == "car"


# ── per-zone isolation ────────────────────────────────────────────────────────

def test_events_isolated_by_zone():
    det = ZoneEventDetector()
    e_a = _events(det, [_track(1)], zone="zone_a")
    e_b = _events(det, [_track(1)], zone="zone_b")
    # Track 1 is new in zone_b too — both should fire ADDED
    assert e_a[0].event_type == "OBJECT_ADDED"
    assert e_b[0].event_type == "OBJECT_ADDED"


def test_events_isolated_by_camera():
    det = ZoneEventDetector()
    e0 = _events(det, [_track(1)], cam="cam0")
    e1 = _events(det, [_track(1)], cam="cam1")
    assert e0[0].event_type == "OBJECT_ADDED"
    assert e1[0].event_type == "OBJECT_ADDED"


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_all_clears_state():
    det = ZoneEventDetector()
    _events(det, [_track(1)])
    det.reset()
    events = _events(det, [_track(1)])
    assert events[0].event_type == "OBJECT_ADDED"  # fires again after reset


def test_reset_specific_camera():
    det = ZoneEventDetector()
    _events(det, [_track(1)], cam="cam0")
    _events(det, [_track(1)], cam="cam1")
    det.reset(camera_id="cam0")
    # cam0 cleared — track 1 fires ADDED again
    assert _events(det, [_track(1)], cam="cam0")[0].event_type == "OBJECT_ADDED"
    # cam1 not cleared — no event
    assert _events(det, [_track(1)], cam="cam1") == []
