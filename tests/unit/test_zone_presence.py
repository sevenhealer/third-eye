from src.event_detection.zone_presence import PresentTrack, ZonePresenceMonitor


def _track(tid=1, name="Rohan", pid="uuid-1", state="verified", sim=0.7):
    return PresentTrack(track_id=tid, person_id=pid, person_name=name,
                        identity_state=state, similarity=sim)


def _monitor(**kw):
    defaults = {"camera_id": "cam0", "zone_id": "bedroom",
                "enter_grace_frames": 3, "exit_grace_frames": 5}
    defaults.update(kw)
    return ZonePresenceMonitor(**defaults)


def test_entered_fires_after_grace_only_once():
    m = _monitor()
    assert m.update([_track()]) == []
    assert m.update([_track()]) == []
    events = m.update([_track()])
    assert [e.event_type for e in events] == ["PERSON_ENTERED"]
    assert events[0].person_name == "Rohan"
    assert events[0].identity_state == "verified"
    # already entered: no repeat
    assert m.update([_track()]) == []


def test_one_frame_phantom_never_enters():
    m = _monitor()
    m.update([_track()])
    for _ in range(10):
        assert m.update([]) == []


def test_exit_fires_after_grace_not_on_blink():
    m = _monitor()
    for _ in range(3):
        m.update([_track()])
    # 2-frame blink: below exit grace, no exit
    m.update([])
    m.update([])
    assert m.update([_track()]) == []
    # real exit
    events = []
    for _ in range(5):
        events += m.update([])
    assert [e.event_type for e in events] == ["PERSON_EXITED"]
    assert m.occupancy == 0


def test_unknown_person_event_fires_with_entry():
    m = _monitor()
    unknown = _track(tid=2, name="unknown", pid=None, state="unknown", sim=0.1)
    m.update([unknown])
    m.update([unknown])
    events = m.update([unknown])
    assert [e.event_type for e in events] == [
        "PERSON_ENTERED", "UNKNOWN_PERSON_DETECTED"
    ]
    assert events[0].person_id is None


def test_two_tracks_independent():
    m = _monitor()
    a, b = _track(tid=1), _track(tid=2, name="unknown", pid=None, state="unknown")
    for _ in range(2):
        m.update([a, b])
    events = m.update([a, b])
    types = sorted(e.event_type for e in events)
    assert types == ["PERSON_ENTERED", "PERSON_ENTERED", "UNKNOWN_PERSON_DETECTED"]
    assert m.occupancy == 2
