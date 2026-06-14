from src.event_detection.zone_presence import PresentTrack, ZonePresenceMonitor


def _track(tid=1, name="Rohan", pid="uuid-1", state="verified", sim=0.7):
    return PresentTrack(track_id=tid, person_id=pid, person_name=name,
                        identity_state=state, similarity=sim)


def _monitor(**kw):
    defaults = {"camera_id": "cam0", "zone_id": "bedroom",
                "enter_grace_frames": 3, "exit_grace_frames": 5,
                "unknown_grace_frames": 3}
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


def test_unknown_person_fires_only_after_grace():
    """UNKNOWN_PERSON_DETECTED fires after the track STAYS unknown past the
    grace — not at entry — so recognition has a chance to catch up first."""
    m = _monitor()   # enter_grace=3, unknown_grace=3
    unknown = _track(tid=2, name="unknown", pid=None, state="unknown", sim=0.1)
    # entry at frame 3 — UNKNOWN must NOT fire yet
    assert [e.event_type for e in m.update([unknown])] == []
    assert [e.event_type for e in m.update([unknown])] == []
    assert [e.event_type for e in m.update([unknown])] == ["PERSON_ENTERED"]
    # stays unknown → UNKNOWN fires after unknown_grace frames post-entry
    events = []
    for _ in range(3):
        events += m.update([unknown])
    assert [e.event_type for e in events] == ["UNKNOWN_PERSON_DETECTED"]
    assert events[0].person_id is None


def test_unknown_not_fired_if_recognized_in_time():
    """An enrolled person who resolves within the grace must NOT raise a
    false UNKNOWN_PERSON_DETECTED."""
    m = _monitor()
    t = _track(tid=2, name="unknown", pid=None, state="unknown", sim=0.1)
    for _ in range(3):
        m.update([t])   # enters as unknown
    # recognition catches up before unknown_grace elapses
    known = _track(tid=2, name="Rohan", state="verified")
    events = []
    for _ in range(10):
        events += m.update([known])
    assert all(e.event_type != "UNKNOWN_PERSON_DETECTED" for e in events)


def test_two_tracks_independent():
    m = _monitor()
    a, b = _track(tid=1), _track(tid=2, name="unknown", pid=None, state="unknown")
    for _ in range(2):
        m.update([a, b])
    events = m.update([a, b])
    types = sorted(e.event_type for e in events)
    assert types == ["PERSON_ENTERED", "PERSON_ENTERED"]   # unknown still in grace
    assert m.occupancy == 2
