from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PresentTrack:
    """One tracked person as seen this frame, with its identity snapshot."""
    track_id: int
    person_id: str | None = None      # UUID string when recognized
    person_name: str = "unknown"
    identity_state: str = "unknown"   # verified | provisional | unknown
    similarity: float = 0.0


@dataclass
class PresenceEvent:
    event_type: str                   # PERSON_ENTERED | PERSON_EXITED | UNKNOWN_PERSON_DETECTED
    camera_id: str
    zone_id: str
    track_id: int
    person_id: str | None
    person_name: str
    identity_state: str
    similarity: float
    timestamp_ns: int


@dataclass
class _TrackState:
    seen_frames: int = 0              # consecutive frames present (pre-entry)
    miss_frames: int = 0              # consecutive frames absent (post-entry)
    entered: bool = False
    unknown_reported: bool = False
    last: PresentTrack = field(default_factory=lambda: PresentTrack(track_id=-1))


class ZonePresenceMonitor:
    """
    Debounced PERSON_ENTERED / PERSON_EXITED per (camera, zone).

    A track must be present `enter_grace_frames` consecutive frames before
    ENTERED fires (kills one-frame phantoms), and absent `exit_grace_frames`
    before EXITED fires (kills blink-exits during occlusion — the tracker
    already coasts, this is a second layer for the event stream).

    UNKNOWN_PERSON_DETECTED fires once per entered track that is not
    recognized at entry time — the alert layer decides what that means
    per zone type. Events carry the identity snapshot current at fire time;
    later corrections are appended by the caller (IDENTITY_CORRECTED), never
    rewritten here.
    """

    def __init__(
        self,
        camera_id: str,
        zone_id: str,
        enter_grace_frames: int = 5,
        exit_grace_frames: int = 45,
    ) -> None:
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.enter_grace_frames = enter_grace_frames
        self.exit_grace_frames = exit_grace_frames
        self._tracks: dict[int, _TrackState] = {}

    def _event(self, event_type: str, t: PresentTrack, ts_ns: int) -> PresenceEvent:
        return PresenceEvent(
            event_type=event_type,
            camera_id=self.camera_id,
            zone_id=self.zone_id,
            track_id=t.track_id,
            person_id=t.person_id,
            person_name=t.person_name,
            identity_state=t.identity_state,
            similarity=t.similarity,
            timestamp_ns=ts_ns,
        )

    def update(
        self, present: list[PresentTrack], timestamp_ns: int | None = None
    ) -> list[PresenceEvent]:
        """Feed one frame's person tracks; returns events that fired."""
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        events: list[PresenceEvent] = []
        present_ids = {p.track_id for p in present}

        for p in present:
            st = self._tracks.setdefault(p.track_id, _TrackState())
            st.last = p
            st.miss_frames = 0
            if not st.entered:
                st.seen_frames += 1
                if st.seen_frames >= self.enter_grace_frames:
                    st.entered = True
                    events.append(self._event("PERSON_ENTERED", p, ts))
                    if p.identity_state == "unknown" and not st.unknown_reported:
                        st.unknown_reported = True
                        events.append(self._event("UNKNOWN_PERSON_DETECTED", p, ts))

        for tid, st in list(self._tracks.items()):
            if tid in present_ids:
                continue
            if not st.entered:
                # never entered and gone again — phantom, drop silently
                del self._tracks[tid]
                continue
            st.miss_frames += 1
            if st.miss_frames >= self.exit_grace_frames:
                events.append(self._event("PERSON_EXITED", st.last, ts))
                del self._tracks[tid]

        return events

    @property
    def occupancy(self) -> int:
        """Currently-entered person count (for zone count displays)."""
        return sum(1 for st in self._tracks.values() if st.entered)
