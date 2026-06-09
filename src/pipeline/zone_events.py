from __future__ import annotations

import time
from dataclasses import dataclass

from src.core.logging import get_logger
from src.tracking.strong_sort import ReidTrack

logger = get_logger(__name__)


@dataclass
class ZoneObjectEvent:
    event_type: str     # "OBJECT_ADDED" | "OBJECT_REMOVED"
    camera_id: str
    zone_id: str
    track_id: int
    class_name: str
    timestamp_ns: int


class ZoneEventDetector:
    """
    Detects OBJECT_ADDED and OBJECT_REMOVED events per (camera_id, zone_id) (E03-S07).

    An OBJECT_ADDED event fires the first time a track_id is observed in a zone.
    An OBJECT_REMOVED event fires after a track is absent for more than
    `removal_grace_frames` consecutive frames, preventing spurious removals
    during brief occlusions.

    This class is stateful: one instance should be shared across all frames
    for a given pipeline run.
    """

    def __init__(self, removal_grace_frames: int = 5) -> None:
        self.removal_grace_frames = removal_grace_frames
        # (camera_id, zone_id) → {track_id: class_name}
        self._known: dict[tuple[str, str], dict[int, str]] = {}
        # (camera_id, zone_id) → {track_id: consecutive-miss count}
        self._miss: dict[tuple[str, str], dict[int, int]] = {}

    def update(
        self,
        camera_id: str,
        zone_id: str,
        tracks: list[ReidTrack],
        timestamp_ns: int | None = None,
    ) -> list[ZoneObjectEvent]:
        """
        Diff current confirmed tracks against the previous frame's inventory.
        Returns zero or more ZoneObjectEvent instances.
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        key = (camera_id, zone_id)
        known = self._known.setdefault(key, {})
        miss = self._miss.setdefault(key, {})
        events: list[ZoneObjectEvent] = []

        current = {t.track_id: t.class_name for t in tracks}

        for tid, cls_name in current.items():
            miss.pop(tid, None)
            if tid not in known:
                known[tid] = cls_name
                events.append(ZoneObjectEvent(
                    event_type="OBJECT_ADDED",
                    camera_id=camera_id,
                    zone_id=zone_id,
                    track_id=tid,
                    class_name=cls_name,
                    timestamp_ns=timestamp_ns,
                ))
                logger.debug(
                    "object_added",
                    camera_id=camera_id,
                    zone_id=zone_id,
                    track_id=tid,
                    class_name=cls_name,
                )

        for tid in list(known.keys()):
            if tid not in current:
                miss[tid] = miss.get(tid, 0) + 1
                if miss[tid] > self.removal_grace_frames:
                    cls_name = known.pop(tid)
                    miss.pop(tid, None)
                    events.append(ZoneObjectEvent(
                        event_type="OBJECT_REMOVED",
                        camera_id=camera_id,
                        zone_id=zone_id,
                        track_id=tid,
                        class_name=cls_name,
                        timestamp_ns=timestamp_ns,
                    ))
                    logger.debug(
                        "object_removed",
                        camera_id=camera_id,
                        zone_id=zone_id,
                        track_id=tid,
                        class_name=cls_name,
                    )

        return events

    def reset(
        self,
        camera_id: str | None = None,
        zone_id: str | None = None,
    ) -> None:
        """Clear state for a specific camera/zone pair, or everything if both are None."""
        if camera_id is None:
            self._known.clear()
            self._miss.clear()
            return
        keys = [
            k for k in self._known
            if k[0] == camera_id and (zone_id is None or k[1] == zone_id)
        ]
        for k in keys:
            self._known.pop(k, None)
            self._miss.pop(k, None)
