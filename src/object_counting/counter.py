from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class CountBucket:
    bucket_time: datetime
    camera_id: str
    zone_id: str | None
    object_class: str
    count: int


class ObjectCountAggregator:
    """
    Zone-level object counts in fixed time buckets (E03-S03).

    Per bucket the stored value is the MAX simultaneous count of each
    class — "how many people/bottles were in the zone at once" — which is
    what occupancy dashboards want; summing per-frame detections would
    just measure fps.
    """

    def __init__(
        self, camera_id: str, zone_id: str | None, bucket_seconds: int = 10
    ) -> None:
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.bucket_seconds = bucket_seconds
        self._current_start: float | None = None
        self._current_max: dict[str, int] = {}
        self._closed: list[CountBucket] = []

    def _bucket_start(self, ts: float) -> float:
        return ts - (ts % self.bucket_seconds)

    def _close_current(self) -> None:
        if self._current_start is None:
            return
        bt = datetime.fromtimestamp(self._current_start, tz=UTC)
        for cls, n in self._current_max.items():
            self._closed.append(
                CountBucket(bt, self.camera_id, self.zone_id, cls, n)
            )
        self._current_max = {}

    def observe(self, class_counts: dict[str, int], timestamp: float) -> None:
        """Feed one frame's {class_name: simultaneous count}."""
        start = self._bucket_start(timestamp)
        if self._current_start is None:
            self._current_start = start
        elif start > self._current_start:
            self._close_current()
            self._current_start = start
        for cls, n in class_counts.items():
            if n > self._current_max.get(cls, 0):
                self._current_max[cls] = n

    def pop_closed(self, now: float | None = None) -> list[CountBucket]:
        """Return finished buckets (and close the current one if its window
        has passed); caller persists them via EventStore.write_object_counts."""
        if (
            now is not None
            and self._current_start is not None
            and self._bucket_start(now) > self._current_start
        ):
            self._close_current()
            self._current_start = None
        out, self._closed = self._closed, []
        return out
