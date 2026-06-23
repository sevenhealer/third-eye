from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

# Same 5-pose-bucket scheme as scripts/enroll.py, ported so a browser
# (getUserMedia) can drive the same guided capture instead of a local
# OpenCV window. Session state lives here, in the API process - short-lived
# (SESSION_TTL_S) and in-memory, same tradeoff as candidates.py's per-camera
# track buffers: there's exactly one operator doing this at a time.
POSE_BUCKETS: list[tuple[str, str]] = [
    ("straight", "Look straight at the camera"),
    ("left", "Turn your head left"),
    ("right", "Turn your head right"),
    ("up", "Tilt your head up"),
    ("down", "Tilt your head down"),
]

SESSION_TTL_S = 180.0
CAPTURE_DEBOUNCE_S = 0.4   # one held pose shouldn't fill a bucket in one burst


def pose_bucket(yaw: float, pitch: float) -> str | None:
    if abs(yaw) < 12 and abs(pitch) < 12:
        return "straight"
    if yaw <= -18:
        return "left"
    if yaw >= 18:
        return "right"
    if pitch >= 14:
        return "up"
    if pitch <= -14:
        return "down"
    return None   # between cells - too ambiguous to file anywhere


@dataclass
class _Session:
    crops_total: int
    need: dict[str, int]
    embeddings: list[np.ndarray] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    last_capture: float = 0.0

    @property
    def pending(self) -> list[str]:
        return [name for name, n in self.need.items() if n > 0]


class ManualEnrollCapture:
    """Per-session accumulator for the browser-driven guided enrollment
    capture (Settings -> Enroll Person)."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def create(self, crops_total: int) -> str:
        self._sweep()
        session_id = uuid.uuid4().hex
        base, rem = divmod(crops_total, len(POSE_BUCKETS))
        need = {
            name: base + (1 if i < rem else 0)
            for i, (name, _) in enumerate(POSE_BUCKETS)
        }
        self._sessions[session_id] = _Session(crops_total=crops_total, need=need)
        return session_id

    def get(self, session_id: str) -> _Session | None:
        return self._sessions.get(session_id)

    def try_capture(self, session_id: str, yaw: float, pitch: float,
                     embedding: np.ndarray) -> tuple[str | None, bool]:
        """Returns (bucket, captured) - the pose bucket this frame matched
        (if any) and whether it was actually captured (debounced/already
        full buckets return captured=False)."""
        session = self._sessions[session_id]
        bucket = pose_bucket(yaw, pitch)
        now = time.monotonic()
        if (
            bucket is not None
            and session.need.get(bucket, 0) > 0
            and now - session.last_capture >= CAPTURE_DEBOUNCE_S
        ):
            session.embeddings.append(embedding)
            session.need[bucket] -= 1
            session.last_capture = now
            return bucket, True
        return bucket, False

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _sweep(self) -> None:
        now = time.monotonic()
        stale = [sid for sid, s in self._sessions.items()
                 if now - s.created_at > SESSION_TTL_S]
        for sid in stale:
            self._sessions.pop(sid, None)


_capture: ManualEnrollCapture | None = None


def get_manual_enroll_capture() -> ManualEnrollCapture:
    global _capture
    if _capture is None:
        _capture = ManualEnrollCapture()
    return _capture
