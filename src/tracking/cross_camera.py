from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from src.core.logging import get_logger
from src.tracking.strong_sort import ReidTrack, StrongSortTracker

logger = get_logger(__name__)

_DEFAULT_REID_THRESHOLD = 0.70


@dataclass
class GlobalTrack:
    """A camera-local confirmed track annotated with a system-wide UUID identity."""

    global_id: str        # UUID string — stable across cameras and re-appearances
    camera_id: str
    local_track_id: int
    bbox: np.ndarray      # [x1, y1, x2, y2] in camera coordinate space
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray


class CrossCameraCoordinator:
    """
    Assigns UUID-based global identities across multiple cameras (E03-S06).

    Each camera runs its own StrongSortTracker.  When a new local track
    appears, its ReID embedding is compared (cosine similarity) against
    all known global-identity embeddings.  If similarity > reid_threshold,
    the track re-uses the existing UUID; otherwise a fresh UUID is minted.

    Global embeddings are EMA-updated on every observation so the appearance
    model stays current as lighting and viewpoint change.

    Embedding comparisons use L2-normalised vectors so cosine similarity
    reduces to a dot-product.
    """

    def __init__(self, reid_threshold: float = _DEFAULT_REID_THRESHOLD) -> None:
        self.reid_threshold = reid_threshold
        self._trackers: dict[str, StrongSortTracker] = {}
        self._global_embeddings: dict[str, np.ndarray] = {}
        self._local_to_global: dict[tuple[str, int], str] = {}
        self._global_to_locals: dict[str, set[tuple[str, int]]] = {}

    # ── camera tracker management ────────────────────────────────────────────

    def get_tracker(self, camera_id: str) -> StrongSortTracker:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = StrongSortTracker()
        return self._trackers[camera_id]

    # ── per-frame update ─────────────────────────────────────────────────────

    def update_camera(
        self,
        camera_id: str,
        tracks: list[ReidTrack],
    ) -> list[GlobalTrack]:
        """
        Assign or look up global UUIDs for all confirmed tracks from one camera.
        Returns GlobalTrack list ready for downstream consumers.
        """
        results: list[GlobalTrack] = []
        for t in tracks:
            key = (camera_id, t.track_id)
            if key not in self._local_to_global:
                global_id = self._mint_or_match(t.embedding)
                self._local_to_global[key] = global_id
                self._global_to_locals.setdefault(global_id, set()).add(key)
                logger.debug(
                    "global_id_assigned",
                    camera_id=camera_id,
                    local_id=t.track_id,
                    global_id=global_id,
                )
            else:
                global_id = self._local_to_global[key]
                self._ema_update(global_id, t.embedding)

            results.append(GlobalTrack(
                global_id=global_id,
                camera_id=camera_id,
                local_track_id=t.track_id,
                bbox=t.bbox.copy(),
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
                embedding=t.embedding.copy(),
            ))
        return results

    # ── identity management ──────────────────────────────────────────────────

    def _mint_or_match(self, embedding: np.ndarray) -> str:
        """Return an existing UUID whose embedding matches, or a fresh UUID."""
        best_id: str | None = None
        best_sim = self.reid_threshold

        if not np.all(embedding == 0):
            na = np.linalg.norm(embedding)
            for gid, stored in self._global_embeddings.items():
                if np.all(stored == 0):
                    continue
                nb = np.linalg.norm(stored)
                if na > 1e-6 and nb > 1e-6:
                    sim = float(np.dot(embedding, stored) / (na * nb))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid

        if best_id is not None:
            self._ema_update(best_id, embedding)
            return best_id

        new_id = str(uuid.uuid4())
        self._global_embeddings[new_id] = embedding.copy()
        return new_id

    def _ema_update(self, global_id: str, new_emb: np.ndarray, alpha: float = 0.9) -> None:
        stored = self._global_embeddings.get(global_id)
        if stored is None or np.all(stored == 0):
            self._global_embeddings[global_id] = new_emb.copy()
            return
        blended = alpha * stored + (1 - alpha) * new_emb
        norm = np.linalg.norm(blended)
        self._global_embeddings[global_id] = blended / norm if norm > 1e-6 else blended

    # ── queries ───────────────────────────────────────────────────────────────

    def active_global_ids(self) -> set[str]:
        return set(self._global_to_locals.keys())

    def cameras_for_global_id(self, global_id: str) -> set[str]:
        return {cam for cam, _ in self._global_to_locals.get(global_id, set())}

    def global_id_for_local(self, camera_id: str, local_track_id: int) -> str | None:
        return self._local_to_global.get((camera_id, local_track_id))
