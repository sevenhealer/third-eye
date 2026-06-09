from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.object_detection.detector import ObjectDetection
from src.tracking.tracker import KalmanBoxTracker, hungarian_match, iou_matrix


@dataclass
class ReidTrack:
    """
    Confirmed track with an EMA-updated appearance embedding.
    Duck-type compatible with TrackedObject for pipeline consumers.
    """

    track_id: int
    bbox: np.ndarray        # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    hits: int = 1
    time_since_update: int = 0
    state: str = "tentative"   # tentative | confirmed | deleted

    def update_embedding(self, new_emb: np.ndarray, alpha: float = 0.9) -> None:
        """EMA blend: retain history (alpha) and incorporate new observation."""
        if np.all(self.embedding == 0):
            self.embedding = new_emb.copy()
        else:
            blended = alpha * self.embedding + (1 - alpha) * new_emb
            norm = np.linalg.norm(blended)
            self.embedding = blended / norm if norm > 1e-6 else blended


class StrongSortTracker:
    """
    StrongSORT — ByteTrack two-stage matching augmented with OSNet ReID.

    Stage 1: high-confidence detections matched against all active tracks
             using combined cost = iou_weight * IoU + (1-iou_weight) * cosine.
    Stage 2: low-confidence detections matched against remaining tracks
             using IoU only (appearance unreliable at low confidence).

    Caller supplies per-detection embeddings (np.ndarray | None).  A None or
    zero embedding falls back to pure IoU for that detection/track pair.
    """

    _next_id: int = 1

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        iou_weight: float = 0.5,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_weight = iou_weight
        self._kalman: dict[int, KalmanBoxTracker] = {}
        self._tracks: dict[int, ReidTrack] = {}

    # ── internal ──────────────────────────────────────────────────────────────

    def _new_track(self, det: ObjectDetection, emb: np.ndarray | None) -> ReidTrack:
        tid = StrongSortTracker._next_id
        StrongSortTracker._next_id += 1
        self._kalman[tid] = KalmanBoxTracker(det.bbox)
        t = ReidTrack(
            track_id=tid,
            bbox=det.bbox.copy(),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
        )
        if emb is not None:
            t.update_embedding(emb)
        if t.hits >= self.min_hits:
            t.state = "confirmed"
        return t

    def _cost_matrix(
        self,
        dets: list[ObjectDetection],
        track_ids: list[int],
        predicted: dict[int, np.ndarray],
        det_embs: list[np.ndarray | None],
    ) -> np.ndarray:
        track_bboxes = [predicted[tid] for tid in track_ids]
        iou_cost = iou_matrix([d.bbox for d in dets], track_bboxes)

        app_cost = np.zeros_like(iou_cost)
        for i, emb in enumerate(det_embs):
            if emb is None or np.all(emb == 0):
                continue
            for j, tid in enumerate(track_ids):
                stored = self._tracks[tid].embedding
                if np.all(stored == 0):
                    continue
                na = np.linalg.norm(emb)
                nb = np.linalg.norm(stored)
                if na > 1e-6 and nb > 1e-6:
                    app_cost[i, j] = float(np.dot(emb, stored) / (na * nb))

        return self.iou_weight * iou_cost + (1 - self.iou_weight) * app_cost

    def _match(
        self,
        dets: list[ObjectDetection],
        track_ids: list[int],
        predicted: dict[int, np.ndarray],
        det_embs: list[np.ndarray | None] | None,
    ) -> tuple[set[int], list[int]]:
        if not dets or not track_ids:
            return set(), list(range(len(dets)))

        embs = det_embs if det_embs is not None else [None] * len(dets)
        cost = self._cost_matrix(dets, track_ids, predicted, embs)
        matches, unmatched_dets, _ = hungarian_match(cost, self.iou_threshold)

        matched_ids: set[int] = set()
        for det_i, trk_i in matches:
            tid = track_ids[trk_i]
            self._kalman[tid].update(dets[det_i].bbox)
            t = self._tracks[tid]
            t.bbox = self._kalman[tid].bbox
            t.confidence = dets[det_i].confidence
            t.time_since_update = 0
            t.hits += 1
            if t.hits >= self.min_hits:
                t.state = "confirmed"
            if embs[det_i] is not None:
                t.update_embedding(embs[det_i])
            matched_ids.add(tid)
        return matched_ids, unmatched_dets

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[ObjectDetection],
        embeddings: list[np.ndarray | None] | None = None,
    ) -> list[ReidTrack]:
        """
        Process one frame of detections with optional appearance embeddings.
        embeddings[i] corresponds to detections[i]; pass None to skip appearance for that detection.
        Returns confirmed tracks only.
        """
        embs: list[np.ndarray | None] = embeddings if embeddings is not None else [None] * len(detections)
        predicted = {tid: kf.predict() for tid, kf in self._kalman.items()}
        active_ids = list(self._tracks.keys())

        high_idx = [i for i, d in enumerate(detections) if d.confidence >= self.high_threshold]
        low_idx = [
            i for i, d in enumerate(detections)
            if self.low_threshold <= d.confidence < self.high_threshold
        ]
        high_dets = [detections[i] for i in high_idx]
        high_embs = [embs[i] for i in high_idx]
        low_dets = [detections[i] for i in low_idx]

        matched1, unmatched_high = self._match(high_dets, active_ids, predicted, high_embs)

        remaining = [tid for tid in active_ids if tid not in matched1]
        matched2, _ = self._match(low_dets, remaining, predicted, None)

        matched_all = matched1 | matched2

        for i in unmatched_high:
            t = self._new_track(high_dets[i], high_embs[i])
            self._tracks[t.track_id] = t

        for tid in active_ids:
            if tid not in matched_all:
                self._tracks[tid].time_since_update += 1
                if self._tracks[tid].time_since_update > self.max_age:
                    self._tracks[tid].state = "deleted"

        for tid in [tid for tid, t in self._tracks.items() if t.state == "deleted"]:
            del self._tracks[tid]
            del self._kalman[tid]

        return [t for t in self._tracks.values() if t.state == "confirmed"]

    def reset(self) -> None:
        self._kalman.clear()
        self._tracks.clear()
        StrongSortTracker._next_id = 1

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
