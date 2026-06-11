from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.object_detection.detector import ObjectDetection

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ── IoU helpers ───────────────────────────────────────────────────────────────

def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(
    dets: list[np.ndarray], tracks: list[np.ndarray]
) -> np.ndarray:
    mat = np.zeros((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            mat[i, j] = iou(d, t)
    return mat


def hungarian_match(
    cost: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Returns (matches, unmatched_row_indices, unmatched_col_indices).
    Pairs with cost < threshold are rejected.
    """
    n_rows, n_cols = cost.shape
    if cost.size == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    if _SCIPY_AVAILABLE:
        row_ind, col_ind = _lsa(-cost)
    else:
        # Greedy fallback: row by row, pick highest unused column
        row_ind, col_ind, used = [], [], set()
        for r in range(n_rows):
            best_c, best_v = -1, -1.0
            for c in range(n_cols):
                if c not in used and cost[r, c] > best_v:
                    best_v, best_c = cost[r, c], c
            if best_c >= 0:
                row_ind.append(r)
                col_ind.append(best_c)
                used.add(best_c)

    matches: list[tuple[int, int]] = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Kalman filter for an axis-aligned bounding box.

    State  : [x_c, y_c, w, h, vx, vy, vw, vh]
    Observe: [x_c, y_c, w, h]
    """

    def __init__(self, bbox: np.ndarray) -> None:
        x_c = (bbox[0] + bbox[2]) / 2.0
        y_c = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4, 1e4])

        self.x = np.array(
            [[x_c], [y_c], [w], [h], [0.0], [0.0], [0.0], [0.0]]
        )

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox()

    def update(self, bbox: np.ndarray) -> None:
        x_c = (bbox[0] + bbox[2]) / 2.0
        y_c = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([[x_c], [y_c], [w], [h]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def _to_bbox(self) -> np.ndarray:
        xc, yc, w, h = self.x[:4, 0]
        return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

    @property
    def bbox(self) -> np.ndarray:
        return self._to_bbox()


def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb


# ── Track dataclass ───────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id: int
    bbox: np.ndarray       # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    hits: int = 1
    time_since_update: int = 0
    state: str = "tentative"   # tentative | confirmed | deleted
    embedding: np.ndarray | None = None   # EMA appearance embedding, L2-normalized
    # bank of distinct past embeddings (poses); re-association matches the
    # best of these, so one frontal sighting can revive a track that was
    # last seen in profile or motion-blurred
    embedding_bank: list[np.ndarray] = field(default_factory=list)


# ── ByteTracker ───────────────────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Three-stage matching:
      Stage 1 — high-confidence detections vs. all active tracks (IoU)
      Stage 2 — low-confidence detections vs. remaining unmatched tracks
      Stage 3 — appearance re-association: unmatched high-conf detections
                vs. remaining tracks by embedding cosine similarity. Recovers
                a track that jumped too far for IoU (feed stall, occlusion,
                re-entry) when detections carry embeddings; no-op otherwise.
    Unmatched high-conf detections start new tentative tracks.
    Tracks confirmed after min_hits matches; deleted after max_age misses.
    """

    _next_id: int = 1

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        appearance_threshold: float = 0.45,
        ema_alpha: float = 0.9,
        nn_budget: int = 30,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.appearance_threshold = appearance_threshold
        self.ema_alpha = ema_alpha
        self.nn_budget = nn_budget
        self._kalman: dict[int, KalmanBoxTracker] = {}
        self._tracks: dict[int, TrackedObject] = {}

    # ── internal ──────────────────────────────────────────────────────────────

    def _new_track(self, det: ObjectDetection) -> TrackedObject:
        tid = ByteTracker._next_id
        ByteTracker._next_id += 1
        self._kalman[tid] = KalmanBoxTracker(det.bbox)
        t = TrackedObject(
            track_id=tid,
            bbox=det.bbox.copy(),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
        )
        if det.embedding is not None:
            self._update_embedding(t, det.embedding)
        if t.hits >= self.min_hits:
            t.state = "confirmed"
        return t

    def _update_embedding(self, t: TrackedObject, emb: np.ndarray) -> None:
        emb = _normalize(emb)
        if t.embedding is None:
            t.embedding = emb
        else:
            t.embedding = _normalize(
                self.ema_alpha * t.embedding + (1.0 - self.ema_alpha) * emb
            )
        # bank keeps only *novel* poses — near-duplicates of stored entries
        # add nothing and would FIFO out the older, genuinely different views
        if all(float(emb @ b) < 0.95 for b in t.embedding_bank):
            t.embedding_bank.append(emb)
            if len(t.embedding_bank) > self.nn_budget:
                t.embedding_bank.pop(0)

    @staticmethod
    def _appearance_sim(t: TrackedObject, d_emb: np.ndarray) -> float:
        """Best cosine similarity between a detection embedding and any stored
        view of the track (DeepSORT-style nearest-neighbor over the bank)."""
        sims = [float(d_emb @ b) for b in t.embedding_bank]
        if t.embedding is not None:
            sims.append(float(d_emb @ t.embedding))
        return max(sims) if sims else 0.0

    def _match_and_update(
        self,
        dets: list[ObjectDetection],
        track_ids: list[int],
        predicted: dict[int, np.ndarray],
        update_embeddings: bool = True,
    ) -> tuple[set[int], list[int]]:
        """Match dets against track_ids; return (matched_track_ids, unmatched_det_indices)."""
        if not dets or not track_ids:
            return set(), list(range(len(dets)))

        track_bboxes = [predicted[tid] for tid in track_ids]
        det_bboxes = [d.bbox for d in dets]
        cost = iou_matrix(det_bboxes, track_bboxes)
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
            if update_embeddings and dets[det_i].embedding is not None:
                self._update_embedding(t, dets[det_i].embedding)
            matched_ids.add(tid)
        return matched_ids, unmatched_dets

    def _reassociate_by_appearance(
        self, high: list[ObjectDetection], unmatched_high: list[int], matched_all: set[int]
    ) -> list[int]:
        """Stage 3: revive unmatched tracks from unmatched high-conf detections
        by embedding similarity. Mutates matched_all; returns the still-unmatched
        detection indices."""
        cand = [i for i in unmatched_high if high[i].embedding is not None]
        track_ids = [
            tid for tid, t in self._tracks.items()
            if tid not in matched_all and t.embedding is not None
        ]
        if not cand or not track_ids:
            return unmatched_high

        sim = np.zeros((len(cand), len(track_ids)), dtype=np.float32)
        for r, det_i in enumerate(cand):
            d_emb = _normalize(high[det_i].embedding)
            for c, tid in enumerate(track_ids):
                t = self._tracks[tid]
                if t.class_id == high[det_i].class_id:
                    sim[r, c] = self._appearance_sim(t, d_emb)

        matches, _, _ = hungarian_match(sim, self.appearance_threshold)
        revived: set[int] = set()
        for r, c in matches:
            det_i, tid = cand[r], track_ids[c]
            det = high[det_i]
            # motion state is stale after the gap — restart the filter at the
            # new position instead of dragging the old velocity across the jump
            self._kalman[tid] = KalmanBoxTracker(det.bbox)
            t = self._tracks[tid]
            t.bbox = det.bbox.copy()
            t.confidence = det.confidence
            t.time_since_update = 0
            t.hits += 1
            if t.hits >= self.min_hits:
                t.state = "confirmed"
            self._update_embedding(t, det.embedding)
            matched_all.add(tid)
            revived.add(det_i)
        return [i for i in unmatched_high if i not in revived]

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detections: list[ObjectDetection]) -> list[TrackedObject]:
        """Process one frame of detections; returns confirmed tracks."""
        predicted: dict[int, np.ndarray] = {
            tid: kf.predict() for tid, kf in self._kalman.items()
        }
        active_ids = list(self._tracks.keys())

        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [
            d for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # Stage 1: high-conf dets vs. all active tracks
        matched1, unmatched_high = self._match_and_update(high, active_ids, predicted)

        # Stage 2: low-conf dets vs. remaining unmatched tracks.
        # Low-conf boxes are usually occluded/blurred — don't let them
        # contaminate the appearance EMA.
        remaining_ids = [tid for tid in active_ids if tid not in matched1]
        matched2, _ = self._match_and_update(
            low, remaining_ids, predicted, update_embeddings=False
        )

        matched_all = matched1 | matched2

        # Stage 3: appearance re-association for tracks IoU couldn't recover
        unmatched_high = self._reassociate_by_appearance(high, unmatched_high, matched_all)

        # New tracks from unmatched high-conf detections
        for det_i in unmatched_high:
            t = self._new_track(high[det_i])
            self._tracks[t.track_id] = t

        # Age unmatched existing tracks; mark deletions
        for tid in active_ids:
            if tid not in matched_all:
                self._tracks[tid].time_since_update += 1
                if self._tracks[tid].time_since_update > self.max_age:
                    self._tracks[tid].state = "deleted"

        # Prune deleted
        for tid in [tid for tid, t in self._tracks.items() if t.state == "deleted"]:
            del self._tracks[tid]
            del self._kalman[tid]

        # Coasting tracks (unmatched this frame) keep a stale bbox — emitting
        # them paints frozen ghost boxes, so only report tracks updated now.
        return [
            t
            for t in self._tracks.values()
            if t.state == "confirmed" and t.time_since_update == 0
        ]

    def reset(self) -> None:
        self._kalman.clear()
        self._tracks.clear()
        ByteTracker._next_id = 1

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
