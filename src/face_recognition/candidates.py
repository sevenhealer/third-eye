from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text

from src.core.database import get_db_session
from src.core.logging import get_logger
from src.face_recognition.recognizer import l2_normalize
from src.storage import ObjectStore, StorageError, get_object_store
from src.storage.object_store import object_key

logger = get_logger(__name__)

# below the gallery reject threshold two faces are different people; we
# only merge a sighting into an existing pending candidate above accept
MERGE_SIMILARITY = 0.45


def _vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec.tolist()) + "]"


@dataclass
class _UnknownTrack:
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    crops: list[bytes] = field(default_factory=list)   # JPEG-encoded face crops
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    persisted: bool = False


@dataclass
class CandidateDraft:
    track_id: int
    camera_id: str
    mean_embedding: np.ndarray
    quality_scores: list[float]
    crops: list[bytes]
    first_seen: float
    last_seen: float


class CandidateCapture:
    """
    Unknown-person enrollment candidate workflow (E02-S06), capture side.

    Accumulates good-view embeddings for tracks the recognition layer
    calls `unknown`. Once a track has >= min_crops good views spanning
    >= min_span_sec, it becomes a candidate draft; persist() merges it
    into an existing *pending* candidate when the faces match
    (similarity >= 0.45), so the same stranger seen twice stays ONE
    review item instead of multiplying.
    """

    def __init__(
        self,
        camera_id: str,
        min_crops: int = 5,
        min_span_sec: float = 3.0,
        max_stored_crops: int = 5,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.min_crops = min_crops
        self.min_span_sec = min_span_sec
        self.max_stored_crops = max_stored_crops
        # object store for face crops; falls back to the configured singleton
        self._store = object_store if object_store is not None else get_object_store()
        self._tracks: dict[int, _UnknownTrack] = {}

    def observe(
        self,
        track_id: int,
        embedding: np.ndarray | None,
        *,
        is_unknown: bool,
        good_view: bool,
        quality: float,
        crop: bytes | None = None,
    ) -> None:
        if not is_unknown:
            self._tracks.pop(track_id, None)   # got recognized — not a candidate
            return
        if embedding is None or not good_view:
            return
        st = self._tracks.setdefault(track_id, _UnknownTrack())
        if st.persisted:
            st.last_seen = time.time()
            return
        st.embeddings.append(l2_normalize(embedding.copy()))
        st.qualities.append(round(float(quality), 3))
        if crop is not None:
            st.crops.append(crop)
        st.last_seen = time.time()

    def due(self) -> list[CandidateDraft]:
        """Drafts that crossed the crop/span thresholds (marked persisted
        so each track produces at most one)."""
        out: list[CandidateDraft] = []
        for tid, st in self._tracks.items():
            if st.persisted:
                continue
            if (
                len(st.embeddings) >= self.min_crops
                and st.last_seen - st.first_seen >= self.min_span_sec
            ):
                mean = l2_normalize(np.mean(st.embeddings, axis=0))
                # keep the highest-quality crops as the candidate's photos
                if st.crops and len(st.crops) == len(st.qualities):
                    order = sorted(range(len(st.crops)),
                                   key=lambda i: st.qualities[i], reverse=True)
                    best_crops = [st.crops[i] for i in order[:self.max_stored_crops]]
                else:
                    best_crops = st.crops[:self.max_stored_crops]
                out.append(CandidateDraft(
                    track_id=tid,
                    camera_id=self.camera_id,
                    mean_embedding=mean,
                    quality_scores=st.qualities[:20],
                    crops=best_crops,
                    first_seen=st.first_seen,
                    last_seen=st.last_seen,
                ))
                st.persisted = True
        return out

    def _upload_crops(self, crops: list[bytes]) -> list[str]:
        """Store face crops in MinIO; return their keys. Best-effort — a
        storage failure must not block candidate creation."""
        keys: list[str] = []
        if not self._store.is_enabled:
            return keys
        for crop in crops:
            key = object_key("crops", self.camera_id, "jpg")
            try:
                self._store.put_bytes(key, crop, content_type="image/jpeg")
                keys.append(key)
            except StorageError as exc:
                logger.warning("candidate_crop_upload_failed", error=str(exc))
        return keys

    async def persist(self, draft: CandidateDraft) -> str:
        """Upsert: merge into a matching pending candidate or insert new.
        Returns the candidate_id."""
        vec = _vec_literal(draft.mean_embedding)
        crop_keys = self._upload_crops(draft.crops)
        async with get_db_session() as session:
            existing = (await session.execute(
                text("""
                    SELECT candidate_id,
                           1 - (mean_embedding <=> CAST(:vec AS vector)) AS sim
                    FROM enrollment_candidates
                    WHERE status = 'pending' AND mean_embedding IS NOT NULL
                    ORDER BY mean_embedding <=> CAST(:vec AS vector)
                    LIMIT 1
                """),
                {"vec": vec},
            )).fetchone()

            if existing is not None and float(existing.sim) >= MERGE_SIMILARITY:
                cid = str(existing.candidate_id)
                await session.execute(
                    text("""
                        UPDATE enrollment_candidates
                        SET last_seen_at = :last_seen,
                            quality_scores = quality_scores ||
                                CAST(:scores AS jsonb),
                            crop_paths = crop_paths || CAST(:crops AS jsonb)
                        WHERE candidate_id = :cid
                    """),
                    {
                        "cid": cid,
                        "last_seen": datetime.fromtimestamp(draft.last_seen, tz=UTC),
                        "scores": json.dumps(draft.quality_scores),
                        "crops": json.dumps(crop_keys),
                    },
                )
                await session.commit()
                logger.info("enrollment_candidate_merged",
                            candidate_id=cid, similarity=round(float(existing.sim), 3))
                return cid

            row = (await session.execute(
                text("""
                    INSERT INTO enrollment_candidates
                        (track_id, camera_id, first_seen_at, last_seen_at,
                         mean_embedding, quality_scores, crop_paths)
                    VALUES (:tid, :cam, :first_seen, :last_seen,
                            CAST(:vec AS vector), CAST(:scores AS jsonb),
                            CAST(:crops AS jsonb))
                    RETURNING candidate_id
                """),
                {
                    "tid": str(draft.track_id),
                    "cam": draft.camera_id,
                    "first_seen": datetime.fromtimestamp(draft.first_seen, tz=UTC),
                    "last_seen": datetime.fromtimestamp(draft.last_seen, tz=UTC),
                    "vec": vec,
                    "scores": json.dumps(draft.quality_scores),
                    "crops": json.dumps(crop_keys),
                },
            )).fetchone()
            await session.commit()
            cid = str(row.candidate_id)
            logger.info("enrollment_candidate_created",
                        candidate_id=cid, track_id=draft.track_id)
            return cid
