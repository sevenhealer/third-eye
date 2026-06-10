from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text

from src.core.database import get_db_session
from src.core.exceptions import GalleryError
from src.core.logging import get_logger
from src.face_recognition.recognizer import (
    ACCEPT_THRESHOLD,
    REJECT_THRESHOLD,
    RecognitionMatch,
    l2_normalize,
)

logger = get_logger(__name__)


class FaceGallery:
    """
    Async interface to the pgvector face embedding gallery.

    Embeddings are stored in the `face_gallery` table with an HNSW index
    (m=48, ef_construction=200) for sub-5ms cosine similarity search.

    Similarity thresholds (cosine):
      > 0.45  → accept (confirmed identity)
      0.35–0.45 → ambiguous (collect more frames)
      < 0.35  → reject (unknown person)
    """

    async def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RecognitionMatch]:
        """
        Find the top-k nearest gallery embeddings to `embedding`.

        Returns matches sorted by descending similarity, each annotated
        with a decision ("accept" / "ambiguous" / "reject").

        Raises:
            GalleryError: On database failure.
        """
        query_vec = l2_normalize(embedding)

        try:
            async with get_db_session() as session:
                result = await session.execute(
                    _GALLERY_SEARCH_SQL,
                    {"vec": _vec_literal(query_vec), "top_k": top_k},
                )
                rows = result.fetchall()
        except Exception as exc:
            raise GalleryError(f"Gallery search failed: {exc}") from exc

        matches = [
            RecognitionMatch.decide(
                embedding_id=str(row.embedding_id),
                person_id=str(row.person_id),
                similarity=float(row.similarity),
            )
            for row in rows
        ]
        # Sort descending by similarity — highest first
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    async def add_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        quality_score: float,
        camera_id: str = "",
    ) -> str:
        """
        Insert a new embedding into the gallery.

        Returns the new embedding_id (UUID).
        Raises GalleryError on failure.
        """
        embedding_id = str(uuid.uuid4())
        try:
            async with get_db_session() as session:
                await session.execute(
                    _GALLERY_INSERT_SQL,
                    {
                        "embedding_id": embedding_id,
                        "person_id": person_id,
                        "vec": _vec_literal(l2_normalize(embedding)),
                        "quality_score": quality_score,
                        "camera_id": camera_id,
                        "captured_at": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as exc:
            raise GalleryError(f"Gallery insert failed: {exc}") from exc

        logger.info("gallery_embedding_added", person_id=person_id, embedding_id=embedding_id)
        return embedding_id

    async def delete_person(self, person_id: str) -> int:
        """
        Remove all embeddings for a person.

        Returns the count of deleted rows.
        Raises GalleryError on failure.
        """
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    _GALLERY_DELETE_SQL, {"person_id": person_id}
                )
                await session.commit()
                return result.rowcount
        except Exception as exc:
            raise GalleryError(f"Gallery delete failed: {exc}") from exc


def _vec_literal(vec: np.ndarray) -> str:
    # asyncpg has no codec for the pgvector type, so the bind param must be the
    # textual form '[v1,v2,...]' that pgvector parses on CAST.
    return "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"


# Raw SQL kept here so the class stays readable and queries stay auditable
_GALLERY_SEARCH_SQL = text("""
    SELECT
        embedding_id,
        person_id,
        1 - (embedding <=> CAST(:vec AS vector)) AS similarity
    FROM face_gallery
    ORDER BY embedding <=> CAST(:vec AS vector)
    LIMIT :top_k
""")

_GALLERY_INSERT_SQL = text("""
    INSERT INTO face_gallery
        (embedding_id, person_id, embedding, quality_score, source_camera_id, captured_at)
    VALUES
        (:embedding_id, :person_id, CAST(:vec AS vector), :quality_score, :camera_id, :captured_at)
""")

_GALLERY_DELETE_SQL = text("""
    DELETE FROM face_gallery WHERE person_id = :person_id
""")


_gallery: FaceGallery | None = None


def get_gallery() -> FaceGallery:
    global _gallery
    if _gallery is None:
        _gallery = FaceGallery()
    return _gallery
