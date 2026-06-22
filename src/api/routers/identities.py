from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db
from src.storage.object_store import StorageError, get_object_store

router = APIRouter()


class PersonOut(BaseModel):
    person_id: UUID
    display_name: str
    role: str | None
    enrolled_at: str
    is_active: bool
    last_seen_at: str | None
    last_seen_zone: str | None
    metadata: dict[str, Any]


class EnrollRequest(BaseModel):
    display_name: str
    role: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = {}


@router.get("", response_model=list[PersonOut])
async def list_identities(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> list[PersonOut]:
    result = await db.execute(
        text("""
            SELECT person_id, display_name, role, enrolled_at,
                   is_active, last_seen_at, last_seen_zone, metadata
            FROM persons
            WHERE is_active = true
            ORDER BY display_name
        """)
    )
    rows = result.fetchall()
    return [
        PersonOut(
            person_id=r.person_id,
            display_name=r.display_name,
            role=r.role,
            enrolled_at=r.enrolled_at.isoformat(),
            is_active=r.is_active,
            last_seen_at=r.last_seen_at.isoformat() if r.last_seen_at else None,
            last_seen_zone=r.last_seen_zone,
            metadata=r.metadata or {},
        )
        for r in rows
    ]


@router.post("", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def enroll_identity(
    body: EnrollRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    current_user.require_role("admin")

    result = await db.execute(
        text("""
            INSERT INTO persons (display_name, role, notes, enrolled_by, metadata)
            VALUES (:name, :role, :notes, :enrolled_by, CAST(:metadata AS jsonb))
            RETURNING person_id, display_name, role, enrolled_at,
                      is_active, last_seen_at, last_seen_zone, metadata
        """),
        {
            "name": body.display_name,
            "role": body.role,
            "notes": body.notes,
            "enrolled_by": str(current_user.user_id),
            "metadata": json.dumps(body.metadata or {}),
        },
    )
    await db.commit()
    row = result.fetchone()

    await write_audit_event(
        db,
        "IDENTITY_ENROLLED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="person",
        resource_id=str(row.person_id),
        details={"display_name": body.display_name, "role": body.role},
    )

    return PersonOut(
        person_id=row.person_id,
        display_name=row.display_name,
        role=row.role,
        enrolled_at=row.enrolled_at.isoformat(),
        is_active=row.is_active,
        last_seen_at=None,
        last_seen_zone=None,
        metadata=row.metadata or {},
    )


# ── Enrollment candidates (E02-S06) ──────────────────────────────────────────
# NOTE: declared before /{person_id} so "candidates" isn't parsed as a UUID.

class CandidateApproveRequest(BaseModel):
    display_name: str | None = None   # required when creating a NEW person
    role: str = "visitor"
    person_id: UUID | None = None     # set to MERGE into an existing person


@router.get("/candidates")
async def list_candidates(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    candidate_status: str = Query("pending", alias="status"),
) -> dict:
    current_user.require_role("admin")
    # LATERAL join surfaces the nearest enrolled person to each candidate —
    # the "this unknown looks like Rohan (0.62)" hint that drives the merge
    # decision (approve-to-existing vs create-new).
    result = await db.execute(
        text("""
            SELECT c.candidate_id, c.track_id, c.camera_id, c.first_seen_at,
                   c.last_seen_at, c.quality_scores, c.crop_paths, c.status,
                   vector_dims(c.mean_embedding) AS dims,
                   m.person_id AS sugg_id, m.display_name AS sugg_name, m.sim AS sugg_sim
            FROM enrollment_candidates c
            LEFT JOIN LATERAL (
                SELECT p.person_id, p.display_name,
                       1 - (g.embedding <=> c.mean_embedding) AS sim
                FROM face_gallery g JOIN persons p USING (person_id)
                WHERE c.mean_embedding IS NOT NULL AND p.is_active
                ORDER BY g.embedding <=> c.mean_embedding
                LIMIT 1
            ) m ON true
            WHERE c.status = :status
            ORDER BY c.last_seen_at DESC
            LIMIT 100
        """),
        {"status": candidate_status},
    )
    return {
        "candidates": [
            {
                "candidate_id": str(r.candidate_id),
                "track_id": r.track_id,
                "camera_id": r.camera_id,
                "first_seen_at": r.first_seen_at.isoformat(),
                "last_seen_at": r.last_seen_at.isoformat(),
                "crop_count": len(r.crop_paths or []),
                "crop_urls": [
                    f"/api/v1/identities/candidates/{r.candidate_id}/crop/{i}"
                    for i in range(len(r.crop_paths or []))
                ],
                "embedding_dims": r.dims,
                "status": r.status,
                # suggested existing match — null if no gallery/no faces
                "suggested_person_id": str(r.sugg_id) if r.sugg_id else None,
                "suggested_name": r.sugg_name,
                "suggested_similarity": round(float(r.sugg_sim), 3) if r.sugg_sim is not None else None,
            }
            for r in result.fetchall()
        ]
    }


@router.post("/candidates/{candidate_id}/approve", response_model=PersonOut,
             status_code=status.HTTP_201_CREATED)
async def approve_candidate(
    candidate_id: UUID,
    body: CandidateApproveRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    current_user.require_role("admin")

    cand = (await db.execute(
        text("""
            SELECT candidate_id FROM enrollment_candidates
            WHERE candidate_id = :cid AND status = 'pending'
              AND mean_embedding IS NOT NULL
        """),
        {"cid": str(candidate_id)},
    )).fetchone()
    if cand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found, not pending, or has no embedding.",
        )

    if body.person_id is not None:
        # MERGE: re-sighting of someone already enrolled — add this candidate's
        # embedding to their gallery (improves recognition) instead of making a
        # duplicate person. This is the operator-confirmed re-sighting case.
        person = (await db.execute(
            text("""
                SELECT person_id, display_name, role, enrolled_at, is_active,
                       last_seen_at, last_seen_zone, metadata
                FROM persons WHERE person_id = :pid AND is_active
            """),
            {"pid": str(body.person_id)},
        )).fetchone()
        if person is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Target person not found or inactive.",
            )
    else:
        if not body.display_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="display_name required when creating a new person "
                       "(or pass person_id to merge into an existing one).",
            )
        person = (await db.execute(
            text("""
                INSERT INTO persons (display_name, role, enrolled_by)
                VALUES (:name, :role, :uid)
                RETURNING person_id, display_name, role, enrolled_at,
                          is_active, last_seen_at, last_seen_zone, metadata
            """),
            {"name": body.display_name, "role": body.role,
             "uid": str(current_user.user_id)},
        )).fetchone()

    # candidate's mean embedding is added to the (new or existing) gallery
    await db.execute(
        text("""
            INSERT INTO face_gallery
                (person_id, embedding, quality_score, source_camera_id)
            SELECT :pid, mean_embedding, 0.75, camera_id
            FROM enrollment_candidates WHERE candidate_id = :cid
        """),
        {"pid": str(person.person_id), "cid": str(candidate_id)},
    )
    await db.execute(
        text("""
            UPDATE enrollment_candidates
            SET status = 'approved', reviewed_by = :uid, reviewed_at = now(),
                assigned_person_id = :pid
            WHERE candidate_id = :cid
        """),
        {"uid": str(current_user.user_id), "pid": str(person.person_id),
         "cid": str(candidate_id)},
    )
    await db.commit()

    await write_audit_event(
        db, "CANDIDATE_MERGED" if body.person_id else "CANDIDATE_APPROVED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="enrollment_candidate",
        resource_id=str(candidate_id),
        details={"display_name": person.display_name,
                 "person_id": str(person.person_id),
                 "merged": bool(body.person_id)},
    )
    return PersonOut(
        person_id=person.person_id,
        display_name=person.display_name,
        role=person.role,
        enrolled_at=person.enrolled_at.isoformat(),
        is_active=person.is_active,
        last_seen_at=None,
        last_seen_zone=None,
        metadata=person.metadata or {},
    )


@router.get("/candidates/{candidate_id}/crop/{index}")
async def get_candidate_crop(
    candidate_id: UUID,
    index: int,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Proxies the crop image from object storage. The browser can't resolve
    the internal 'minio' hostname used for presigned URLs, so this streams
    the bytes through the API instead."""
    current_user.require_role("admin")
    row = (await db.execute(
        text("SELECT crop_paths FROM enrollment_candidates WHERE candidate_id = :cid"),
        {"cid": str(candidate_id)},
    )).fetchone()
    if row is None or not row.crop_paths or index >= len(row.crop_paths):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop not found.")
    try:
        data = get_object_store().get_bytes(row.crop_paths[index])
    except StorageError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(content=data, media_type="image/jpeg")


@router.post("/candidates/{candidate_id}/reject")
async def reject_candidate(
    candidate_id: UUID,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("admin")
    result = await db.execute(
        text("""
            UPDATE enrollment_candidates
            SET status = 'rejected', reviewed_by = :uid, reviewed_at = now()
            WHERE candidate_id = :cid AND status = 'pending'
            RETURNING candidate_id
        """),
        {"uid": str(current_user.user_id), "cid": str(candidate_id)},
    )
    if not result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found or not pending.",
        )
    await db.commit()
    await write_audit_event(
        db, "CANDIDATE_REJECTED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="enrollment_candidate",
        resource_id=str(candidate_id),
    )
    return {"message": "Candidate rejected."}


@router.get("/{person_id}", response_model=PersonOut)
async def get_identity(
    person_id: UUID,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> PersonOut:
    result = await db.execute(
        text("""
            SELECT person_id, display_name, role, enrolled_at,
                   is_active, last_seen_at, last_seen_zone, metadata
            FROM persons WHERE person_id = :pid
        """),
        {"pid": str(person_id)},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")
    return PersonOut(
        person_id=row.person_id,
        display_name=row.display_name,
        role=row.role,
        enrolled_at=row.enrolled_at.isoformat(),
        is_active=row.is_active,
        last_seen_at=row.last_seen_at.isoformat() if row.last_seen_at else None,
        last_seen_zone=row.last_seen_zone,
        metadata=row.metadata or {},
    )


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_identity(
    person_id: UUID,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    current_user.require_role("admin")

    result = await db.execute(
        text("UPDATE persons SET is_active = false WHERE person_id = :pid RETURNING person_id"),
        {"pid": str(person_id)},
    )
    if not result.fetchone():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")
    await db.commit()

    await write_audit_event(
        db,
        "IDENTITY_SOFT_DELETED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="person",
        resource_id=str(person_id),
    )


@router.get("/{person_id}/timeline")
async def get_timeline(
    person_id: UUID,
    current_user: ActiveUser,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("analyst")

    result = await db.execute(
        text("""
            SELECT zone_id, camera_id, entry_time, exit_time, confidence
            FROM zone_presence
            WHERE person_id = :pid
              AND entry_time > now() - (:days || ' days')::interval
            ORDER BY entry_time DESC
            LIMIT 500
        """),
        {"pid": str(person_id), "days": days},
    )
    rows = result.fetchall()
    return {
        "person_id": str(person_id),
        "days_requested": days,
        "events": [
            {
                "zone_id": r.zone_id,
                "camera_id": r.camera_id,
                "entry_time": r.entry_time.isoformat(),
                "exit_time": r.exit_time.isoformat() if r.exit_time else None,
                "confidence": r.confidence,
            }
            for r in rows
        ],
    }
