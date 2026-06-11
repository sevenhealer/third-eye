from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db

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
