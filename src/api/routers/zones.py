from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db, get_redis

router = APIRouter()

# Mirrors the DB's zones_zone_type_check constraint — validated here too so
# a bad value gets a clean 400 instead of a raw asyncpg error.
VALID_ZONE_TYPES = {"general", "restricted", "entrance", "exit", "monitored"}
_ZONE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,99}$")


class ZoneStatus(BaseModel):
    zone_id: str
    display_name: str
    zone_type: str
    occupant_count: int
    occupants: list[dict]
    object_counts: dict[str, int]


class PresenceLogEntry(BaseModel):
    person_id: str | None
    display_name: str
    camera_id: str | None
    entry_time: str
    exit_time: str | None
    is_unknown: bool


class CreateZoneRequest(BaseModel):
    zone_id: str
    display_name: str
    zone_type: str = "general"
    description: str | None = None


class UpdateZoneRequest(BaseModel):
    zone_type: str | None = None
    display_name: str | None = None
    description: str | None = None


@router.get("", response_model=list[ZoneStatus])
async def list_zones(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> list[ZoneStatus]:
    rows = (await db.execute(
        text("SELECT zone_id, display_name, zone_type FROM zones")
    )).fetchall()
    redis = await get_redis()

    zones = []
    for row in rows:
        raw = await redis.get(f"zone:{row.zone_id}:occupants")
        occupant_ids = json.loads(raw) if raw else []
        count = await redis.get(f"zone:{row.zone_id}:count") or "0"

        occupants: list[dict] = []
        if occupant_ids:
            people = (await db.execute(
                text("SELECT person_id, display_name FROM persons WHERE person_id = ANY(:ids)"),
                {"ids": occupant_ids},
            )).fetchall()
            names = {str(p.person_id): p.display_name for p in people}
            occupants = [
                {"person_id": pid, "display_name": names.get(pid, pid)}
                for pid in occupant_ids
            ]

        object_counts: dict[str, int] = {}
        count_keys = await redis.keys(f"zone:{row.zone_id}:count:*")
        for key in count_keys:
            class_name = key.split(":")[-1]
            val = await redis.get(key)
            if val is not None:
                object_counts[class_name] = int(val)

        zones.append(ZoneStatus(
            zone_id=row.zone_id,
            display_name=row.display_name,
            zone_type=row.zone_type,
            occupant_count=int(count),
            occupants=occupants,
            object_counts=object_counts,
        ))
    return zones


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_zone(
    body: CreateZoneRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("admin")

    if body.zone_type not in VALID_ZONE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"zone_type must be one of {sorted(VALID_ZONE_TYPES)}.",
        )
    if not _ZONE_ID_RE.match(body.zone_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="zone_id must be lowercase letters/digits/underscores, "
                   "starting with a letter (e.g. 'dining_room').",
        )

    try:
        await db.execute(
            text("""
                INSERT INTO zones (zone_id, display_name, zone_type, description)
                VALUES (:zone_id, :display_name, :zone_type, :description)
            """),
            {
                "zone_id": body.zone_id,
                "display_name": body.display_name,
                "zone_type": body.zone_type,
                "description": body.description,
            },
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Zone '{body.zone_id}' already exists.",
        )

    await write_audit_event(
        db, "ZONE_CREATED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="zone",
        resource_id=body.zone_id,
        details={"display_name": body.display_name, "zone_type": body.zone_type},
    )
    return {"zone_id": body.zone_id, "display_name": body.display_name, "zone_type": body.zone_type}


@router.patch("/{zone_id}")
async def update_zone(
    zone_id: str,
    body: UpdateZoneRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("admin")

    if body.zone_type is not None and body.zone_type not in VALID_ZONE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"zone_type must be one of {sorted(VALID_ZONE_TYPES)}.",
        )

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    set_clause = ", ".join(f"{col} = :{col}" for col in updates)
    result = await db.execute(
        text(f"UPDATE zones SET {set_clause} WHERE zone_id = :zone_id RETURNING zone_id"),
        {**updates, "zone_id": zone_id},
    )
    if not result.fetchone():
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zone not found.")
    await db.commit()

    await write_audit_event(
        db, "ZONE_UPDATED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="zone",
        resource_id=zone_id,
        details=updates,
    )
    return {"zone_id": zone_id, **updates}


@router.get("/{zone_id}/presence-log", response_model=list[PresenceLogEntry])
async def get_zone_presence_log(
    zone_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=200),
) -> list[PresenceLogEntry]:
    """Who entered/exited this zone and when. zone_presence is written by
    the live pipeline's ZonePresenceMonitor (see sprint4 live-test STEP 2)
    on every PERSON_ENTERED/PERSON_EXITED — this just reads that history
    back, most recent first."""
    rows = (await db.execute(
        text("""
            SELECT zp.person_id, p.display_name, zp.camera_id,
                   zp.entry_time, zp.exit_time, zp.is_unknown
            FROM zone_presence zp
            LEFT JOIN persons p ON p.person_id = zp.person_id
            WHERE zp.zone_id = :zone_id
            ORDER BY zp.entry_time DESC
            LIMIT :limit
        """),
        {"zone_id": zone_id, "limit": limit},
    )).fetchall()
    return [
        PresenceLogEntry(
            person_id=str(r.person_id) if r.person_id else None,
            display_name=r.display_name or ("Unknown" if r.is_unknown else "Unknown (unresolved)"),
            camera_id=r.camera_id,
            entry_time=r.entry_time.isoformat(),
            exit_time=r.exit_time.isoformat() if r.exit_time else None,
            is_unknown=r.is_unknown or False,
        )
        for r in rows
    ]
