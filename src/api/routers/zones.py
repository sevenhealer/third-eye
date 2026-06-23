from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.database import get_db, get_redis

router = APIRouter()


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
