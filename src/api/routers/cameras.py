from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.database import get_db, get_redis

router = APIRouter()


class CameraStatus(BaseModel):
    camera_id: str
    display_name: str
    is_active: bool
    status: str
    zone_id: str | None
    location_desc: str | None


@router.get("", response_model=list[CameraStatus])
async def list_cameras(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> list[CameraStatus]:
    result = await db.execute(
        text("SELECT camera_id, display_name, is_active, zone_id, location_desc FROM cameras")
    )
    rows = result.fetchall()
    redis = await get_redis()

    cameras = []
    for row in rows:
        health_status = await redis.get(f"camera:{row.camera_id}:status") or "unknown"
        cameras.append(
            CameraStatus(
                camera_id=row.camera_id,
                display_name=row.display_name,
                is_active=row.is_active,
                status=health_status,
                zone_id=row.zone_id,
                location_desc=row.location_desc,
            )
        )
    return cameras


@router.get("/{camera_id}/snapshot")
async def get_snapshot(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        text("SELECT camera_id FROM cameras WHERE camera_id = :cid AND is_active = true"),
        {"cid": camera_id},
    )
    if not result.fetchone():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    return {"camera_id": camera_id, "snapshot_url": f"/stream/{camera_id}/latest.jpg"}
