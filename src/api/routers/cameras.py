from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.database import get_db, get_redis, get_redis_binary

router = APIRouter()

MJPEG_BOUNDARY = b"frame"


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


@router.get("/{camera_id}/mjpeg")
async def stream_camera_mjpeg(camera_id: str, token: str = Query(...)) -> StreamingResponse:
    """Low-latency live view for the dashboard's click-to-expand modal.

    An <img> tag can't send an Authorization header, so — same workaround
    already used by alerts.py's WebSocket — the token travels as a query
    param instead. Relays whatever run_live.py publishes to
    camera:{camera_id}:frames (see ShortTermMemory.publish_frame) as a
    multipart/x-mixed-replace stream, which browsers render natively in
    an <img> with zero JS polling needed on the client side.
    """
    import jwt as _jwt

    from src.core.config import get_settings

    settings = get_settings()
    try:
        payload = _jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        if payload.get("type") != "access":
            raise _jwt.InvalidTokenError("not an access token")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token.")

    async def gen():
        redis = await get_redis_binary()
        pubsub = redis.pubsub()
        channel = f"camera:{camera_id}:frames"
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                frame_bytes = message["data"]
                yield (
                    b"--" + MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    return StreamingResponse(
        gen(), media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}"
    )
