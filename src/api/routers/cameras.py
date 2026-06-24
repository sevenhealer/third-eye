from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db, get_db_session, get_redis, get_redis_binary

router = APIRouter()

MJPEG_BOUNDARY = b"frame"
KICK_CHANNEL = "camera:control:kick"
_CAMERA_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,99}$")
_CREDS_RE = re.compile(r"^(\w+://)[^:/@]+:[^:/@]+@(.+)$")


def _mask_stream_url(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://••••:••••@host/..., so credentials
    never show up in the camera list, an edit form, or an audit log entry."""
    m = _CREDS_RE.match(url)
    if not m:
        return url
    scheme, rest = m.groups()
    return f"{scheme}••••:••••@{rest}"


class CameraLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fps: int = 15
    det_size: int = 0
    det_thresh: float = Field(0.6, ge=0.3, le=1.0)
    reid_thresh: float = Field(0.45, ge=0.35, le=1.0)
    min_face: int = 24
    recognize: bool = False
    persist_events: bool = False
    cpu: bool = False
    # Liveness/anti-spoofing. enforce_liveness hard-blocks a static
    # presentation (photo/screen/statue) — it's forced to unknown, never
    # recognized or persisted. collect_antispoofing auto-saves face crops,
    # auto-labelled live/spoof by the gate's own verdict, for later CDCN++
    # training (reviewable/correctable from the Anti-Spoofing page).
    enforce_liveness: bool = False
    collect_antispoofing: bool = False
    # Normalized [x1,y1,x2,y2] rectangles (0-1) whose detections are dropped —
    # masks a burnt-in timestamp/OSD (its ticking digits fake motion/liveness)
    # or a fixed false positive like a statue.
    ignore_regions: list[list[float]] = Field(default_factory=list)
    # --show (opens a cv2 GUI window) and --bypass-antispoofing (skips
    # liveness checks) are intentionally not fields here at all - extra
    # ="forbid" means a client can't even smuggle them through by name.


class CameraStatus(BaseModel):
    camera_id: str
    display_name: str
    is_active: bool
    status: str
    zone_id: str | None
    location_desc: str | None
    gpu_id: int | None
    desired_state: str
    process_state: str | None
    config_version: int


class CameraDetail(CameraStatus):
    stream_url: str
    resolution_w: int
    resolution_h: int
    fps_target: int
    launch_args: CameraLaunchArgs


class CreateCameraRequest(BaseModel):
    camera_id: str
    display_name: str
    stream_url: str
    zone_id: str | None = None
    location_desc: str | None = None
    resolution_w: int = 1920
    resolution_h: int = 1080
    fps_target: int = 30
    gpu_id: int | None = None
    launch_args: CameraLaunchArgs = CameraLaunchArgs()


class UpdateCameraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = None
    stream_url: str | None = None
    zone_id: str | None = None
    location_desc: str | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps_target: int | None = None
    gpu_id: int | None = None
    is_active: bool | None = None
    launch_args: CameraLaunchArgs | None = None


# Fields that actually change what gets passed to run_live.py - editing any
# of these bumps config_version so the orchestrator restarts the camera.
_RESTART_TRIGGERING_FIELDS = {"stream_url", "zone_id", "gpu_id", "launch_args"}


async def _kick_supervisor() -> None:
    redis = await get_redis()
    await redis.publish(KICK_CHANNEL, "1")


def _camera_status_row(row, health_status: str) -> CameraStatus:
    return CameraStatus(
        camera_id=row.camera_id,
        display_name=row.display_name,
        is_active=row.is_active,
        status=health_status,
        zone_id=row.zone_id,
        location_desc=row.location_desc,
        gpu_id=row.gpu_id,
        desired_state=row.desired_state,
        process_state=row.process_state,
        config_version=row.config_version,
    )


_LIST_QUERY = """
    SELECT c.camera_id, c.display_name, c.is_active, c.zone_id, c.location_desc,
           c.gpu_id, c.desired_state, c.config_version,
           cps.state AS process_state
    FROM cameras c
    LEFT JOIN camera_process_status cps ON cps.camera_id = c.camera_id
"""


@router.get("", response_model=list[CameraStatus])
async def list_cameras(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> list[CameraStatus]:
    # Deactivated (soft-deleted) cameras stay in the table for their
    # zone_presence/audit history but shouldn't clutter the main list -
    # get_camera (detail/edit) deliberately has no such filter, since you
    # still need to be able to look at/reactivate one.
    rows = (await db.execute(text(f"{_LIST_QUERY} WHERE c.is_active = true"))).fetchall()
    redis = await get_redis()
    out = []
    for row in rows:
        health_status = await redis.get(f"camera:{row.camera_id}:status") or "unknown"
        out.append(_camera_status_row(row, health_status))
    return out


@router.get("/{camera_id}", response_model=CameraDetail)
async def get_camera(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> CameraDetail:
    row = (await db.execute(
        text(f"""
            {_LIST_QUERY}
            WHERE c.camera_id = :camera_id
        """),
        {"camera_id": camera_id},
    )).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    full = (await db.execute(
        text("SELECT stream_url, resolution_w, resolution_h, fps_target, launch_args FROM cameras WHERE camera_id = :camera_id"),
        {"camera_id": camera_id},
    )).fetchone()
    redis = await get_redis()
    health_status = await redis.get(f"camera:{camera_id}:status") or "unknown"
    base = _camera_status_row(row, health_status)
    return CameraDetail(
        **base.model_dump(),
        stream_url=_mask_stream_url(full.stream_url),
        resolution_w=full.resolution_w,
        resolution_h=full.resolution_h,
        fps_target=full.fps_target,
        launch_args=full.launch_args or {},
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CameraDetail)
async def create_camera(
    body: CreateCameraRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> CameraDetail:
    current_user.require_role("admin")
    if not _CAMERA_ID_RE.match(body.camera_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="camera_id must be lowercase letters/digits/underscores, "
                   "starting with a letter (e.g. 'front_door').",
        )
    try:
        await db.execute(
            text("""
                INSERT INTO cameras
                    (camera_id, display_name, stream_url, zone_id, location_desc,
                     resolution_w, resolution_h, fps_target, gpu_id, launch_args)
                VALUES
                    (:camera_id, :display_name, :stream_url, :zone_id, :location_desc,
                     :resolution_w, :resolution_h, :fps_target, :gpu_id, CAST(:launch_args AS jsonb))
            """),
            {
                "camera_id": body.camera_id, "display_name": body.display_name,
                "stream_url": body.stream_url, "zone_id": body.zone_id,
                "location_desc": body.location_desc, "resolution_w": body.resolution_w,
                "resolution_h": body.resolution_h, "fps_target": body.fps_target,
                "gpu_id": body.gpu_id, "launch_args": json.dumps(body.launch_args.model_dump()),
            },
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera '{body.camera_id}' already exists.",
        )

    await write_audit_event(
        db, "CAMERA_CREATED",
        actor_id=current_user.user_id, actor_username=current_user.username,
        resource_type="camera", resource_id=body.camera_id,
        details={"display_name": body.display_name, "zone_id": body.zone_id, "gpu_id": body.gpu_id},
    )
    return await get_camera(body.camera_id, current_user, db)


@router.patch("/{camera_id}", response_model=CameraDetail)
async def update_camera(
    camera_id: str,
    body: UpdateCameraRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> CameraDetail:
    current_user.require_role("admin")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")
    bumps_config = bool(_RESTART_TRIGGERING_FIELDS & updates.keys())

    # sql_params carries the stringified launch_args for the jsonb cast;
    # updates (the dict of real Python values) stays untouched so the
    # audit log below doesn't end up double-JSON-encoding it.
    sql_params = dict(updates)
    if "launch_args" in sql_params:
        sql_params["launch_args"] = json.dumps(sql_params["launch_args"])

    set_parts = []
    for col in updates:
        if col == "launch_args":
            set_parts.append("launch_args = CAST(:launch_args AS jsonb)")
        else:
            set_parts.append(f"{col} = :{col}")
    if bumps_config:
        set_parts.append("config_version = config_version + 1")
    set_parts.append("updated_at = now()")

    result = await db.execute(
        text(f"UPDATE cameras SET {', '.join(set_parts)} WHERE camera_id = :camera_id RETURNING camera_id"),
        {**sql_params, "camera_id": camera_id},
    )
    if not result.fetchone():
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    await db.commit()

    audit_details = {k: v for k, v in updates.items() if k != "stream_url"}
    if "stream_url" in updates:
        audit_details["stream_url"] = _mask_stream_url(updates["stream_url"])
    await write_audit_event(
        db, "CAMERA_UPDATED",
        actor_id=current_user.user_id, actor_username=current_user.username,
        resource_type="camera", resource_id=camera_id, details=audit_details,
    )
    if bumps_config:
        await _kick_supervisor()
    return await get_camera(camera_id, current_user, db)


@router.post("/{camera_id}/start")
async def start_camera(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("operator")
    result = await db.execute(
        text("UPDATE cameras SET desired_state = 'running', updated_at = now() WHERE camera_id = :camera_id RETURNING camera_id"),
        {"camera_id": camera_id},
    )
    if not result.fetchone():
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    await db.commit()
    await write_audit_event(
        db, "CAMERA_STARTED",
        actor_id=current_user.user_id, actor_username=current_user.username,
        resource_type="camera", resource_id=camera_id,
    )
    await _kick_supervisor()
    return {"camera_id": camera_id, "desired_state": "running"}


@router.post("/{camera_id}/stop")
async def stop_camera(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("operator")
    result = await db.execute(
        text("UPDATE cameras SET desired_state = 'stopped', updated_at = now() WHERE camera_id = :camera_id RETURNING camera_id"),
        {"camera_id": camera_id},
    )
    if not result.fetchone():
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    await db.commit()
    await write_audit_event(
        db, "CAMERA_STOPPED",
        actor_id=current_user.user_id, actor_username=current_user.username,
        resource_type="camera", resource_id=camera_id,
    )
    await _kick_supervisor()
    return {"camera_id": camera_id, "desired_state": "stopped"}


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Soft-deletes (is_active=false) rather than a hard DELETE: any camera
    that's ever run with persist_events has real zone_presence/tracked_objects
    history referencing it, so a hard delete would either fail on the FK or
    require cascading away genuine history - is_active=false removes it from
    the active list and (since the supervisor's query already filters on
    is_active) stops the orchestrator from ever spawning it again, without
    losing anything."""
    current_user.require_role("admin")
    row = (await db.execute(
        text("SELECT desired_state FROM cameras WHERE camera_id = :camera_id"),
        {"camera_id": camera_id},
    )).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    if row.desired_state == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop the camera before deleting it.",
        )
    await db.execute(
        text("UPDATE cameras SET is_active = false, desired_state = 'stopped', updated_at = now() WHERE camera_id = :camera_id"),
        {"camera_id": camera_id},
    )
    await db.commit()
    await write_audit_event(
        db, "CAMERA_DELETED",
        actor_id=current_user.user_id, actor_username=current_user.username,
        resource_type="camera", resource_id=camera_id,
    )
    return {"message": "Camera deactivated."}


_LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"


@router.get("/{camera_id}/log")
async def get_camera_log(
    camera_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    lines: int = Query(80, le=1000),
) -> dict:
    """Tail of this camera's run_live.py stdout/stderr - the actual reason
    behind a 'crashed' status (bad RTSP credentials, wrong GPU index, model
    load failure, ...). camera_process_status.last_error only has the exit
    code/timing, not the process's own error output; this is that gap.
    Looks the camera up in the DB first (rather than trusting the path
    param directly) so a nonexistent/foreign camera_id can't be used to
    probe the filesystem."""
    row = (await db.execute(
        text("SELECT 1 FROM cameras WHERE camera_id = :camera_id"), {"camera_id": camera_id}
    )).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")

    log_path = _LOG_DIR / f"{camera_id}.log"
    if not log_path.is_file():
        return {"lines": [], "message": "No log file yet - this camera has never been started."}
    text_content = log_path.read_text(errors="replace").splitlines()
    return {"lines": text_content[-lines:]}


@router.websocket("/{camera_id}/status-ws")
async def camera_status_ws(websocket: WebSocket, camera_id: str) -> None:
    """Live push of camera_process_status changes - same ?token= workaround
    as alerts_ws/the mjpeg route, since a WebSocket can't send headers."""
    import jwt as _jwt

    from src.core.config import get_settings

    settings = get_settings()
    token = websocket.query_params.get("token", "")
    try:
        payload = _jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise _jwt.InvalidTokenError("not an access token")
    except _jwt.InvalidTokenError:
        await websocket.close(code=4401, reason="invalid or missing token")
        return

    await websocket.accept()
    channel = f"camera:{camera_id}:process_status"
    redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    try:
        # Send the current state immediately so the UI doesn't wait for the
        # next change to know where things stand.
        async with get_db_session() as session:
            row = (await session.execute(
                text("SELECT state, pid, last_error, restart_count FROM camera_process_status WHERE camera_id = :cid"),
                {"cid": camera_id},
            )).fetchone()
        if row is not None:
            await websocket.send_json({
                "state": row.state, "pid": row.pid,
                "last_error": row.last_error, "restart_count": row.restart_count,
            })
        while True:
            # get_message(timeout=...), not listen() - see the mjpeg
            # route's comment; a camera that's been stable for a while
            # (no status changes) shouldn't kill this connection.
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=20.0)
            if message is None:
                continue
            await websocket.send_text(
                message["data"] if isinstance(message["data"], str) else message["data"].decode()
            )
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


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
            while True:
                # get_message(timeout=...) rather than `async for ... in
                # pubsub.listen()`: listen() blocks indefinitely on the
                # connection's read, which raised redis.exceptions.TimeoutError
                # (interacting with health_check_interval) and silently killed
                # the whole stream whenever frames were more than a few
                # seconds apart (confirmed live - a sparser-FPS camera
                # disconnected within ~10s). get_message has its own
                # bounded wait and just returns None on no-message, so a
                # quiet camera no longer kills the connection.
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=20.0)
                if message is None:
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
