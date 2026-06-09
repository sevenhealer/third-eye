from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.database import get_db

router = APIRouter()


@router.get("")
async def list_events(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    event_type: str | None = Query(None),
    severity: str | None = Query(None),
    zone_id: str | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
) -> dict:
    conditions = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if event_type:
        conditions.append("event_type = :event_type")
        params["event_type"] = event_type
    if severity:
        conditions.append("severity = :severity")
        params["severity"] = severity
    if zone_id:
        conditions.append("zone_id = :zone_id")
        params["zone_id"] = zone_id

    where = " AND ".join(conditions)
    result = await db.execute(
        text(f"""
            SELECT event_id, event_time, event_type, camera_id, zone_id,
                   person_id, severity, confidence, payload
            FROM system_events
            WHERE {where}
            ORDER BY event_time DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = result.fetchall()
    return {
        "total": len(rows),
        "offset": offset,
        "events": [
            {
                "event_id": str(r.event_id),
                "event_time": r.event_time.isoformat(),
                "event_type": r.event_type,
                "camera_id": r.camera_id,
                "zone_id": r.zone_id,
                "person_id": str(r.person_id) if r.person_id else None,
                "severity": r.severity,
                "confidence": r.confidence,
                "payload": r.payload,
            }
            for r in rows
        ],
    }
