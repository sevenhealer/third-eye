from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db

router = APIRouter()


@router.get("")
async def list_alerts(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    alert_status: str | None = Query(None, alias="status"),
    severity: str | None = Query(None),
    limit: int = Query(50, le=200),
) -> dict:
    conditions = ["1=1"]
    params: dict = {"limit": limit}

    if alert_status:
        conditions.append("status = :status")
        params["status"] = alert_status
    if severity:
        conditions.append("severity = :severity")
        params["severity"] = severity

    where = " AND ".join(conditions)
    result = await db.execute(
        text(f"""
            SELECT alert_id, event_type, severity, camera_id, zone_id,
                   person_id, description, triggered_at, status
            FROM alerts
            WHERE {where}
            ORDER BY triggered_at DESC
            LIMIT :limit
        """),
        params,
    )
    rows = result.fetchall()
    return {
        "alerts": [
            {
                "alert_id": str(r.alert_id),
                "event_type": r.event_type,
                "severity": r.severity,
                "camera_id": r.camera_id,
                "zone_id": r.zone_id,
                "person_id": str(r.person_id) if r.person_id else None,
                "description": r.description,
                "triggered_at": r.triggered_at.isoformat(),
                "status": r.status,
            }
            for r in rows
        ]
    }


@router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: UUID,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("operator")

    result = await db.execute(
        text("""
            UPDATE alerts
            SET status = 'acknowledged',
                acknowledged_at = now(),
                acknowledged_by = :uid
            WHERE alert_id = :aid AND status = 'open'
            RETURNING alert_id
        """),
        {"aid": str(alert_id), "uid": str(current_user.user_id)},
    )
    if not result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found or already acknowledged.",
        )
    await db.commit()
    await write_audit_event(
        db, "ALERT_ACKNOWLEDGED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="alert",
        resource_id=str(alert_id),
    )
    return {"message": "Alert acknowledged."}


@router.put("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: UUID,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("operator")

    result = await db.execute(
        text("""
            UPDATE alerts
            SET status = 'resolved',
                resolved_at = now(),
                resolved_by = :uid
            WHERE alert_id = :aid AND status IN ('open','acknowledged')
            RETURNING alert_id
        """),
        {"aid": str(alert_id), "uid": str(current_user.user_id)},
    )
    if not result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found or already resolved.",
        )
    await db.commit()
    await write_audit_event(
        db, "ALERT_RESOLVED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="alert",
        resource_id=str(alert_id),
    )
    return {"message": "Alert resolved."}
