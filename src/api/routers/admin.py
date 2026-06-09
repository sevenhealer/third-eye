from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import verify_audit_chain
from src.core.database import get_db
from src.core.gpu_manager import get_gpu_manager
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/audit-log")
async def get_audit_log(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
) -> dict:
    current_user.require_role("security_officer")

    result = await db.execute(
        text("""
            SELECT log_id, event_time, actor_username, action_type,
                   resource_type, resource_id, details, current_hash
            FROM audit_log
            ORDER BY log_id DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": limit, "offset": offset},
    )
    rows = result.fetchall()
    return {
        "entries": [
            {
                "log_id": r.log_id,
                "event_time": r.event_time.isoformat(),
                "actor_username": r.actor_username,
                "action_type": r.action_type,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "details": r.details,
                "hash": r.current_hash,
            }
            for r in rows
        ]
    }


@router.get("/audit-log/verify")
async def verify_audit(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("security_officer")

    is_valid, first_invalid = await verify_audit_chain(db)
    return {
        "chain_valid": is_valid,
        "first_invalid_log_id": first_invalid if not is_valid else None,
    }


@router.get("/models")
async def list_models(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    current_user.require_role("admin")

    result = await db.execute(
        text("""
            SELECT model_id, model_name, model_type, version,
                   stage, metrics, deployed_at
            FROM model_registry
            ORDER BY model_name, version DESC
        """)
    )
    rows = result.fetchall()
    gpu_status = get_gpu_manager().status()
    return {
        "gpu": {
            "allocated_mb": gpu_status["allocated_mb"],
            "free_mb": gpu_status["free_mb"],
            "total_mb": gpu_status["total_vram_mb"],
        },
        "models": [
            {
                "model_id": str(r.model_id),
                "model_name": r.model_name,
                "version": r.version,
                "stage": r.stage,
                "metrics": r.metrics,
                "deployed_at": r.deployed_at.isoformat() if r.deployed_at else None,
            }
            for r in rows
        ],
    }


@router.get("/system-health")
async def system_health(current_user: ActiveUser) -> dict:
    current_user.require_role("operator")
    gpu_status = get_gpu_manager().status()
    return {
        "status": "ok",
        "gpu": gpu_status,
    }
