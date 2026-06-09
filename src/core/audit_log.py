from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger

logger = get_logger(__name__)


async def write_audit_event(
    db: AsyncSession,
    action_type: str,
    *,
    actor_id: UUID | None = None,
    actor_username: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
    source_ip: str | None = None,
) -> int:
    """
    Append an entry to the hash-chained audit log.
    Returns the new log_id.
    The hash chain is computed by PostgreSQL's generated column.
    """
    import json

    result = await db.execute(
        text("""
            INSERT INTO audit_log
                (actor_id, actor_username, action_type,
                 resource_type, resource_id, details, source_ip,
                 prev_hash)
            VALUES
                (:actor_id, :actor_username, :action_type,
                 :resource_type, :resource_id, :details::jsonb, :source_ip,
                 (SELECT COALESCE(
                     (SELECT current_hash FROM audit_log ORDER BY log_id DESC LIMIT 1),
                     '0000000000000000000000000000000000000000000000000000000000000000'
                 ))
                )
            RETURNING log_id
        """),
        {
            "actor_id": str(actor_id) if actor_id else None,
            "actor_username": actor_username,
            "action_type": action_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": json.dumps(details or {}),
            "source_ip": source_ip,
        },
    )
    await db.commit()
    log_id = result.scalar_one()
    logger.debug("audit_event_written", action=action_type, log_id=log_id)
    return log_id


async def verify_audit_chain(db: AsyncSession) -> tuple[bool, int]:
    """
    Verify the hash chain integrity of the audit log.
    Returns (is_valid, first_invalid_log_id).
    Returns (True, -1) if chain is intact.
    """
    rows = await db.execute(
        text("""
            SELECT log_id, current_hash, prev_hash, actor_id,
                   action_type, resource_type, resource_id,
                   details::text, event_time::text
            FROM audit_log
            ORDER BY log_id ASC
        """)
    )
    entries = rows.fetchall()

    if not entries:
        return True, -1

    prev_hash = "0" * 64
    for row in entries:
        log_id, stored_hash, stored_prev, actor_id, action_type, \
            resource_type, resource_id, details, event_time = row

        if stored_prev != prev_hash:
            logger.error(
                "audit_chain_broken",
                log_id=log_id,
                expected_prev=prev_hash,
                stored_prev=stored_prev,
            )
            return False, log_id

        # Recompute expected hash
        raw = (
            str(log_id)
            + str(event_time)
            + str(actor_id or "")
            + str(action_type or "")
            + str(resource_type or "")
            + str(resource_id or "")
            + str(details or "")
            + str(stored_prev or "")
        )
        expected = hashlib.sha256(raw.encode()).hexdigest()

        if expected != stored_hash:
            logger.error(
                "audit_hash_mismatch",
                log_id=log_id,
                expected=expected,
                stored=stored_hash,
            )
            return False, log_id

        prev_hash = stored_hash

    logger.info("audit_chain_verified", total_entries=len(entries))
    return True, -1
