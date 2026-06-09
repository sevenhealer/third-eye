from __future__ import annotations

import re
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.core.audit_log import write_audit_event
from src.core.database import get_db
from src.core.exceptions import PromptInjectionError
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Characters and patterns to strip before sending to query engine
_INJECTION_PATTERNS = re.compile(
    r"(--|;|DROP\s|DELETE\s|INSERT\s|UPDATE\s|UNION\s|EXEC\s|"
    r"MATCH\s.*DETACH|CREATE\s|MERGE\s|<script|javascript:)",
    re.IGNORECASE,
)


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def sanitize(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 500:
            raise ValueError("Query must be 500 characters or fewer.")
        if _INJECTION_PATTERNS.search(v):
            raise PromptInjectionError("Query contains disallowed patterns.")
        return v


class QueryResponse(BaseModel):
    query_id: str
    query_text: str
    answer: str
    confidence: float
    latency_ms: int
    data_sources: list[str]


@router.post("", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    current_user.require_role("operator")

    start = time.perf_counter()
    query_id = str(uuid4())

    # Delegate to NL query engine (imported lazily to avoid circular deps)
    from src.nlq.query_planner import execute_nl_query

    result = await execute_nl_query(
        query_text=body.query,
        user_id=current_user.user_id,
        user_role=current_user.role,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    await db.execute(
        text("""
            INSERT INTO query_history
                (query_id, user_id, query_text, intent, answer_text,
                 confidence, latency_ms, data_sources)
            VALUES
                (:qid, :uid, :query, :intent, :answer,
                 :confidence, :latency, :sources::jsonb)
        """),
        {
            "qid": query_id,
            "uid": str(current_user.user_id),
            "query": body.query,
            "intent": result.get("intent"),
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "latency": latency_ms,
            "sources": str(result.get("data_sources", [])).replace("'", '"'),
        },
    )
    await db.commit()

    await write_audit_event(
        db,
        "NL_QUERY_SUBMITTED",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="query",
        resource_id=query_id,
        details={"query": body.query, "intent": result.get("intent")},
    )

    return QueryResponse(
        query_id=query_id,
        query_text=body.query,
        answer=result["answer"],
        confidence=result.get("confidence", 0.0),
        latency_ms=latency_ms,
        data_sources=result.get("data_sources", []),
    )


@router.get("/history")
async def query_history(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
) -> dict:
    result = await db.execute(
        text("""
            SELECT query_id, query_text, answer_text, confidence,
                   latency_ms, queried_at
            FROM query_history
            WHERE user_id = :uid
            ORDER BY queried_at DESC
            LIMIT :limit
        """),
        {"uid": str(current_user.user_id), "limit": limit},
    )
    rows = result.fetchall()
    return {
        "queries": [
            {
                "query_id": str(r.query_id),
                "query_text": r.query_text,
                "answer_text": r.answer_text,
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "queried_at": r.queried_at.isoformat(),
            }
            for r in rows
        ]
    }
