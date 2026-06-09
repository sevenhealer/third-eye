from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from src.api.auth import (
    ActiveUser,
    create_access_token,
    create_refresh_token,
    verify_password,
)
from src.core.audit_log import write_audit_event
from src.core.database import get_db
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(
        text("SELECT user_id, hashed_pw, role, is_active FROM users WHERE username = :u"),
        {"u": form.username},
    )
    row = result.fetchone()

    if not row or not verify_password(form.password, row.hashed_pw) or not row.is_active:
        await write_audit_event(
            db,
            "LOGIN_FAILED",
            actor_username=form.username,
            details={"reason": "invalid credentials"},
            source_ip=request.client.host if request.client else None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = {"sub": str(row.user_id), "username": form.username, "role": row.role}
    await write_audit_event(
        db,
        "LOGIN_SUCCESS",
        actor_id=row.user_id,
        actor_username=form.username,
        source_ip=request.client.host if request.client else None,
    )
    await db.execute(
        text("UPDATE users SET last_login = now() WHERE user_id = :uid"),
        {"uid": row.user_id},
    )
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(payload),
        refresh_token=create_refresh_token(payload),
    )


@router.post("/logout")
async def logout(
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    await write_audit_event(
        db,
        "LOGOUT",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
    )
    return {"message": "Logged out."}


@router.get("/me")
async def me(current_user: ActiveUser) -> dict:
    return {
        "user_id": str(current_user.user_id),
        "username": current_user.username,
        "role": current_user.role,
    }
