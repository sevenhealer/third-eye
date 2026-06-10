"""
Tests for src/api/auth.py — pure-function coverage (no DB, no HTTP).
"""
import os
import time
import uuid

import jwt
import pytest

os.environ.setdefault("APP_SECRET_KEY", "testsecrettestsecrettestsecrettestse")
os.environ.setdefault("JWT_SECRET_KEY", "testjwttestjwttestjwttestjwttestjwttest")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("TIMESCALE_URL", "postgresql+asyncpg://u:p@localhost/ts")
os.environ.setdefault("NEO4J_PASSWORD", "secret")
os.environ.setdefault("POSTGRES_PASSWORD", "secret")

from src.api.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
    ROLE_HIERARCHY,
)
from src.core.config import get_settings
from src.core.exceptions import AuthorizationError

settings = get_settings()


# ── password hashing ─────────────────────────────────────────────────────────

def test_hash_password_is_not_plaintext():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert len(hashed) > 20


def test_verify_password_correct():
    pw = "correcthorsebatterystaple"
    assert verify_password(pw, hash_password(pw)) is True


def test_verify_password_wrong():
    assert verify_password("wrong", hash_password("right")) is False


# ── JWT tokens ───────────────────────────────────────────────────────────────

def test_access_token_has_type_access():
    payload = {"sub": str(uuid.uuid4()), "username": "alice", "role": "operator"}
    token = create_access_token(payload)
    decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert decoded["type"] == "access"


def test_refresh_token_has_type_refresh():
    payload = {"sub": str(uuid.uuid4()), "username": "bob", "role": "readonly"}
    token = create_refresh_token(payload)
    decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert decoded["type"] == "refresh"


def test_access_token_carries_username():
    payload = {"sub": str(uuid.uuid4()), "username": "carol", "role": "analyst"}
    token = create_access_token(payload)
    decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert decoded["username"] == "carol"


def test_token_has_future_expiry():
    payload = {"sub": str(uuid.uuid4()), "username": "dave", "role": "admin"}
    token = create_access_token(payload)
    decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert decoded["exp"] > time.time()


# ── RBAC ─────────────────────────────────────────────────────────────────────

def _user(role: str) -> CurrentUser:
    return CurrentUser(
        user_id=uuid.uuid4(),
        username="test",
        role=role,
        is_active=True,
    )


def test_require_role_passes_equal():
    _user("operator").require_role("operator")   # should not raise


def test_require_role_passes_higher():
    _user("admin").require_role("analyst")   # admin > analyst


def test_require_role_fails_lower():
    with pytest.raises(AuthorizationError):
        _user("readonly").require_role("operator")


def test_role_hierarchy_correct_order():
    expected = ["readonly", "operator", "analyst", "admin", "security_officer"]
    assert ROLE_HIERARCHY == expected
