from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from neo4j import AsyncDriver, AsyncGraphDatabase
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# ── SQLAlchemy (PostgreSQL + TimescaleDB) ─────────────────────────────────────

class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.db_echo,   # opt-in; was flooding logs in development
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None
_redis_binary_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            password=settings.redis_password or None,
            decode_responses=True,
            health_check_interval=30,
        )
    return _redis_client


async def get_redis_binary() -> aioredis.Redis:
    """Separate client with decode_responses=False, for payloads that
    aren't valid UTF-8 (e.g. JPEG frame pub/sub — see
    ShortTermMemory.publish_frame and cameras.py's /mjpeg route). The
    default client above decodes every response as text, which crashes
    on binary data; everything else in the codebase is string-keyed, so
    that client is left alone rather than flipping decode_responses
    globally."""
    global _redis_binary_client
    if _redis_binary_client is None:
        settings = get_settings()
        _redis_binary_client = aioredis.from_url(
            settings.redis_url,
            password=settings.redis_password or None,
            decode_responses=False,
            health_check_interval=30,
        )
    return _redis_binary_client


# ── Neo4j ──────────────────────────────────────────────────────────────────────

_neo4j_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    global _neo4j_driver
    if _neo4j_driver is None:
        settings = get_settings()
        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=20,
        )
    return _neo4j_driver


async def close_all() -> None:
    global _engine, _redis_client, _neo4j_driver
    if _engine:
        await _engine.dispose()
        _engine = None
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
    if _neo4j_driver:
        await _neo4j_driver.close()
        _neo4j_driver = None
    logger.info("database_connections_closed")
