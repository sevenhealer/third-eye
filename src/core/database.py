from __future__ import annotations

from typing import AsyncIterator

import redis.asyncio as aioredis
from neo4j import AsyncGraphDatabase, AsyncDriver
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
            echo=not settings.is_production,
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


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None


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
