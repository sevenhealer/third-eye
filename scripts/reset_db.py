#!/usr/bin/env python3
"""Wipe all Third-Eye runtime data for a clean-slate test run.

Clears every store the app writes to, reading connection settings from the
app's own config (.env), so it stays in sync with the running system:

  - PostgreSQL : truncates every data table (schema + alembic_version kept),
                 then re-seeds the default zones and the admin/admin login.
  - Redis      : flushes the app DB (zone occupants/counts, camera heartbeats).
  - MinIO/S3   : empties the evidence bucket (face crops / candidate images).
  - Neo4j      : detaches and deletes the entire graph.

Schema is preserved (TRUNCATE, not DROP) so the applied migrations stay in
place. Destructive and irreversible — for dev/test resets only.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Default zones + admin seed, kept in lockstep with
# infrastructure/postgres/init.sql so a reset matches a fresh DB init.
DEFAULT_ZONES = [
    ("building_entrance", "Building Entrance", "entrance", "Main building entry/exit"),
    ("room_a", "Room A", "general", "General purpose room"),
    ("server_room", "Server Room", "restricted", "Data center / server racks"),
    ("corridor_main", "Main Corridor", "monitored", "Primary hallway"),
]


async def reset_postgres() -> None:
    from sqlalchemy import text

    from src.api.auth import hash_password
    from src.core.database import get_db_session

    async with get_db_session() as session:
        tables = (await session.execute(text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
        ))).scalars().all()
        if tables:
            joined = ", ".join(f'"{t}"' for t in tables)
            # CASCADE resolves FK order; RESTART IDENTITY resets serial PKs.
            await session.execute(text(f"TRUNCATE {joined} RESTART IDENTITY CASCADE"))

        await session.execute(
            text(
                "INSERT INTO zones (zone_id, display_name, zone_type, description) "
                "VALUES (:zid, :name, :ztype, :descr) ON CONFLICT (zone_id) DO NOTHING"
            ),
            [{"zid": z, "name": n, "ztype": t, "descr": d} for z, n, t, d in DEFAULT_ZONES],
        )
        await session.execute(
            text(
                "INSERT INTO users (username, email, hashed_pw, role) "
                "VALUES ('admin', 'admin@thirdeye.local', :pw, 'admin') "
                "ON CONFLICT (username) DO NOTHING"
            ),
            {"pw": hash_password("admin")},
        )
        await session.commit()
        print(f"  postgres : truncated {len(tables)} tables, re-seeded "
              f"{len(DEFAULT_ZONES)} zones + admin/admin")


async def reset_redis() -> None:
    from src.core.database import get_redis

    redis = await get_redis()
    await redis.flushdb()
    print("  redis    : flushed")


def reset_minio() -> None:
    from src.core.config import get_settings

    settings = get_settings()
    if not settings.s3_endpoint_url:
        print("  minio    : object storage disabled, skipped")
        return
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
    )
    try:
        deleted = 0
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=settings.s3_bucket):
            contents = page.get("Contents", [])
            if contents:
                client.delete_objects(
                    Bucket=settings.s3_bucket,
                    Delete={"Objects": [{"Key": o["Key"]} for o in contents]},
                )
                deleted += len(contents)
        print(f"  minio    : deleted {deleted} object(s) from {settings.s3_bucket}")
    except Exception as exc:  # bucket missing / endpoint down — non-fatal for a reset
        print(f"  minio    : skipped ({exc})")


async def reset_neo4j() -> None:
    # The configured URI uses the docker-internal hostname ("neo4j"), which a
    # native process can't resolve; the container publishes bolt on localhost,
    # so rewrite the host for the host-side reset.
    from neo4j import AsyncGraphDatabase

    from src.core.config import get_settings

    settings = get_settings()
    uri = settings.neo4j_uri.replace("//neo4j:", "//localhost:")
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(settings.neo4j_user, settings.neo4j_password))
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await driver.close()
        print("  neo4j    : graph cleared")
    except Exception as exc:  # neo4j optional for core flows — non-fatal
        print(f"  neo4j    : skipped ({exc})")


async def main() -> None:
    print("Resetting Third-Eye data stores (DESTRUCTIVE) ...")
    await reset_postgres()
    await reset_redis()
    reset_minio()
    await reset_neo4j()
    from src.core.database import close_all

    await close_all()
    print("Done. Login is  admin / admin")


if __name__ == "__main__":
    asyncio.run(main())
