from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from src.core.database import get_redis
from src.core.logging import get_logger

logger = get_logger(__name__)

# Key TTLs (seconds)
TTL_IDENTITY_STATE = 600      # 10 minutes
TTL_ZONE_OCCUPANCY = 60       # 1 minute
TTL_CAMERA_STATUS = 120       # 2 minutes
TTL_ALERT_STATE = 3600        # 1 hour


class ShortTermMemory:
    """
    Redis-backed short-term memory for live system state.

    Keys:
      identity:{person_id}:location   → zone_id string
      identity:{person_id}:action     → action string
      identity:{person_id}:track_id   → track_id string
      zone:{zone_id}:count            → integer
      zone:{zone_id}:occupants        → JSON list of person_id strings
      camera:{camera_id}:status       → status string
      name:{lower_name}:person_id     → UUID string  (name lookup)
    """

    async def set_identity_location(
        self, person_id: UUID | str, zone_id: str, track_id: str | None = None
    ) -> None:
        redis = await get_redis()
        pid = str(person_id)
        async with redis.pipeline(transaction=True) as pipe:
            pipe.setex(f"identity:{pid}:location", TTL_IDENTITY_STATE, zone_id)
            if track_id:
                pipe.setex(f"identity:{pid}:track_id", TTL_IDENTITY_STATE, track_id)
            await pipe.execute()

    async def set_identity_action(self, person_id: UUID | str, action: str) -> None:
        redis = await get_redis()
        await redis.setex(
            f"identity:{str(person_id)}:action", TTL_IDENTITY_STATE, action
        )

    async def get_identity_state(self, person_id: UUID | str) -> dict[str, str | None]:
        redis = await get_redis()
        pid = str(person_id)
        location, action, track_id = await redis.mget(
            f"identity:{pid}:location",
            f"identity:{pid}:action",
            f"identity:{pid}:track_id",
        )
        return {"location": location, "action": action, "track_id": track_id}

    async def add_zone_occupant(self, zone_id: str, person_id: UUID | str) -> None:
        redis = await get_redis()
        key = f"zone:{zone_id}:occupants"
        raw = await redis.get(key)
        occupants: list[str] = json.loads(raw) if raw else []
        pid = str(person_id)
        if pid not in occupants:
            occupants.append(pid)
        await redis.setex(key, TTL_ZONE_OCCUPANCY, json.dumps(occupants))
        await redis.setex(f"zone:{zone_id}:count", TTL_ZONE_OCCUPANCY, str(len(occupants)))

    async def remove_zone_occupant(self, zone_id: str, person_id: UUID | str) -> None:
        redis = await get_redis()
        key = f"zone:{zone_id}:occupants"
        raw = await redis.get(key)
        occupants: list[str] = json.loads(raw) if raw else []
        pid = str(person_id)
        occupants = [o for o in occupants if o != pid]
        await redis.setex(key, TTL_ZONE_OCCUPANCY, json.dumps(occupants))
        await redis.setex(f"zone:{zone_id}:count", TTL_ZONE_OCCUPANCY, str(len(occupants)))

    async def get_zone_occupants(self, zone_id: str) -> list[str]:
        redis = await get_redis()
        raw = await redis.get(f"zone:{zone_id}:occupants")
        return json.loads(raw) if raw else []

    async def set_zone_object_count(
        self, zone_id: str, object_class: str, count: int
    ) -> None:
        redis = await get_redis()
        await redis.setex(
            f"zone:{zone_id}:count:{object_class}", TTL_ZONE_OCCUPANCY, str(count)
        )

    async def set_camera_status(self, camera_id: str, status: str) -> None:
        redis = await get_redis()
        await redis.setex(f"camera:{camera_id}:status", TTL_CAMERA_STATUS, status)

    async def register_name_lookup(self, name: str, person_id: UUID | str) -> None:
        redis = await get_redis()
        await redis.set(f"name:{name.lower()}:person_id", str(person_id))

    async def lookup_person_by_name(self, name: str) -> str | None:
        redis = await get_redis()
        return await redis.get(f"name:{name.lower()}:person_id")


_stm: ShortTermMemory | None = None


def get_short_term_memory() -> ShortTermMemory:
    global _stm
    if _stm is None:
        _stm = ShortTermMemory()
    return _stm
