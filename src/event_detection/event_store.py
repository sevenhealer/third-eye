from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.core.database import get_db_session
from src.core.logging import get_logger
from src.event_detection.zone_presence import PresenceEvent

logger = get_logger(__name__)


def _ts(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


class EventStore:
    """
    Append-only persistence for the event layer.

    Attribution contract (see sprint3 guide STEP 9): every person event
    carries an identity snapshot in its payload (person, identity_state,
    similarity). Identity corrections are NEW `IDENTITY_CORRECTED` rows
    covering (camera, track, time range) — past rows are never updated,
    keeping the event stream tamper-evident.
    """

    async def write_event(
        self,
        *,
        event_type: str,
        event_time_ns: int,
        camera_id: str,
        zone_id: str | None = None,
        track_id: int | str | None = None,
        person_id: str | None = None,
        severity: str = "INFO",
        confidence: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with get_db_session() as session:
            await session.execute(
                text("""
                    INSERT INTO system_events
                        (event_time, event_type, camera_id, zone_id, person_id,
                         track_id, severity, confidence, payload)
                    VALUES
                        (:ts, :etype, :cam, :zone, :pid, :tid, :sev, :conf,
                         CAST(:payload AS jsonb))
                """),
                {
                    "ts": _ts(event_time_ns),
                    "etype": event_type,
                    "cam": camera_id,
                    "zone": zone_id,
                    "pid": person_id,
                    "tid": str(track_id) if track_id is not None else None,
                    "sev": severity,
                    "conf": confidence,
                    "payload": json.dumps(payload or {}),
                },
            )
            await session.commit()

    async def write_presence_event(self, ev: PresenceEvent) -> None:
        await self.write_event(
            event_type=ev.event_type,
            event_time_ns=ev.timestamp_ns,
            camera_id=ev.camera_id,
            zone_id=ev.zone_id,
            track_id=ev.track_id,
            person_id=ev.person_id,
            severity="INFO",
            confidence=ev.similarity or None,
            payload={
                "person": ev.person_name,
                "identity_state": ev.identity_state,
                "similarity": round(ev.similarity, 3),
            },
        )

    async def open_presence(self, ev: PresenceEvent) -> int | None:
        """zone_presence row on PERSON_ENTERED; returns row id for closing."""
        async with get_db_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO zone_presence
                        (person_id, zone_id, camera_id, entry_time,
                         confidence, track_id, is_unknown)
                    VALUES (:pid, :zone, :cam, :ts, :conf, :tid, :unknown)
                    RETURNING id
                """),
                {
                    "pid": ev.person_id,
                    "zone": ev.zone_id,
                    "cam": ev.camera_id,
                    "ts": _ts(ev.timestamp_ns),
                    "conf": ev.similarity or None,
                    "tid": str(ev.track_id),
                    "unknown": ev.person_id is None,
                },
            )
            await session.commit()
            row = result.fetchone()
            return int(row[0]) if row else None

    async def close_presence(self, presence_id: int, exit_time_ns: int) -> None:
        async with get_db_session() as session:
            await session.execute(
                text("UPDATE zone_presence SET exit_time = :ts WHERE id = :id"),
                {"ts": _ts(exit_time_ns), "id": presence_id},
            )
            await session.commit()

    async def resolve_presence_identity(self, presence_id: int, person_id: str | None) -> None:
        """Patch an open zone_presence row once a track's identity resolves —
        the row was opened as unknown before recognition caught up."""
        async with get_db_session() as session:
            await session.execute(
                text("""
                    UPDATE zone_presence
                    SET person_id = :pid, is_unknown = false
                    WHERE id = :id
                """),
                {"pid": person_id, "id": presence_id},
            )
            await session.commit()

    async def write_identity_correction(
        self,
        *,
        camera_id: str,
        track_id: int,
        from_label: str,
        to_label: str,
        since_ns: int,
        zone_id: str | None = None,
        similarity: float | None = None,
    ) -> None:
        """Append-only re-attribution: events for (camera, track) in
        [since, now] were attributed to `from_label`; the system now
        believes `to_label`. Consumers (NLQ, dashboards) resolve identity
        through these records."""
        now_ns = time.time_ns()
        await self.write_event(
            event_type="IDENTITY_CORRECTED",
            event_time_ns=now_ns,
            camera_id=camera_id,
            zone_id=zone_id,
            track_id=track_id,
            severity="LOW",
            confidence=similarity,
            payload={
                "from": from_label,
                "to": to_label,
                "applies_from": _ts(since_ns).isoformat(),
                "applies_to": _ts(now_ns).isoformat(),
            },
        )

    async def write_object_counts(
        self, rows: list[tuple[datetime, str, str | None, str, int]]
    ) -> None:
        """rows: (bucket_time, camera_id, zone_id, object_class, count)."""
        if not rows:
            return
        async with get_db_session() as session:
            for bucket_time, camera_id, zone_id, object_class, count in rows:
                await session.execute(
                    text("""
                        INSERT INTO object_counts
                            (bucket_time, camera_id, zone_id, object_class, count)
                        VALUES (:ts, :cam, :zone, :cls, :n)
                        ON CONFLICT (bucket_time, camera_id, object_class)
                        DO UPDATE SET count = GREATEST(object_counts.count, EXCLUDED.count)
                    """),
                    {"ts": bucket_time, "cam": camera_id, "zone": zone_id,
                     "cls": object_class, "n": count},
                )
            await session.commit()

    async def write_alert(
        self,
        *,
        event_type: str,
        severity: str,
        camera_id: str,
        zone_id: str | None,
        person_id: str | None,
        description: str,
        payload: dict[str, Any] | None = None,
    ) -> str | None:
        """Insert an alerts row; returns alert_id."""
        async with get_db_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO alerts
                        (event_type, severity, camera_id, zone_id, person_id,
                         description, payload)
                    VALUES (:etype, :sev, :cam, :zone, :pid, :descr,
                            CAST(:payload AS jsonb))
                    RETURNING alert_id
                """),
                {
                    "etype": event_type,
                    "sev": severity,
                    "cam": camera_id,
                    "zone": zone_id,
                    "pid": person_id,
                    "descr": description,
                    "payload": json.dumps(payload or {}),
                },
            )
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else None

    async def ensure_zone(self, zone_id: str, zone_type: str = "general") -> None:
        """Self-register the zone so zone_presence's FK is satisfied. Existing
        zones (incl. manually-seeded 'restricted' ones) are left untouched."""
        async with get_db_session() as session:
            await session.execute(
                text("""
                    INSERT INTO zones (zone_id, display_name, zone_type)
                    VALUES (:z, :z, :zt)
                    ON CONFLICT (zone_id) DO NOTHING
                """),
                {"z": zone_id, "zt": zone_type},
            )
            await session.commit()

    async def ensure_camera(
        self, camera_id: str, stream_url: str = "", zone_id: str | None = None
    ) -> None:
        """Self-register the camera so zone_presence's camera_id FK is satisfied.
        Without this, the very first PERSON_ENTERED's presence insert throws a
        foreign-key violation and aborts event handling for the frame."""
        async with get_db_session() as session:
            await session.execute(
                text("""
                    INSERT INTO cameras (camera_id, display_name, stream_url, zone_id)
                    VALUES (:cid, :name, :url, :zone)
                    ON CONFLICT (camera_id) DO NOTHING
                """),
                {"cid": camera_id, "name": camera_id,
                 "url": stream_url or camera_id, "zone": zone_id},
            )
            await session.commit()

    async def load_zone_types(self) -> dict[str, str]:
        """zone_id -> zone_type, for alert rule evaluation."""
        async with get_db_session() as session:
            rows = (await session.execute(
                text("SELECT zone_id, zone_type FROM zones")
            )).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}
