from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.core.database import get_db_session, get_redis
from src.core.logging import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_LIVE_PATH = ROOT / "scripts" / "run_live.py"
LOG_DIR = ROOT / "data" / "logs"

POLL_INTERVAL_S = 5
GRACEFUL_STOP_TIMEOUT_S = 10
# A child that dies within this long of being spawned is a launch failure
# (bad RTSP URL, OOM, missing model), not a normal stop - feed the crash
# backoff instead of respawning in a tight loop.
CRASH_WINDOW_S = 15
BACKOFF_SCHEDULE_S = [30, 60, 120, 240, 300]
KICK_CHANNEL = "camera:control:kick"

# camera_id already advertising a fresh camera:{id}:status heartbeat (written
# by run_live.py's stm.set_camera_status, TTL 120s) but with no PID tracked
# by us means someone launched this camera_id manually outside the
# orchestrator - refuse to spawn a competing second process for it.
EXTERNAL_HEARTBEAT_FRESH_S = 30


class _CrashState:
    __slots__ = ("consecutive_fails", "next_retry_at")

    def __init__(self) -> None:
        self.consecutive_fails = 0
        self.next_retry_at = 0.0


class CameraSupervisor:
    """Reconciles the `cameras` table's desired_state/config_version against
    real run_live.py subprocesses - what makes editing a camera in Settings
    actually do something, instead of just changing a DB row nobody reads.

    Poll-driven (every POLL_INTERVAL_S) with a Redis pub/sub "kick" on top
    for near-immediate reaction to an API-initiated change; the poll loop
    stays authoritative so a missed pub/sub message (no persistence in
    Redis pub/sub) self-heals within one interval instead of getting stuck.
    """

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watch_tasks: dict[str, asyncio.Task] = {}
        self._launched_config_version: dict[str, int] = {}
        self._launched_at: dict[str, float] = {}
        self._intentional_stop: set[str] = set()
        self._crash_state: dict[str, _CrashState] = {}
        self._kick_event = asyncio.Event()
        self._shutdown = False
        self._loop_task: asyncio.Task | None = None
        self._kick_listener_task: asyncio.Task | None = None
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._loop(), name="camera_supervisor_loop")
        self._kick_listener_task = asyncio.create_task(
            self._listen_for_kicks(), name="camera_supervisor_kick_listener"
        )
        logger.info("camera_supervisor_started", poll_interval_s=POLL_INTERVAL_S)

    async def stop(self) -> None:
        self._shutdown = True
        self._kick_event.set()
        for task in (self._loop_task, self._kick_listener_task):
            if task is not None:
                task.cancel()
        # Stop every child we're tracking so a uvicorn restart doesn't orphan
        # subprocesses with nothing left to reconcile them.
        for camera_id, proc in list(self._processes.items()):
            await self._stop_process(camera_id, proc)
        logger.info("camera_supervisor_stopped")

    async def _listen_for_kicks(self) -> None:
        redis = await get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(KICK_CHANNEL)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                self._kick_event.set()
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(KICK_CHANNEL)
            await pubsub.close()

    async def _loop(self) -> None:
        while not self._shutdown:
            try:
                await self._reconcile_once()
            except Exception:
                logger.exception("camera_supervisor_reconcile_failed")
            try:
                await asyncio.wait_for(self._kick_event.wait(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
            self._kick_event.clear()

    async def _reconcile_once(self) -> None:
        async with get_db_session() as session:
            rows = (await session.execute(text(
                """
                SELECT camera_id, display_name, stream_url, zone_id, gpu_id,
                       desired_state, config_version, launch_args
                FROM cameras
                WHERE is_active = true
                """
            ))).fetchall()
        for row in rows:
            try:
                await self._reconcile_camera(row)
            except Exception:
                logger.exception("camera_supervisor_camera_reconcile_failed", camera_id=row.camera_id)

    async def _reconcile_camera(self, row: Any) -> None:
        camera_id = row.camera_id
        proc = self._processes.get(camera_id)
        alive = proc is not None and proc.returncode is None

        if row.desired_state == "stopped":
            if alive:
                await self._stop_process(camera_id, proc)
                await self._set_status(camera_id, state="stopped", pid=None, last_error=None)
            return

        # desired_state == "running"
        if alive:
            if self._launched_config_version.get(camera_id) != row.config_version:
                logger.info("camera_supervisor_config_changed_restarting", camera_id=camera_id)
                await self._stop_process(camera_id, proc)
                await self._spawn(row)
            return

        crash = self._crash_state.get(camera_id)
        if crash is not None and time.monotonic() < crash.next_retry_at:
            return  # still backing off after repeated crashes

        if await self._already_running_externally(camera_id):
            # The camera IS running - we just didn't spawn this instance of
            # it (e.g. a prior supervisor started it and the API server was
            # since restarted, orphaning the child). Reflect the truth:
            # "running", not "crashed". We don't own the OS process so the
            # pid is unknown, but the operator's status pill must not lie.
            await self._set_status(
                camera_id, state="running", pid=None, last_error=None,
                last_heartbeat_at=datetime.now(timezone.utc),
                config_version_applied=row.config_version,
            )
            logger.info("camera_supervisor_external_process_observed", camera_id=camera_id)
            return

        await self._spawn(row)

    async def _already_running_externally(self, camera_id: str) -> bool:
        redis = await get_redis()
        status = await redis.get(f"camera:{camera_id}:status")
        ttl = await redis.ttl(f"camera:{camera_id}:status")
        # status TTL is 120s (TTL_CAMERA_STATUS); a value with most of that
        # left was refreshed recently by a process we're not tracking.
        return status is not None and ttl > (120 - EXTERNAL_HEARTBEAT_FRESH_S)

    def _build_argv(self, row: Any) -> list[str]:
        launch_args: dict = row.launch_args or {}
        argv = [
            sys.executable, str(RUN_LIVE_PATH),
            "--source", row.stream_url,
            "--camera-id", row.camera_id,
        ]
        if row.zone_id:
            argv += ["--zone-id", row.zone_id]
        if row.gpu_id is not None:
            argv += ["--gpu", str(row.gpu_id)]
        if launch_args.get("cpu"):
            argv += ["--cpu"]
        if launch_args.get("recognize"):
            argv += ["--recognize"]
        if launch_args.get("persist_events"):
            argv += ["--persist-events"]
        if launch_args.get("fps") is not None:
            argv += ["--fps", str(launch_args["fps"])]
        if launch_args.get("det_size"):
            argv += ["--det-size", str(launch_args["det_size"])]
        if launch_args.get("det_thresh") is not None:
            argv += ["--det-thresh", str(launch_args["det_thresh"])]
        if launch_args.get("reid_thresh") is not None:
            argv += ["--reid-thresh", str(launch_args["reid_thresh"])]
        if launch_args.get("min_face") is not None:
            argv += ["--min-face", str(launch_args["min_face"])]
        if launch_args.get("enforce_liveness"):
            argv += ["--enforce-liveness"]
        if launch_args.get("collect_antispoofing"):
            argv += ["--collect-antispoofing"]
        if launch_args.get("ignore_regions"):
            import json
            argv += ["--ignore-regions", json.dumps(launch_args["ignore_regions"])]
        # --show opens a cv2 GUI window - meaningless and dangerous on a
        # headless server process. --bypass-antispoofing skips liveness
        # checks entirely. Neither is ever accepted into launch_args at the
        # API layer (CameraLaunchArgs forbids extra fields), but the
        # orchestrator itself never passes them either, as a second layer.
        return argv

    async def _spawn(self, row: Any) -> None:
        camera_id = row.camera_id
        argv = self._build_argv(row)
        log_path = LOG_DIR / f"{camera_id}.log"
        log_file = open(log_path, "ab")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=log_file, stderr=asyncio.subprocess.STDOUT,
                cwd=str(ROOT),
            )
        finally:
            log_file.close()
        self._processes[camera_id] = proc
        self._launched_config_version[camera_id] = row.config_version
        self._launched_at[camera_id] = time.monotonic()
        self._watch_tasks[camera_id] = asyncio.create_task(
            self._watch(camera_id, proc), name=f"camera_watch_{camera_id}"
        )
        await self._set_status(
            camera_id, state="starting", pid=proc.pid, gpu_id=row.gpu_id,
            started_at=datetime.now(timezone.utc),
            config_version_applied=row.config_version, last_error=None,
        )
        logger.info("camera_supervisor_spawned", camera_id=camera_id, pid=proc.pid, gpu_id=row.gpu_id)
        # Give it a moment, then flip starting -> running if it's still alive
        # (cheap optimism: a real health signal would need run_live.py to
        # report readiness, which is out of scope here - "didn't immediately
        # die" is good enough signal for the Settings UI's status pill).
        await asyncio.sleep(2)
        if proc.returncode is None:
            await self._set_status(
                camera_id, state="running", pid=proc.pid,
                last_heartbeat_at=datetime.now(timezone.utc),
            )

    async def _watch(self, camera_id: str, proc: asyncio.subprocess.Process) -> None:
        await proc.wait()
        self._processes.pop(camera_id, None)
        self._watch_tasks.pop(camera_id, None)
        uptime = time.monotonic() - self._launched_at.pop(camera_id, time.monotonic())
        intentional = camera_id in self._intentional_stop
        self._intentional_stop.discard(camera_id)

        if intentional:
            await self._set_status(
                camera_id, state="stopped", pid=None, last_exit_code=proc.returncode,
                last_error=None, restart_count=0,
            )
            self._crash_state.pop(camera_id, None)
            return

        # Died on its own. If it ran for a while first, treat this one exit
        # as a fresh start (reset backoff) - only a *rapid* repeat crash is
        # actually a crash loop worth backing off from.
        crash = self._crash_state.setdefault(camera_id, _CrashState())
        if uptime < CRASH_WINDOW_S:
            crash.consecutive_fails += 1
        else:
            crash.consecutive_fails = 1
        idx = min(crash.consecutive_fails - 1, len(BACKOFF_SCHEDULE_S) - 1)
        backoff = BACKOFF_SCHEDULE_S[idx]
        crash.next_retry_at = time.monotonic() + backoff
        logger.warning(
            "camera_supervisor_process_crashed", camera_id=camera_id,
            exit_code=proc.returncode, uptime_s=round(uptime, 1),
            consecutive_fails=crash.consecutive_fails, retry_in_s=backoff,
        )
        await self._set_status(
            camera_id, state="crashed", pid=None, last_exit_code=proc.returncode,
            last_error=f"exited after {uptime:.1f}s (code {proc.returncode}); retrying in {backoff}s",
            restart_count=crash.consecutive_fails,
        )

    async def _stop_process(self, camera_id: str, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        self._intentional_stop.add(camera_id)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=GRACEFUL_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("camera_supervisor_force_kill", camera_id=camera_id)
            proc.kill()
            await proc.wait()
        # _watch() (already running as its own task) observes this exit and
        # writes the final status - nothing further to do here.

    async def _set_status(self, camera_id: str, **fields: Any) -> None:
        """Upsert camera_process_status. Caller passes exactly the columns
        it wants written (including started_at/last_heartbeat_at as real
        datetimes where relevant) - no implicit defaulting here, so it's
        obvious from each call site what actually changes."""
        if not fields:
            return
        cols = list(fields.keys())
        insert_cols = ["camera_id"] + cols
        insert_vals = [":camera_id"] + [f":{c}" for c in cols]
        update_clause = ", ".join(f"{c} = :{c}" for c in cols)
        params = {"camera_id": camera_id, **fields}
        async with get_db_session() as session:
            row = (await session.execute(text(f"""
                INSERT INTO camera_process_status ({", ".join(insert_cols)})
                VALUES ({", ".join(insert_vals)})
                ON CONFLICT (camera_id) DO UPDATE SET {update_clause}
                RETURNING state, pid, last_error, restart_count
            """), params)).fetchone()
            await session.commit()

        # Push to anyone with the camera's status-ws open (cameras.py) so the
        # Settings UI updates live instead of only on its initial fetch.
        redis = await get_redis()
        await redis.publish(f"camera:{camera_id}:process_status", json.dumps({
            "state": row.state, "pid": row.pid,
            "last_error": row.last_error, "restart_count": row.restart_count,
        }))
