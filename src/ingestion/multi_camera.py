from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from src.core.logging import get_logger
from src.ingestion.camera import CameraReader
from src.ingestion.frame_producer import FrameProducer
from src.pipeline.face_pipeline import FrameMeta

logger = get_logger(__name__)

MAX_CAMERAS = 8

FrameCallback = Callable[[np.ndarray, FrameMeta], Awaitable[None]]


@dataclass
class CameraConfig:
    """Configuration for a single camera stream."""

    source: str | int
    camera_id: str
    zone_id: str = ""
    max_fps: int = 15
    reconnect_max_seconds: int = 60
    drop_stale: bool = False  # True for live cameras: process newest frame, drop backlog


class MultiCameraManager:
    """
    Manages up to 8 concurrent camera streams (E01-S03).

    Each camera runs in an independent asyncio Task.  A single camera failure
    does not interrupt other streams — failed cameras are marked unhealthy and
    the rest continue.  All frames from all cameras are routed to a single
    shared callback for downstream processing.

    Usage:
        mgr = MultiCameraManager(configs)
        await mgr.start(callback)
        # ... wait for shutdown signal ...
        await mgr.stop_all()
    """

    def __init__(self, configs: list[CameraConfig]) -> None:
        if len(configs) > MAX_CAMERAS:
            raise ValueError(
                f"MultiCameraManager supports at most {MAX_CAMERAS} cameras; "
                f"got {len(configs)}"
            )
        self._configs = list(configs)
        self._tasks: dict[str, asyncio.Task] = {}
        self._healthy: dict[str, bool] = {}
        self._producers: dict[str, FrameProducer] = {}
        self._cameras: dict[str, CameraReader] = {}

    async def start(self, callback: FrameCallback) -> None:
        """
        Open all cameras and launch per-camera producer tasks.
        Cameras that fail to open are marked unhealthy and skipped.
        """
        for cfg in self._configs:
            cam = CameraReader(
                source=cfg.source,
                reconnect_max_seconds=cfg.reconnect_max_seconds,
            )
            if not cam.open():
                logger.error(
                    "multi_camera_open_failed",
                    camera_id=cfg.camera_id,
                    source=str(cfg.source),
                )
                self._healthy[cfg.camera_id] = False
                continue

            producer = FrameProducer(
                camera=cam,
                camera_id=cfg.camera_id,
                zone_id=cfg.zone_id,
                max_fps=cfg.max_fps,
                drop_stale=cfg.drop_stale,
            )
            self._cameras[cfg.camera_id] = cam
            self._producers[cfg.camera_id] = producer
            self._healthy[cfg.camera_id] = True

            task = asyncio.create_task(
                self._run_camera(cfg.camera_id, producer, callback),
                name=f"camera_{cfg.camera_id}",
            )
            self._tasks[cfg.camera_id] = task
            logger.info("multi_camera_started", camera_id=cfg.camera_id)

    async def _run_camera(
        self,
        camera_id: str,
        producer: FrameProducer,
        callback: FrameCallback,
    ) -> None:
        try:
            await producer.run(callback)
        except Exception as exc:
            logger.error("multi_camera_task_error", camera_id=camera_id, error=str(exc))
        finally:
            self._healthy[camera_id] = False

    async def stop_all(self) -> None:
        """Stop all producers, cancel tasks, and release camera hardware."""
        for producer in self._producers.values():
            producer.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for cam in self._cameras.values():
            cam.release()
        self._tasks.clear()
        self._cameras.clear()
        self._producers.clear()
        logger.info("multi_camera_all_stopped", total=len(self._configs))

    def health(self) -> dict[str, bool]:
        """Return per-camera health status (True = running, False = failed or not started)."""
        return dict(self._healthy)

    @property
    def active_count(self) -> int:
        return sum(1 for h in self._healthy.values() if h)
