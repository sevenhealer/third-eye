from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import numpy as np

from src.core.logging import get_logger
from src.ingestion.camera import CameraReader
from src.pipeline.face_pipeline import FrameMeta

logger = get_logger(__name__)


class FrameProducer:
    """
    Async camera frame producer.

    Reads frames from a CameraReader at up to max_fps and invokes an
    async callback for each frame. Stops automatically when the camera
    fails or stop() is called.

    max_fps=0 means no throttle — reads as fast as the camera delivers.
    """

    def __init__(
        self,
        camera: CameraReader,
        camera_id: str,
        zone_id: str = "",
        max_fps: int = 15,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self.zone_id = zone_id
        self._frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._running = False
        self._frame_count = 0

    async def run(
        self,
        callback: Callable[[np.ndarray, FrameMeta], Awaitable[None]],
    ) -> None:
        """
        Continuously read frames and call callback(frame, meta) for each.

        Exits when the camera produces a read failure or stop() is called.
        """
        self._running = True
        logger.info("frame_producer_started", camera_id=self.camera_id, zone_id=self.zone_id)

        while self._running:
            t0 = time.monotonic()

            ok, frame = self.camera.read_frame()
            if not ok:
                logger.warning(
                    "frame_producer_camera_failed",
                    camera_id=self.camera_id,
                    frames_produced=self._frame_count,
                )
                self._running = False
                break

            meta = FrameMeta(
                camera_id=self.camera_id,
                frame_id=str(self._frame_count),
                timestamp_ns=time.time_ns(),
                zone_id=self.zone_id,
            )
            await callback(frame, meta)
            self._frame_count += 1

            if self._frame_interval > 0:
                elapsed = time.monotonic() - t0
                sleep_for = self._frame_interval - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        logger.info(
            "frame_producer_stopped",
            camera_id=self.camera_id,
            total_frames=self._frame_count,
        )

    def stop(self) -> None:
        """Signal the producer to stop after the current frame."""
        self._running = False
