from __future__ import annotations

import asyncio
import threading
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

    drop_stale=False reads frames sequentially: every frame the camera
    delivers is processed, in order. Right for files and tests, but on a
    live source a slow callback makes the camera's internal buffer grow
    without bound and the feed lags further behind real time every second.

    drop_stale=True runs a grabber thread that drains the camera at full
    speed and keeps only the newest frame; the callback always receives
    the most recent frame and intermediate ones are dropped. Use this for
    any live camera or RTSP stream.
    """

    def __init__(
        self,
        camera: CameraReader,
        camera_id: str,
        zone_id: str = "",
        max_fps: int = 15,
        drop_stale: bool = False,
    ) -> None:
        self.camera = camera
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.drop_stale = drop_stale
        self._frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._running = False
        self._frame_count = 0
        self._frames_dropped = 0
        self._latest_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = 0
        self._grabber_failed = False

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

        if self.drop_stale:
            await self._run_latest(callback)
        else:
            await self._run_sequential(callback)

        logger.info(
            "frame_producer_stopped",
            camera_id=self.camera_id,
            total_frames=self._frame_count,
            frames_dropped=self._frames_dropped,
        )

    async def _run_sequential(
        self,
        callback: Callable[[np.ndarray, FrameMeta], Awaitable[None]],
    ) -> None:
        while self._running:
            t0 = time.monotonic()

            ok, frame = self.camera.read_frame()
            if not ok:
                logger.warning(
                    "frame_producer_camera_lost_reconnecting",
                    camera_id=self.camera_id,
                    frames_produced=self._frame_count,
                )
                # blocking backoff — sequential mode has no separate grabber
                # thread, so there's nothing else to keep responsive here
                if not await asyncio.to_thread(self.camera.reopen_with_backoff):
                    logger.warning(
                        "frame_producer_camera_failed",
                        camera_id=self.camera_id,
                        frames_produced=self._frame_count,
                    )
                    self._running = False
                    break
                continue

            await self._emit(frame, callback)
            await self._throttle(t0)

    async def _run_latest(
        self,
        callback: Callable[[np.ndarray, FrameMeta], Awaitable[None]],
    ) -> None:
        grabber = threading.Thread(
            target=self._grab_loop,
            name=f"frame_grabber_{self.camera_id}",
            daemon=True,
        )
        grabber.start()
        last_seq = 0

        while self._running:
            t0 = time.monotonic()

            with self._latest_lock:
                frame = self._latest_frame
                seq = self._latest_seq

            if seq == last_seq:
                if self._grabber_failed:
                    logger.warning(
                        "frame_producer_camera_failed",
                        camera_id=self.camera_id,
                        frames_produced=self._frame_count,
                    )
                    self._running = False
                    break
                await asyncio.sleep(0.005)
                continue

            self._frames_dropped += seq - last_seq - 1
            last_seq = seq
            await self._emit(frame, callback)
            await self._throttle(t0)

        # let the grabber finish its in-flight read before the caller
        # releases the camera — closing the container under a live read
        # stalls shutdown (and can crash PyAV)
        self._running = False
        grabber.join(timeout=2.0)

    def _grab_loop(self) -> None:
        while self._running:
            ok, frame = self.camera.read_frame()
            if not ok:
                logger.warning(
                    "frame_producer_camera_lost_reconnecting",
                    camera_id=self.camera_id,
                    frames_produced=self._frame_count,
                )
                if not self.camera.reopen_with_backoff():
                    self._grabber_failed = True
                    return
                continue
            with self._latest_lock:
                self._latest_frame = frame
                self._latest_seq += 1

    async def _emit(
        self,
        frame: np.ndarray,
        callback: Callable[[np.ndarray, FrameMeta], Awaitable[None]],
    ) -> None:
        meta = FrameMeta(
            camera_id=self.camera_id,
            frame_id=str(self._frame_count),
            timestamp_ns=time.time_ns(),
            zone_id=self.zone_id,
        )
        await callback(frame, meta)
        self._frame_count += 1

    async def _throttle(self, t0: float) -> None:
        if self._frame_interval > 0:
            elapsed = time.monotonic() - t0
            sleep_for = self._frame_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def stop(self) -> None:
        """Signal the producer to stop after the current frame."""
        self._running = False
