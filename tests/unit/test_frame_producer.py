from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.ingestion.frame_producer import FrameProducer
from src.pipeline.face_pipeline import FrameMeta


def make_camera(frames: list[np.ndarray] | None = None) -> MagicMock:
    """Camera mock that yields frames in order, then signals failure."""
    camera = MagicMock()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    if frames is None:
        frames = [blank]
    # Build read_frame side_effect: (True, frame) for each frame, then (False, None)
    side_effects = [(True, f) for f in frames] + [(False, None)]
    camera.read_frame.side_effect = side_effects
    return camera


# ── run ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_callback_called_once_per_frame():
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
    camera = make_camera(frames)
    callback = AsyncMock()

    producer = FrameProducer(camera=camera, camera_id="cam0", max_fps=0)
    await producer.run(callback)

    assert callback.call_count == 3


@pytest.mark.asyncio
async def test_stops_when_camera_fails():
    camera = make_camera(frames=[])   # immediately fails
    callback = AsyncMock()

    producer = FrameProducer(camera=camera, camera_id="cam0", max_fps=0)
    await producer.run(callback)

    assert callback.call_count == 0
    assert producer._running is False


@pytest.mark.asyncio
async def test_frame_meta_has_correct_camera_id():
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    camera = make_camera(frames)
    received_metas: list[FrameMeta] = []

    async def capture(frame, meta):
        received_metas.append(meta)

    producer = FrameProducer(camera=camera, camera_id="entrance_cam", max_fps=0)
    await producer.run(capture)

    assert received_metas[0].camera_id == "entrance_cam"


@pytest.mark.asyncio
async def test_frame_meta_has_correct_zone_id():
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    camera = make_camera(frames)
    received_metas: list[FrameMeta] = []

    async def capture(frame, meta):
        received_metas.append(meta)

    producer = FrameProducer(camera=camera, camera_id="c0", zone_id="room_a", max_fps=0)
    await producer.run(capture)

    assert received_metas[0].zone_id == "room_a"


@pytest.mark.asyncio
async def test_frame_id_increments_across_frames():
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
    camera = make_camera(frames)
    ids: list[str] = []

    async def capture(frame, meta):
        ids.append(meta.frame_id)

    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0)
    await producer.run(capture)

    assert ids == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_stop_halts_producer_mid_run():
    """stop() called inside callback terminates the loop."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Provide more frames than we'll consume
    camera = MagicMock()
    camera.read_frame.return_value = (True, frame)
    call_count = 0

    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0)

    async def stop_after_one(f, meta):
        nonlocal call_count
        call_count += 1
        producer.stop()

    await producer.run(stop_after_one)

    assert call_count == 1
    assert producer._running is False


@pytest.mark.asyncio
async def test_frame_count_tracked():
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 5
    camera = make_camera(frames)
    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0)
    await producer.run(AsyncMock())
    assert producer._frame_count == 5


# ── drop_stale mode ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drop_stale_processes_newest_and_counts_drops():
    """Grabber drains the camera; the final processed frame is the newest one."""
    n = 50
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]
    camera = make_camera(frames)
    received: list[np.ndarray] = []

    async def capture(frame, meta):
        received.append(frame)

    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0, drop_stale=True)
    await producer.run(capture)

    assert len(received) >= 1
    # last frame delivered to the callback is the last one the camera produced
    assert received[-1][0, 0, 0] == n - 1
    # every frame is either processed or counted as dropped
    assert producer._frame_count + producer._frames_dropped == n


@pytest.mark.asyncio
async def test_drop_stale_stops_when_camera_fails():
    camera = make_camera(frames=[])   # immediately fails
    callback = AsyncMock()

    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0, drop_stale=True)
    await producer.run(callback)

    assert callback.call_count == 0
    assert producer._running is False


@pytest.mark.asyncio
async def test_drop_stale_stop_halts_producer():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    camera = MagicMock()
    camera.read_frame.return_value = (True, frame)

    producer = FrameProducer(camera=camera, camera_id="c0", max_fps=0, drop_stale=True)

    async def stop_after_one(f, meta):
        producer.stop()

    await producer.run(stop_after_one)
    assert producer._running is False
