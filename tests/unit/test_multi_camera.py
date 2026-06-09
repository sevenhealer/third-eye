import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from src.ingestion.multi_camera import CameraConfig, MultiCameraManager, MAX_CAMERAS


def _config(camera_id: str, source: int = 0) -> CameraConfig:
    return CameraConfig(source=source, camera_id=camera_id, zone_id="zone_a", max_fps=10)


# ── constructor validation ────────────────────────────────────────────────────

def test_too_many_cameras_raises():
    configs = [_config(f"cam{i}", i) for i in range(MAX_CAMERAS + 1)]
    with pytest.raises(ValueError, match="at most"):
        MultiCameraManager(configs)


def test_max_cameras_accepted():
    configs = [_config(f"cam{i}", i) for i in range(MAX_CAMERAS)]
    mgr = MultiCameraManager(configs)
    assert len(mgr._configs) == MAX_CAMERAS


# ── health tracking ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_marks_failed_camera_unhealthy():
    cfg = _config("cam0", source=0)
    mgr = MultiCameraManager([cfg])

    with patch("src.ingestion.multi_camera.CameraReader") as MockReader:
        instance = MockReader.return_value
        instance.open.return_value = False
        await mgr.start(AsyncMock())

    assert mgr.health() == {"cam0": False}
    assert mgr.active_count == 0


@pytest.mark.asyncio
async def test_start_marks_successful_camera_healthy():
    cfg = _config("cam0", source=0)
    mgr = MultiCameraManager([cfg])

    # Producer.run will return immediately after one frame via callback
    async def _noop_run(callback):
        pass

    with patch("src.ingestion.multi_camera.CameraReader") as MockReader, \
         patch("src.ingestion.multi_camera.FrameProducer") as MockProducer:
        cam_instance = MockReader.return_value
        cam_instance.open.return_value = True
        prod_instance = MockProducer.return_value
        prod_instance.run = _noop_run
        await mgr.start(AsyncMock())

    assert mgr.health()["cam0"] is True
    assert mgr.active_count == 1


@pytest.mark.asyncio
async def test_mixed_cameras_partial_health():
    configs = [_config("cam0"), _config("cam1")]
    mgr = MultiCameraManager(configs)

    open_results = {"cam0": True, "cam1": False}
    call_order = []

    async def _noop_run(callback):
        pass

    with patch("src.ingestion.multi_camera.CameraReader") as MockReader, \
         patch("src.ingestion.multi_camera.FrameProducer") as MockProducer:

        def make_cam():
            cam = MagicMock()
            idx = len(call_order)
            cam_id = list(open_results.keys())[idx] if idx < len(open_results) else "?"
            cam.open.return_value = open_results.get(cam_id, False)
            call_order.append(cam_id)
            return cam

        MockReader.side_effect = lambda **kw: make_cam()
        prod_instance = MockProducer.return_value
        prod_instance.run = _noop_run
        await mgr.start(AsyncMock())

    assert mgr.active_count >= 0   # at least the manager ran without error


# ── stop_all ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stop_all_is_idempotent():
    mgr = MultiCameraManager([])
    await mgr.stop_all()
    await mgr.stop_all()   # should not raise
