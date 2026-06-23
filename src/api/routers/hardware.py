from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.auth import ActiveUser
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GPU_POLL_INTERVAL_S = 2


def _nvml_gpus() -> list[dict[str, Any]]:
    """Real GPU stats via pynvml (nvidia-ml-py). Returns [] rather than
    raising on any failure (no NVIDIA driver - e.g. a Mac dev box, or the
    driver/library just not being there) so callers can show "CPU mode"
    instead of a 500."""
    try:
        import pynvml
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
                power_w = round(power_mw / 1000, 1)
            except Exception:
                power_w = None
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append({
                "gpu_id": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "utilization_pct": util.gpu,
                "memory_used_mb": mem.used // (1024 * 1024),
                "memory_total_mb": mem.total // (1024 * 1024),
                "temperature_c": temp,
                "power_w": power_w,
            })
        return gpus
    finally:
        pynvml.nvmlShutdown()


@router.get("/gpus")
async def list_gpus(current_user: ActiveUser) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_nvml_gpus)


@router.websocket("/gpu-stats-ws")
async def gpu_stats_ws(websocket: WebSocket, token: str = Query(...)) -> None:
    """Live GPU stats for the Settings Hardware page - same ?token=
    workaround as alerts_ws/cameras.py's status-ws (a WebSocket can't send
    an Authorization header). Polls pynvml every GPU_POLL_INTERVAL_S rather
    than pushing on a Redis channel, since there's no event to push on -
    GPU load changes continuously, not in discrete steps like camera state."""
    import jwt as _jwt

    from src.core.config import get_settings

    settings = get_settings()
    try:
        payload = _jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise _jwt.InvalidTokenError("not an access token")
    except _jwt.InvalidTokenError:
        await websocket.close(code=4401, reason="invalid or missing token")
        return

    await websocket.accept()
    try:
        while True:
            gpus = await asyncio.to_thread(_nvml_gpus)
            await websocket.send_json({"gpus": gpus})
            await asyncio.sleep(GPU_POLL_INTERVAL_S)
    except WebSocketDisconnect:
        pass


@router.get("/system")
async def system_stats(current_user: ActiveUser) -> dict[str, Any]:
    import psutil

    def _disk(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "used_gb": round(usage.used / (1024 ** 3), 1),
            "total_gb": round(usage.total / (1024 ** 3), 1),
        }

    cpu_pct, mem = await asyncio.to_thread(
        lambda: (psutil.cpu_percent(interval=0.2), psutil.virtual_memory())
    )
    return {
        "cpu_pct": cpu_pct,
        "memory_used_gb": round(mem.used / (1024 ** 3), 1),
        "memory_total_gb": round(mem.total / (1024 ** 3), 1),
        "disks": [d for d in (_disk(ROOT / "data"), _disk(ROOT / "models")) if d is not None],
    }
