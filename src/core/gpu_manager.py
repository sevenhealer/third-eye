from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from src.core.exceptions import VRAMBudgetError
from src.core.logging import get_logger

logger = get_logger(__name__)

# VRAM budget allocations in MB (must sum to < 23,000 to leave OS headroom)
VRAM_ALLOCATIONS: dict[str, int] = {
    "scrfd_10gf":       500,
    "scrfd_25gf":       250,
    "minifasnet_v2":    150,
    "cdcn_plus":        400,
    "adaface_r100":     800,
    "yolov9c":          900,
    "osnet_x10":        200,
    "x3d_m":            400,
    "clip_vitl14":      900,
    "videomae_b":       1200,   # on-demand
    "llava_7b_4bit":    4000,   # time-multiplexed slot
    "mistral_7b_4bit":  4000,   # shares llava slot
    "efficientnet_b3":  150,    # on-demand
    "dm_count":         200,    # on-demand
    "frame_buffers":    1500,
    "cuda_workspace":   1500,
}

CONTINUOUS_MODELS = {
    "scrfd_10gf", "scrfd_25gf", "minifasnet_v2", "cdcn_plus",
    "adaface_r100", "yolov9c", "osnet_x10", "x3d_m", "clip_vitl14",
    "frame_buffers", "cuda_workspace",
}

# LLaVA and Mistral share one 4 GB slot — never both loaded simultaneously
SHARED_LLM_SLOT = {"llava_7b_4bit", "mistral_7b_4bit"}


def preload_cuda_libraries() -> None:
    """
    Make the CUDA runtime libraries bundled with the torch wheels
    (nvidia-* pip packages) resolvable for ONNX Runtime's CUDA provider.

    onnxruntime-gpu dlopens libcublasLt/libcudnn at session creation but
    only searches the system loader path; the copies the torch wheels put
    under site-packages/nvidia/*/lib are invisible to it, so the provider
    fails with "libcublasLt.so.12: cannot open shared object file" and
    inference silently falls back to CPU. Loading those libraries into the
    process with RTLD_GLOBAL first lets the provider resolve them.

    No-op on non-Linux platforms or when the nvidia wheels are absent.
    """
    import sys

    if sys.platform != "linux":
        return

    import ctypes
    import sysconfig
    from pathlib import Path

    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia_root.is_dir():
        return

    loaded = 0
    pending = sorted(nvidia_root.glob("*/lib/lib*.so*"))
    # two passes: inter-library deps (e.g. cudnn -> cublas) can need a retry
    for _ in range(2):
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(lib)
        pending = failed
        if not pending:
            break
    logger.info(
        "cuda_libraries_preloaded", loaded=loaded, unresolved=len(pending)
    )


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


@dataclass
class ModelEntry:
    name: str
    vram_mb: int
    state: ModelState = ModelState.UNLOADED
    ref_count: int = 0


class GPUManager:
    """
    Tracks VRAM allocation across all models.

    Rules:
    - Continuous models are loaded at startup and never evicted.
    - On-demand models (videomae_b, efficientnet_b3, dm_count) are
      loaded on request, evicted when ref_count drops to 0.
    - llava_7b_4bit and mistral_7b_4bit share one 4 GB slot.
      Loading one evicts the other.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = {
            name: ModelEntry(name=name, vram_mb=mb)
            for name, mb in VRAM_ALLOCATIONS.items()
        }
        self._active_llm_slot: str | None = None

    @property
    def total_vram_mb(self) -> int:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(self.device).total_memory // (1024 * 1024)

    @property
    def allocated_mb(self) -> int:
        with self._lock:
            return sum(
                e.vram_mb for e in self._entries.values()
                if e.state == ModelState.LOADED
            )

    @property
    def free_mb(self) -> int:
        return self.total_vram_mb - self.allocated_mb

    def _would_exceed_budget(self, model_name: str) -> bool:
        entry = self._entries[model_name]
        # LLM slot: loading one evicts the other, so budget stays same
        if model_name in SHARED_LLM_SLOT and self._active_llm_slot is not None:
            return False
        return (self.allocated_mb + entry.vram_mb) > (self.total_vram_mb - 1024)

    def request_load(self, model_name: str) -> None:
        with self._lock:
            if model_name not in self._entries:
                raise VRAMBudgetError(f"Unknown model: {model_name}")

            entry = self._entries[model_name]

            if entry.state == ModelState.LOADED:
                entry.ref_count += 1
                return

            # Handle LLM slot contention
            if model_name in SHARED_LLM_SLOT:
                other = (SHARED_LLM_SLOT - {model_name}).pop()
                other_entry = self._entries[other]
                if other_entry.state == ModelState.LOADED:
                    logger.info(
                        "evicting_llm_slot",
                        evicted=other,
                        loading=model_name,
                    )
                    other_entry.state = ModelState.UNLOADED
                    other_entry.ref_count = 0
                self._active_llm_slot = model_name

            if self._would_exceed_budget(model_name):
                raise VRAMBudgetError(
                    f"Loading {model_name} ({entry.vram_mb} MB) would exceed "
                    f"VRAM budget. Free: {self.free_mb} MB"
                )

            entry.state = ModelState.LOADED
            entry.ref_count = 1
            logger.info(
                "model_loaded",
                model=model_name,
                vram_mb=entry.vram_mb,
                total_allocated_mb=self.allocated_mb,
            )

    def release(self, model_name: str) -> None:
        with self._lock:
            entry = self._entries.get(model_name)
            if entry is None or entry.state != ModelState.LOADED:
                return
            entry.ref_count = max(0, entry.ref_count - 1)
            if entry.ref_count == 0 and model_name not in CONTINUOUS_MODELS:
                entry.state = ModelState.UNLOADED
                if model_name in SHARED_LLM_SLOT:
                    self._active_llm_slot = None
                logger.info("model_evicted", model=model_name, vram_mb=entry.vram_mb)

    def status(self) -> dict:
        with self._lock:
            return {
                "device": self.device,
                "total_vram_mb": self.total_vram_mb,
                "allocated_mb": self.allocated_mb,
                "free_mb": self.free_mb,
                "models": {
                    name: {
                        "state": e.state.value,
                        "vram_mb": e.vram_mb,
                        "ref_count": e.ref_count,
                    }
                    for name, e in self._entries.items()
                },
            }

    @contextmanager
    def use_model(self, model_name: str) -> Iterator[None]:
        self.request_load(model_name)
        try:
            yield
        finally:
            self.release(model_name)

    @asynccontextmanager
    async def ause_model(self, model_name: str) -> AsyncIterator[None]:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.request_load, model_name)
        try:
            yield
        finally:
            await loop.run_in_executor(None, self.release, model_name)


_gpu_manager: GPUManager | None = None


def get_gpu_manager() -> GPUManager:
    global _gpu_manager
    if _gpu_manager is None:
        from src.core.config import get_settings
        _gpu_manager = GPUManager(device=get_settings().device)
    return _gpu_manager
