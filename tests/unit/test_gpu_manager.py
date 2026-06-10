from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import VRAMBudgetError
from src.core.gpu_manager import GPUManager, ModelState


@pytest.fixture
def manager():
    """GPUManager with mocked torch — simulates 24 GB VRAM card."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.total_memory = 24 * 1024 * 1024 * 1024  # 24 GB
    mock_torch.cuda.get_device_properties.return_value = props

    with patch("src.core.gpu_manager.torch", mock_torch), \
         patch("src.core.gpu_manager._TORCH_AVAILABLE", True):
        yield GPUManager(device="cuda:0")


def test_load_continuous_model(manager):
    manager.request_load("scrfd_10gf")
    assert manager._entries["scrfd_10gf"].state == ModelState.LOADED
    assert manager._entries["scrfd_10gf"].ref_count == 1


def test_load_increments_refcount(manager):
    manager.request_load("scrfd_10gf")
    manager.request_load("scrfd_10gf")
    assert manager._entries["scrfd_10gf"].ref_count == 2


def test_release_continuous_stays_loaded(manager):
    manager.request_load("scrfd_10gf")
    manager.release("scrfd_10gf")
    # Continuous models are never evicted
    assert manager._entries["scrfd_10gf"].state == ModelState.LOADED


def test_ondemand_model_evicted_on_release(manager):
    manager.request_load("videomae_b")
    assert manager._entries["videomae_b"].state == ModelState.LOADED
    manager.release("videomae_b")
    assert manager._entries["videomae_b"].state == ModelState.UNLOADED


def test_llm_slot_evicts_other(manager):
    manager.request_load("llava_7b_4bit")
    assert manager._entries["llava_7b_4bit"].state == ModelState.LOADED

    manager.request_load("mistral_7b_4bit")
    assert manager._entries["mistral_7b_4bit"].state == ModelState.LOADED
    # LLaVA must have been evicted to free the shared slot
    assert manager._entries["llava_7b_4bit"].state == ModelState.UNLOADED


def test_llm_slot_reverse_eviction(manager):
    manager.request_load("mistral_7b_4bit")
    assert manager._entries["mistral_7b_4bit"].state == ModelState.LOADED

    manager.request_load("llava_7b_4bit")
    assert manager._entries["llava_7b_4bit"].state == ModelState.LOADED
    assert manager._entries["mistral_7b_4bit"].state == ModelState.UNLOADED


def test_unknown_model_raises(manager):
    with pytest.raises(VRAMBudgetError):
        manager.request_load("nonexistent_model")


def test_status_returns_dict(manager):
    manager.request_load("scrfd_10gf")
    status = manager.status()
    assert "allocated_mb" in status
    assert "models" in status
    assert status["models"]["scrfd_10gf"]["state"] == "loaded"


def test_context_manager_loads_and_releases(manager):
    with manager.use_model("videomae_b"):
        assert manager._entries["videomae_b"].state == ModelState.LOADED
    # on-demand → should be unloaded after context exit
    assert manager._entries["videomae_b"].state == ModelState.UNLOADED


def test_preload_cuda_libraries_is_safe_everywhere():
    """No-op on macOS / missing nvidia wheels; must never raise."""
    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()
