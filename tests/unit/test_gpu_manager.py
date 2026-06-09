from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import VRAMBudgetError
from src.core.gpu_manager import GPUManager, ModelState


@pytest.fixture
def manager():
    with patch("src.core.gpu_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 24 * 1024 * 1024 * 1024  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = props
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
    # continuous models never evicted
    assert manager._entries["scrfd_10gf"].state == ModelState.LOADED


def test_llm_slot_evicts_other(manager):
    manager.request_load("llava_7b_4bit")
    assert manager._entries["llava_7b_4bit"].state == ModelState.LOADED

    manager.request_load("mistral_7b_4bit")
    assert manager._entries["mistral_7b_4bit"].state == ModelState.LOADED
    assert manager._entries["llava_7b_4bit"].state == ModelState.UNLOADED


def test_unknown_model_raises(manager):
    with pytest.raises(VRAMBudgetError):
        manager.request_load("nonexistent_model")
