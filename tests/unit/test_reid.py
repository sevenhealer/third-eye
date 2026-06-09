import numpy as np
import pytest

from src.tracking.reid import OSNetReID


def _reid() -> OSNetReID:
    return OSNetReID(device="cpu")


def test_extract_returns_zeros_when_no_model():
    reid = _reid()
    # torchreid absent in CI — model is None
    if reid._model is None:
        crop = np.zeros((64, 32, 3), dtype=np.uint8)
        result = reid.extract(crop)
        assert result.shape == (OSNetReID.EMBED_DIM,)
        assert np.all(result == 0)


def test_extract_returns_correct_shape():
    reid = _reid()
    crop = np.zeros((64, 32, 3), dtype=np.uint8)
    result = reid.extract(crop)
    assert result.shape == (OSNetReID.EMBED_DIM,)
    assert result.dtype == np.float32


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert OSNetReID.cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert OSNetReID.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert OSNetReID.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    z = np.zeros(3, dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert OSNetReID.cosine_similarity(z, v) == pytest.approx(0.0)
    assert OSNetReID.cosine_similarity(z, z) == pytest.approx(0.0)
