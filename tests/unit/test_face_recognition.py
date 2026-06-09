import numpy as np
import pytest

from src.core.exceptions import ModelLoadError
from src.face_recognition.recognizer import (
    ACCEPT_THRESHOLD,
    EMBEDDING_DIM,
    REJECT_THRESHOLD,
    AdaFaceRecognizer,
    RecognitionMatch,
    cosine_similarity,
    l2_normalize,
)


# ── l2_normalize ──────────────────────────────────────────────────────────────

def test_l2_normalize_unit_length():
    v = np.array([3.0, 4.0])
    result = l2_normalize(v)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_l2_normalize_zero_vector_unchanged():
    v = np.zeros(4)
    result = l2_normalize(v)
    assert np.array_equal(result, v)


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_similarity_identical_vectors_returns_one():
    v = np.random.randn(EMBEDDING_DIM)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_returns_zero():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


# ── RecognitionMatch.decide ───────────────────────────────────────────────────

def test_match_decide_accept():
    m = RecognitionMatch.decide("eid1", "pid1", ACCEPT_THRESHOLD + 0.05)
    assert m.decision == "accept"


def test_match_decide_reject():
    m = RecognitionMatch.decide("eid2", "pid2", REJECT_THRESHOLD - 0.05)
    assert m.decision == "reject"


def test_match_decide_ambiguous():
    mid = (ACCEPT_THRESHOLD + REJECT_THRESHOLD) / 2
    m = RecognitionMatch.decide("eid3", "pid3", mid)
    assert m.decision == "ambiguous"


# ── AdaFaceRecognizer ─────────────────────────────────────────────────────────

def test_recognizer_raises_when_no_model():
    rec = AdaFaceRecognizer.__new__(AdaFaceRecognizer)
    rec._model = None
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    with pytest.raises(ModelLoadError):
        rec.get_embedding(crop)
