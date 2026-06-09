from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.exceptions import EnrollmentError
from src.face_recognition.enrollment import EnrollmentBuffer


def make_recognizer(embedding: np.ndarray | None = None) -> MagicMock:
    rec = MagicMock()
    if embedding is not None:
        rec.get_embedding.return_value = embedding
    else:
        rec.get_embedding.return_value = np.ones(512) / np.sqrt(512)
    return rec


def blank_crop() -> np.ndarray:
    return np.zeros((112, 112, 3), dtype=np.uint8)


# ── start + add_crop + is_ready ───────────────────────────────────────────────

def test_start_assigns_unknown_person_id():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=5)
    pid = buf.start("track1")
    assert pid.startswith("UNKNOWN_")


def test_start_is_idempotent():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=5)
    pid1 = buf.start("track1")
    pid2 = buf.start("track1")
    assert pid1 == pid2


def test_not_ready_below_min_crops():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=5)
    buf.start("track1")
    for _ in range(3):
        buf.add_crop("track1", blank_crop(), quality_score=0.8)
    assert buf.is_ready("track1") is False


def test_ready_at_min_crops():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=5)
    buf.start("track1")
    for _ in range(5):
        buf.add_crop("track1", blank_crop(), quality_score=0.8)
    assert buf.is_ready("track1") is True


def test_finalize_raises_when_insufficient_crops():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=10)
    buf.start("track1")
    for _ in range(3):
        buf.add_crop("track1", blank_crop(), quality_score=0.7)
    with pytest.raises(EnrollmentError, match="Insufficient"):
        buf.finalize("track1")


def test_finalize_clears_buffer():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=3)
    buf.start("track1")
    for _ in range(3):
        buf.add_crop("track1", blank_crop(), quality_score=0.8)
    buf.finalize("track1")
    assert not buf.is_ready("track1")   # buffer cleared


def test_quality_weighted_embedding_high_quality_dominates():
    """
    Two crops: high quality → embedding [1, 0, ...], low quality → [0, 1, ...].
    The weighted mean should have first component larger than second.
    """
    high_q_emb = np.zeros(512)
    high_q_emb[0] = 1.0

    low_q_emb = np.zeros(512)
    low_q_emb[1] = 1.0

    rec = MagicMock()
    rec.get_embedding.side_effect = [high_q_emb.copy(), low_q_emb.copy()]

    buf = EnrollmentBuffer(recognizer=rec, min_crops=2)
    buf.start("track1")
    buf.add_crop("track1", blank_crop(), quality_score=0.9)   # high quality
    buf.add_crop("track1", blank_crop(), quality_score=0.1)   # low quality
    candidate = buf.finalize("track1")
    assert candidate.embedding[0] > candidate.embedding[1]


def test_abandon_removes_state():
    buf = EnrollmentBuffer(recognizer=make_recognizer(), min_crops=5)
    buf.start("track1")
    buf.add_crop("track1", blank_crop(), quality_score=0.8)
    buf.abandon("track1")
    assert not buf.is_ready("track1")
    # add_crop after abandon should be silently ignored
    buf.add_crop("track1", blank_crop(), quality_score=0.8)
    assert not buf.is_ready("track1")
