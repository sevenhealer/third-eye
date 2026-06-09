import collections
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.antispoofing.ensemble import (
    MIN_CONFIDENCE,
    AntiSpoofingEnsemble,
    CDCNWrapper,
    MiniFASWrapper,
    SpoofingResult,
    TemporalConsistencyChecker,
)
from src.core.exceptions import LowConfidenceError, SpoofingDetectedError


# ── helpers ──────────────────────────────────────────────────────────────────

def make_minifas(score: float) -> MiniFASWrapper:
    m = MagicMock(spec=MiniFASWrapper)
    m.score.return_value = score
    m._model = object()  # truthy → "loaded"
    return m


def make_cdcn(score: float, loaded: bool = False) -> CDCNWrapper:
    m = MagicMock(spec=CDCNWrapper)
    m.score.return_value = score
    m._model = object() if loaded else None
    return m


def make_temporal(score: float, frames: int = 0) -> TemporalConsistencyChecker:
    t = MagicMock(spec=TemporalConsistencyChecker)
    t.score.return_value = score
    t.MINIMUM_FRAMES = 3
    buf = collections.deque(range(frames))
    t._buffers = {"track1": buf} if frames > 0 else {}
    t.update = MagicMock()
    return t


def crop() -> np.ndarray:
    return np.zeros((112, 112, 3), dtype=np.uint8)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_all_pass_returns_live():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.93),
        cdcn=make_cdcn(0.0, loaded=False),   # CDCN not loaded → skipped
        temporal=make_temporal(0.8, frames=5),
    )
    result = ensemble.check(crop(), "track1")
    assert result.is_live is True
    assert result.rejected_by is None


def test_minifas_low_score_raises_spoof_error():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.0),   # not loaded → fail-safe deny
        cdcn=make_cdcn(0.0),
        temporal=make_temporal(0.9, frames=5),
    )
    with pytest.raises(SpoofingDetectedError):
        ensemble.check(crop(), "track1")


def test_minifas_non_zero_below_threshold_raises_low_confidence():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.4),   # loaded but low confidence
        cdcn=make_cdcn(0.0),
        temporal=make_temporal(0.9, frames=5),
        confidence_threshold=MIN_CONFIDENCE,
    )
    with pytest.raises(LowConfidenceError):
        ensemble.check(crop(), "track1")


def test_cdcn_loaded_and_low_raises_spoof_error():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.92),
        cdcn=make_cdcn(0.0, loaded=True),   # loaded but score=0 → deny
        temporal=make_temporal(0.9, frames=5),
    )
    with pytest.raises(SpoofingDetectedError):
        ensemble.check(crop(), "track1")


def test_temporal_still_image_with_frames_raises_spoof_error():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.92),
        cdcn=make_cdcn(0.0),
        temporal=make_temporal(score=0.01, frames=5),   # near-zero variance
    )
    with pytest.raises(SpoofingDetectedError, match="Temporal"):
        ensemble.check(crop(), "track1")


def test_temporal_insufficient_frames_neutral_pass():
    """Fewer than MINIMUM_FRAMES accumulated → temporal check skipped."""
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.92),
        cdcn=make_cdcn(0.0),
        temporal=make_temporal(score=0.0, frames=1),   # only 1 frame
    )
    result = ensemble.check(crop(), "track1")
    assert result.is_live is True


def test_ensemble_confidence_is_mean_of_active_models():
    """With only MiniFASNet active, confidence == MiniFAS score."""
    score = 0.91
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(score),
        cdcn=make_cdcn(0.0, loaded=False),
        temporal=make_temporal(0.8, frames=5),
    )
    result = ensemble.check(crop(), "track1")
    assert result.confidence == pytest.approx(score)


def test_both_cdcn_and_minifas_active_confidence_is_average():
    ensemble = AntiSpoofingEnsemble(
        minifas=make_minifas(0.90),
        cdcn=make_cdcn(0.88, loaded=True),
        temporal=make_temporal(0.8, frames=5),
    )
    result = ensemble.check(crop(), "track1")
    assert result.confidence == pytest.approx((0.90 + 0.88) / 2)
