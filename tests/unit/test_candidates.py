import numpy as np

from src.face_recognition.candidates import CandidateCapture
from src.storage.object_store import ObjectStore


def _disabled_store() -> ObjectStore:
    return ObjectStore(endpoint_url="", access_key="", secret_key="", bucket="b")


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=64).astype("float32")


def _capture(**kw) -> CandidateCapture:
    defaults = {"camera_id": "cam0", "min_crops": 3, "min_span_sec": 0.0,
                "object_store": _disabled_store()}
    defaults.update(kw)
    return CandidateCapture(**defaults)


def test_recognized_track_is_not_a_candidate():
    cap = _capture()
    cap.observe(1, _emb(1), is_unknown=False, good_view=True, quality=0.9)
    assert cap.due() == []


def test_unknown_accumulates_then_becomes_due():
    cap = _capture()
    for i in range(3):
        cap.observe(1, _emb(i), is_unknown=True, good_view=True, quality=0.8,
                    crop=f"jpeg{i}".encode())
    drafts = cap.due()
    assert len(drafts) == 1
    assert drafts[0].track_id == 1
    assert drafts[0].mean_embedding.shape == (64,)
    # crops captured
    assert len(drafts[0].crops) == 3
    # due is one-shot per track
    assert cap.due() == []


def test_best_crops_selected_by_quality():
    cap = _capture(min_crops=2, max_stored_crops=2)
    cap.observe(1, _emb(0), is_unknown=True, good_view=True, quality=0.5, crop=b"low")
    cap.observe(1, _emb(1), is_unknown=True, good_view=True, quality=0.9, crop=b"high")
    cap.observe(1, _emb(2), is_unknown=True, good_view=True, quality=0.7, crop=b"mid")
    draft = cap.due()[0]
    assert b"high" in draft.crops          # highest quality kept
    assert b"low" not in draft.crops       # lowest dropped (max 2)


def test_bad_view_not_captured():
    cap = _capture()
    cap.observe(1, _emb(1), is_unknown=True, good_view=False, quality=0.9, crop=b"x")
    assert cap.due() == []


def test_upload_crops_noop_when_store_disabled():
    cap = _capture()
    assert cap._upload_crops([b"a", b"b"]) == []
