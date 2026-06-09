from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.exceptions import GalleryError
from src.face_recognition.gallery import FaceGallery
from src.face_recognition.recognizer import ACCEPT_THRESHOLD, REJECT_THRESHOLD


def unit_vec(dim: int = 512, index: int = 0) -> np.ndarray:
    v = np.zeros(dim)
    v[index] = 1.0
    return v


def make_row(embedding_id: str, person_id: str, similarity: float) -> MagicMock:
    row = MagicMock()
    row.embedding_id = embedding_id
    row.person_id = person_id
    row.similarity = similarity
    return row


def mock_session(rows: list | None = None, rowcount: int = 0) -> MagicMock:
    """Build a mock async context-manager session that returns rows on execute."""
    result = MagicMock()
    result.fetchall.return_value = rows or []
    result.rowcount = rowcount

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ── search ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_sorted_matches():
    rows = [
        make_row("e1", "p1", 0.55),
        make_row("e2", "p2", 0.30),
        make_row("e3", "p3", 0.40),
    ]
    ctx = mock_session(rows)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        matches = await gallery.search(unit_vec())
    # sorted descending by similarity
    sims = [m.similarity for m in matches]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.asyncio
async def test_search_assigns_correct_decisions():
    rows = [
        make_row("e1", "p1", ACCEPT_THRESHOLD + 0.05),   # accept
        make_row("e2", "p2", (ACCEPT_THRESHOLD + REJECT_THRESHOLD) / 2),  # ambiguous
        make_row("e3", "p3", REJECT_THRESHOLD - 0.05),   # reject
    ]
    ctx = mock_session(rows)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        matches = await gallery.search(unit_vec())
    decisions = {m.person_id: m.decision for m in matches}
    assert decisions["p1"] == "accept"
    assert decisions["p2"] == "ambiguous"
    assert decisions["p3"] == "reject"


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_no_rows():
    ctx = mock_session(rows=[])
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        matches = await gallery.search(unit_vec())
    assert matches == []


@pytest.mark.asyncio
async def test_search_raises_gallery_error_on_db_failure():
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        with pytest.raises(GalleryError):
            await gallery.search(unit_vec())


# ── add_embedding ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_embedding_returns_uuid_string():
    ctx = mock_session()
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        eid = await gallery.add_embedding("person-1", unit_vec(), quality_score=0.85)
    assert isinstance(eid, str) and len(eid) == 36  # UUID format


@pytest.mark.asyncio
async def test_add_embedding_raises_gallery_error_on_db_failure():
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("disk full"))
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        with pytest.raises(GalleryError):
            await gallery.add_embedding("person-1", unit_vec(), quality_score=0.85)


# ── delete_person ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_person_returns_rowcount():
    ctx = mock_session(rowcount=3)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        count = await gallery.delete_person("person-1")
    assert count == 3


@pytest.mark.asyncio
async def test_delete_person_raises_gallery_error_on_db_failure():
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("timeout"))
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("src.face_recognition.gallery.get_db_session", return_value=ctx):
        gallery = FaceGallery()
        with pytest.raises(GalleryError):
            await gallery.delete_person("person-1")
