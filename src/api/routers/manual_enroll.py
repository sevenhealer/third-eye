from __future__ import annotations

import base64
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.api.routers.identities import PersonOut
from src.core.audit_log import write_audit_event
from src.core.database import get_db
from src.core.logging import get_logger
from src.face_recognition.gallery import get_gallery
from src.face_recognition.manual_enroll import POSE_BUCKETS, get_manual_enroll_capture
from src.face_recognition.pose_detector import get_pose_detector
from src.face_recognition.recognizer import ACCEPT_THRESHOLD, REJECT_THRESHOLD, l2_normalize

logger = get_logger(__name__)
router = APIRouter()

DEFAULT_CROPS = 10
MANUAL_ENROLL_QUALITY = 0.85   # guided, well-framed capture - matches scripts/enroll.py
# Surface a "looks like <X>" suggestion at/above this similarity so the operator
# can choose to merge instead of creating a duplicate. Above ACCEPT_THRESHOLD
# it's almost certainly the same person; between reject..accept it's a maybe —
# either way the operator decides, we don't merge silently.
SUGGEST_THRESHOLD = REJECT_THRESHOLD


class SessionOut(BaseModel):
    session_id: str
    pose_buckets: list[dict[str, str | int]]


class FrameRequest(BaseModel):
    image_b64: str   # bare base64 JPEG, or a data: URL (prefix tolerated)


class FrameResult(BaseModel):
    face_found: bool
    yaw: float | None = None
    pitch: float | None = None
    bucket: str | None = None
    captured: bool = False
    progress: int
    total: int
    pending: list[str]
    done: bool


class FinalizeRequest(BaseModel):
    display_name: str | None = None   # required when creating a NEW person
    role: str = "visitor"
    person_id: UUID | None = None     # set to add this embedding to an existing person


class FinalizeResult(BaseModel):
    person: PersonOut
    merged_into_existing: bool        # added to an already-enrolled identity?
    matched_similarity: float | None  # set when merged into a suggested match
    templates_added: int              # gallery rows added (one per pose)


class SuggestedMatch(BaseModel):
    person_id: UUID
    display_name: str
    similarity: float
    likely_same: bool                 # >= accept threshold (very probably the same person)


class SuggestionResult(BaseModel):
    match: SuggestedMatch | None


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(current_user: ActiveUser, crops: int = DEFAULT_CROPS) -> SessionOut:
    current_user.require_role("admin")
    capture = get_manual_enroll_capture()
    session_id = capture.create(crops_total=crops)
    session = capture.get(session_id)
    assert session is not None
    labels = dict(POSE_BUCKETS)
    return SessionOut(
        session_id=session_id,
        pose_buckets=[{"key": k, "label": labels[k], "needed": n} for k, n in session.need.items()],
    )


@router.post("/sessions/{session_id}/frame", response_model=FrameResult)
async def submit_frame(
    session_id: str, body: FrameRequest, current_user: ActiveUser
) -> FrameResult:
    current_user.require_role("admin")
    capture = get_manual_enroll_capture()
    session = capture.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    if session.pending == []:
        return FrameResult(face_found=False, progress=len(session.embeddings),
                            total=session.crops_total, pending=[], done=True)

    raw = body.image_b64.split(",", 1)[-1]
    try:
        jpeg_bytes = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")

    import cv2

    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    app = get_pose_detector()
    faces = app.get(frame)
    face = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ) if faces else None

    if (
        face is None
        or face.normed_embedding is None
        or getattr(face, "pose", None) is None
    ):
        return FrameResult(
            face_found=face is not None, progress=len(session.embeddings),
            total=session.crops_total, pending=session.pending, done=False,
        )

    pitch, yaw = float(face.pose[0]), float(face.pose[1])
    bucket, captured = capture.try_capture(
        session_id, yaw, pitch, face.normed_embedding.copy()
    )
    return FrameResult(
        face_found=True, yaw=yaw, pitch=pitch, bucket=bucket, captured=captured,
        progress=len(session.embeddings), total=session.crops_total,
        pending=session.pending, done=session.pending == [],
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def discard_session(session_id: str, current_user: ActiveUser) -> None:
    current_user.require_role("admin")
    get_manual_enroll_capture().discard(session_id)


async def _fetch_person(db: AsyncSession, person_id: str):
    return (await db.execute(
        text("""
            SELECT person_id, display_name, role, enrolled_at, is_active,
                   last_seen_at, last_seen_zone, metadata
            FROM persons WHERE person_id = :pid AND is_active
        """),
        {"pid": person_id},
    )).fetchone()


@router.get("/sessions/{session_id}/suggestion", response_model=SuggestionResult)
async def session_suggestion(
    session_id: str,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> SuggestionResult:
    """Does this captured face already look like an enrolled person? The
    review UI calls this before finalizing so the operator can choose to
    merge into the match instead of creating a duplicate identity."""
    current_user.require_role("admin")
    session = get_manual_enroll_capture().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    pose_embeddings = [l2_normalize(e) for e in session.pose_means()]
    if not pose_embeddings:
        return SuggestionResult(match=None)
    search_vec = l2_normalize(np.mean(pose_embeddings, axis=0))
    matches = await get_gallery().search(search_vec, top_k=1)
    if not matches or matches[0].similarity < SUGGEST_THRESHOLD:
        return SuggestionResult(match=None)
    person = await _fetch_person(db, str(matches[0].person_id))
    if person is None:
        return SuggestionResult(match=None)
    sim = round(float(matches[0].similarity), 3)
    return SuggestionResult(match=SuggestedMatch(
        person_id=person.person_id,
        display_name=person.display_name,
        similarity=sim,
        likely_same=sim >= ACCEPT_THRESHOLD,
    ))


@router.post("/sessions/{session_id}/finalize", response_model=FinalizeResult,
             status_code=status.HTTP_201_CREATED)
async def finalize_session(
    session_id: str,
    body: FinalizeRequest,
    current_user: ActiveUser,
    db: AsyncSession = Depends(get_db),
) -> FinalizeResult:
    current_user.require_role("admin")
    capture = get_manual_enroll_capture()
    session = capture.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if session.pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Capture incomplete: still need {', '.join(session.pending)}.",
        )

    # One template per pose bucket (frontal/left/right/up/down), so the side
    # views are stored as their own gallery rows rather than averaged away.
    pose_embeddings = [l2_normalize(e) for e in session.pose_means()]
    if not pose_embeddings:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No usable captures in this session.")
    # The operator made the call (merge vs. new) in the review UI, informed by
    # the /suggestion lookup — finalize honors that choice rather than merging
    # silently behind their back.
    merged_into_existing = body.person_id is not None
    matched_similarity: float | None = None
    if body.person_id is not None:
        person = await _fetch_person(db, str(body.person_id))
        if person is None:
            raise HTTPException(status_code=404, detail="Target person not found or inactive.")
    else:
        if not body.display_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="display_name required when creating a new person "
                       "(or pass person_id to add to an existing one).",
            )
        person = (await db.execute(
            text("""
                INSERT INTO persons (display_name, role, enrolled_by)
                VALUES (:name, :role, :uid)
                RETURNING person_id, display_name, role, enrolled_at,
                          is_active, last_seen_at, last_seen_zone, metadata
            """),
            {"name": body.display_name, "role": body.role,
             "uid": str(current_user.user_id)},
        )).fetchone()
        await db.commit()

    for emb in pose_embeddings:
        await get_gallery().add_embedding(
            person_id=str(person.person_id),
            embedding=emb,
            quality_score=MANUAL_ENROLL_QUALITY,
            camera_id="manual-enroll",
        )
    capture.discard(session_id)

    await write_audit_event(
        db, "IDENTITY_ENROLLED_MANUAL",
        actor_id=current_user.user_id,
        actor_username=current_user.username,
        resource_type="person",
        resource_id=str(person.person_id),
        details={"display_name": person.display_name,
                 "templates_added": len(pose_embeddings),
                 "merged_into_existing": merged_into_existing,
                 "matched_similarity": matched_similarity},
    )

    person_out = PersonOut(
        person_id=person.person_id,
        display_name=person.display_name,
        role=person.role,
        enrolled_at=person.enrolled_at.isoformat(),
        is_active=person.is_active,
        last_seen_at=person.last_seen_at.isoformat() if getattr(person, "last_seen_at", None) else None,
        last_seen_zone=getattr(person, "last_seen_zone", None),
        metadata=person.metadata or {},
    )
    return FinalizeResult(
        person=person_out,
        merged_into_existing=merged_into_existing,
        matched_similarity=matched_similarity,
        templates_added=len(pose_embeddings),
    )
