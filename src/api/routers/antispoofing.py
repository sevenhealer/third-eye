from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import ActiveUser
from src.api.routers.cameras import KICK_CHANNEL
from src.core.database import get_db, get_redis
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "data" / "antispoofing"
TRAIN_SCRIPT = ROOT / "scripts" / "train_antispoofing.py"
TRAIN_LOG = ROOT / "data" / "logs" / "antispoofing_train.log"

LABELS = ("live", "spoof")
# Crops are written by run_live as <camera>_<frame>_t<track>.jpg — a safe,
# fixed character set. Reject anything else so a crafted name can't escape the
# dataset dir (path traversal) when we move/delete/serve a file.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.jpg$")

# Module-level handle to the at-most-one training subprocess. The API runs as a
# single process (the camera supervisor lives in its lifespan), so a module
# global is sufficient — no cross-worker coordination needed.
_train_proc: subprocess.Popen | None = None


def _safe_crop_path(label: str, filename: str) -> Path:
    if label not in LABELS or not _SAFE_NAME.match(filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid crop reference.")
    return DATASET_DIR / label / filename


class DatasetItem(BaseModel):
    label: str
    filename: str
    url: str


class DatasetSummary(BaseModel):
    live: int
    spoof: int
    items: list[DatasetItem]


@router.get("/dataset", response_model=DatasetSummary)
async def get_dataset(current_user: ActiveUser, limit: int = 200) -> DatasetSummary:
    """Counts per label + the most recent crops (newest first) for review."""
    current_user.require_role("admin")
    counts = {lbl: 0 for lbl in LABELS}
    everything: list[tuple[float, str, str]] = []
    for lbl in LABELS:
        d = DATASET_DIR / lbl
        if not d.is_dir():
            continue
        for f in d.glob("*.jpg"):
            counts[lbl] += 1
            everything.append((f.stat().st_mtime, lbl, f.name))
    everything.sort(key=lambda t: t[0], reverse=True)
    items = [
        DatasetItem(label=lbl, filename=name,
                    url=f"/api/v1/antispoofing/crop/{lbl}/{name}")
        for _, lbl, name in everything[:limit]
    ]
    return DatasetSummary(live=counts["live"], spoof=counts["spoof"], items=items)


@router.get("/crop/{label}/{filename}")
async def get_crop(label: str, filename: str, current_user: ActiveUser) -> Response:
    """Stream a dataset crop image (auth'd; the browser fetches it as a blob)."""
    current_user.require_role("admin")
    path = _safe_crop_path(label, filename)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Crop not found.")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


class RelabelBody(BaseModel):
    label: str
    filename: str
    new_label: str


@router.post("/relabel")
async def relabel_crop(body: RelabelBody, current_user: ActiveUser) -> dict:
    """Correct an auto-label by moving the crop to the other label's folder."""
    current_user.require_role("admin")
    if body.new_label not in LABELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid target label.")
    src = _safe_crop_path(body.label, body.filename)
    if not src.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Crop not found.")
    dst_dir = DATASET_DIR / body.new_label
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / body.filename
    src.rename(dst)
    logger.info("antispoofing_relabel", **{"from": body.label}, to=body.new_label,
                filename=body.filename)
    return {"label": body.new_label, "filename": body.filename}


@router.delete("/crop/{label}/{filename}")
async def delete_crop(label: str, filename: str, current_user: ActiveUser) -> Response:
    """Remove a crop from the dataset (e.g. a junk/blurred capture)."""
    current_user.require_role("admin")
    path = _safe_crop_path(label, filename)
    if path.is_file():
        path.unlink()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/dataset")
async def clear_dataset(current_user: ActiveUser, label: str | None = None) -> dict:
    """Bulk-delete crops. ``label`` clears just that label (live/spoof);
    omitting it wipes the whole dataset. Useful to discard a contaminated set
    before re-collecting clean."""
    current_user.require_role("admin")
    targets = LABELS if label is None else (label,)
    if label is not None and label not in LABELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid label.")
    deleted = 0
    for lbl in targets:
        d = DATASET_DIR / lbl
        if not d.is_dir():
            continue
        for f in d.glob("*.jpg"):
            f.unlink()
            deleted += 1
    logger.info("antispoofing_dataset_cleared", label=label or "all", deleted=deleted)
    return {"label": label or "all", "deleted": deleted}


class TrainStatus(BaseModel):
    running: bool
    latest_epoch: int | None
    total_epochs: int | None
    best_val_acc: float | None
    finished: bool
    tail: list[str]


def _parse_train_log() -> TrainStatus:
    running = _train_proc is not None and _train_proc.poll() is None
    latest_epoch = total_epochs = None
    best_val_acc = None
    tail: list[str] = []
    if TRAIN_LOG.is_file():
        lines = TRAIN_LOG.read_text(errors="replace").splitlines()
        tail = lines[-12:]
        for ln in lines:
            m = re.search(r"epoch\s+(\d+)/(\d+)\s+.*val_acc=([0-9.]+)", ln)
            if m:
                latest_epoch, total_epochs = int(m.group(1)), int(m.group(2))
                acc = float(m.group(3))
                best_val_acc = acc if best_val_acc is None else max(best_val_acc, acc)
    finished = (not running) and _train_proc is not None
    return TrainStatus(
        running=running, latest_epoch=latest_epoch, total_epochs=total_epochs,
        best_val_acc=best_val_acc, finished=finished, tail=tail,
    )


class TrainBody(BaseModel):
    epochs: int = 30


@router.post("/train", response_model=TrainStatus)
async def start_training(body: TrainBody, current_user: ActiveUser) -> TrainStatus:
    """Kick off CDCN++ training on the current dataset as a background job."""
    current_user.require_role("admin")
    global _train_proc
    if _train_proc is not None and _train_proc.poll() is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Training already in progress.")
    live_n = len(list((DATASET_DIR / "live").glob("*.jpg"))) if (DATASET_DIR / "live").is_dir() else 0
    spoof_n = len(list((DATASET_DIR / "spoof").glob("*.jpg"))) if (DATASET_DIR / "spoof").is_dir() else 0
    if live_n == 0 or spoof_n == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Need both live and spoof crops to train (have live={live_n}, spoof={spoof_n}).",
        )
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(TRAIN_LOG, "w")  # noqa: SIM115 — handed to the child for its lifetime
    _train_proc = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT), "--epochs", str(body.epochs), "--sign"],
        stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
    )
    logger.info("antispoofing_train_started", epochs=body.epochs, pid=_train_proc.pid)
    return _parse_train_log()


@router.get("/train/status", response_model=TrainStatus)
async def training_status(current_user: ActiveUser) -> TrainStatus:
    current_user.require_role("admin")
    return _parse_train_log()


# ── Collection start/stop (per camera) ───────────────────────────────────────

class CollectionCamera(BaseModel):
    camera_id: str
    display_name: str
    desired_state: str
    collecting: bool


@router.get("/collection", response_model=list[CollectionCamera])
async def list_collection(
    current_user: ActiveUser, db: AsyncSession = Depends(get_db)
) -> list[CollectionCamera]:
    """Active cameras and whether each is currently collecting anti-spoof data."""
    current_user.require_role("admin")
    rows = (await db.execute(text(
        "SELECT camera_id, display_name, desired_state, "
        "COALESCE((launch_args->>'collect_antispoofing')::boolean, false) AS collecting "
        "FROM cameras WHERE is_active = true ORDER BY display_name"
    ))).fetchall()
    return [
        CollectionCamera(camera_id=r.camera_id, display_name=r.display_name,
                         desired_state=r.desired_state, collecting=r.collecting)
        for r in rows
    ]


class CollectionToggle(BaseModel):
    camera_id: str
    enabled: bool


@router.post("/collection")
async def set_collection(
    body: CollectionToggle, current_user: ActiveUser, db: AsyncSession = Depends(get_db)
) -> dict:
    """Start/stop anti-spoof data collection on a camera. Flips the camera's
    collect_antispoofing launch arg and bumps config_version so the supervisor
    restarts it with the change."""
    current_user.require_role("admin")
    result = await db.execute(text(
        "UPDATE cameras SET "
        "launch_args = jsonb_set(COALESCE(launch_args, '{}'::jsonb), "
        "'{collect_antispoofing}', to_jsonb(cast(:enabled AS boolean))), "
        "config_version = config_version + 1 "
        "WHERE camera_id = :cid AND is_active = true"
    ), {"enabled": body.enabled, "cid": body.camera_id})
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found.")
    await db.commit()
    try:  # nudge the supervisor to apply immediately (poll is the backstop)
        redis = await get_redis()
        await redis.publish(KICK_CHANNEL, "1")
    except Exception as exc:
        logger.warning("antispoofing_collection_kick_failed", error=str(exc))
    logger.info("antispoofing_collection_set", camera_id=body.camera_id, enabled=body.enabled)
    return {"camera_id": body.camera_id, "collecting": body.enabled}
