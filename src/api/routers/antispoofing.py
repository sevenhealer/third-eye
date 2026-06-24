from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
SNAPSHOTS_DIR = ROOT / "data" / "antispoofing_snapshots"
CKPT_DIR = ROOT / "models" / "weights" / "cdcn_checkpoints"
ACTIVE_CKPT = ROOT / "models" / "weights" / "cdcn_pp.pt"
ACTIVE_META = ROOT / "models" / "weights" / "cdcn_pp.json"

LABELS = ("live", "spoof")
# Crops are written by run_live as <camera>_<frame>_t<track>.jpg — a safe,
# fixed character set. Reject anything else so a crafted name can't escape the
# dataset dir (path traversal) when we move/delete/serve a file.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.jpg$")
# Snapshot / checkpoint names are operator-supplied — constrain them hard so a
# crafted name can't escape its directory.
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Module-level handle to the at-most-one training subprocess. The API runs as a
# single process (the camera supervisor lives in its lifespan), so a module
# global is sufficient — no cross-worker coordination needed.
_train_proc: subprocess.Popen | None = None


def _safe_crop_path(label: str, filename: str) -> Path:
    if label not in LABELS or not _SAFE_NAME.match(filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid crop reference.")
    return DATASET_DIR / label / filename


def _safe_slug(name: str) -> str:
    if not _SAFE_SLUG.match(name):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Name must be 1-64 chars of letters, digits, . _ - only.")
    return name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_meta(json_path: Path) -> dict:
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return {}


def _count_jpgs(d: Path) -> int:
    return len(list(d.glob("*.jpg"))) if d.is_dir() else 0


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
    # Auto-archive the model we're about to overwrite: training writes
    # cdcn_pp.pt in place, so without this a worse retrain would destroy a
    # better model with no way back. The operator can re-activate it later.
    if ACTIVE_CKPT.is_file():
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archive = CKPT_DIR / f"pretrain-{stamp}.pt"
        shutil.copy2(ACTIVE_CKPT, archive)
        if ACTIVE_META.is_file():
            shutil.copy2(ACTIVE_META, archive.with_suffix(".json"))
        logger.info("antispoofing_pretrain_archived", name=archive.stem)
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


# ── Dataset snapshots (save labelled set → clear → re-collect, restore later) ──

class Snapshot(BaseModel):
    name: str
    live: int
    spoof: int
    created: str | None


class NameBody(BaseModel):
    name: str


@router.get("/snapshots", response_model=list[Snapshot])
async def list_snapshots(current_user: ActiveUser) -> list[Snapshot]:
    """Saved dataset snapshots, newest first, with their per-label counts."""
    current_user.require_role("admin")
    out: list[Snapshot] = []
    if SNAPSHOTS_DIR.is_dir():
        for d in sorted(SNAPSHOTS_DIR.iterdir(),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta = _read_meta(d / "meta.json")
            out.append(Snapshot(name=d.name, live=_count_jpgs(d / "live"),
                                spoof=_count_jpgs(d / "spoof"),
                                created=meta.get("created")))
    return out


@router.post("/snapshots", response_model=Snapshot)
async def save_snapshot(body: NameBody, current_user: ActiveUser) -> Snapshot:
    """Copy the current labelled dataset into a named snapshot, so it can be
    cleared and rebuilt without losing the labels."""
    current_user.require_role("admin")
    name = _safe_slug(body.name)
    dst = SNAPSHOTS_DIR / name
    if dst.exists():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "A snapshot with that name already exists.")
    live = spoof = 0
    for lbl in LABELS:
        (dst / lbl).mkdir(parents=True, exist_ok=True)
        src = DATASET_DIR / lbl
        if src.is_dir():
            for f in src.glob("*.jpg"):
                shutil.copy2(f, dst / lbl / f.name)
                if lbl == "live":
                    live += 1
                else:
                    spoof += 1
    created = datetime.now(timezone.utc).isoformat()
    (dst / "meta.json").write_text(json.dumps(
        {"created": created, "live": live, "spoof": spoof}))
    logger.info("antispoofing_snapshot_saved", name=name, live=live, spoof=spoof)
    return Snapshot(name=name, live=live, spoof=spoof, created=created)


@router.post("/snapshots/{name}/restore")
async def restore_snapshot(name: str, current_user: ActiveUser) -> dict:
    """Replace the working dataset with a snapshot (clears the current crops
    first, then copies the snapshot's in)."""
    current_user.require_role("admin")
    name = _safe_slug(name)
    src = SNAPSHOTS_DIR / name
    if not src.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Snapshot not found.")
    restored = 0
    for lbl in LABELS:
        wd = DATASET_DIR / lbl
        wd.mkdir(parents=True, exist_ok=True)
        for f in wd.glob("*.jpg"):
            f.unlink()
        sd = src / lbl
        if sd.is_dir():
            for f in sd.glob("*.jpg"):
                shutil.copy2(f, wd / f.name)
                restored += 1
    logger.info("antispoofing_snapshot_restored", name=name, restored=restored)
    return {"name": name, "restored": restored}


@router.delete("/snapshots/{name}")
async def delete_snapshot(name: str, current_user: ActiveUser) -> Response:
    current_user.require_role("admin")
    name = _safe_slug(name)
    d = SNAPSHOTS_DIR / name
    if d.is_dir():
        shutil.rmtree(d)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Model checkpoints (archive/restore CDCN++ weights; training overwrites) ───

class Checkpoint(BaseModel):
    name: str
    val_acc: float | None
    epoch: int | None
    created: str | None
    source: str | None
    active: bool


@router.get("/checkpoints", response_model=list[Checkpoint])
async def list_checkpoints(current_user: ActiveUser) -> list[Checkpoint]:
    """The active model plus every saved checkpoint, newest first. ``active``
    marks whichever archived file currently matches cdcn_pp.pt by content."""
    current_user.require_role("admin")
    active_sha = _sha256(ACTIVE_CKPT) if ACTIVE_CKPT.is_file() else None
    out: list[Checkpoint] = []
    if ACTIVE_CKPT.is_file():
        m = _read_meta(ACTIVE_META)
        out.append(Checkpoint(name="(active model)", val_acc=m.get("val_acc"),
                              epoch=m.get("epoch"), created=m.get("created"),
                              source=m.get("source", "active"), active=True))
    if CKPT_DIR.is_dir():
        for pt in sorted(CKPT_DIR.glob("*.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            m = _read_meta(pt.with_suffix(".json"))
            out.append(Checkpoint(
                name=pt.stem, val_acc=m.get("val_acc"), epoch=m.get("epoch"),
                created=m.get("created"), source=m.get("source"),
                active=active_sha is not None and _sha256(pt) == active_sha))
    return out


@router.post("/checkpoints", response_model=Checkpoint)
async def save_checkpoint(body: NameBody, current_user: ActiveUser) -> Checkpoint:
    """Archive the current active model under a name so a later retrain can't
    destroy it."""
    current_user.require_role("admin")
    name = _safe_slug(body.name)
    if not ACTIVE_CKPT.is_file():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No trained model to save yet.")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dst = CKPT_DIR / f"{name}.pt"
    if dst.exists():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "A checkpoint with that name already exists.")
    shutil.copy2(ACTIVE_CKPT, dst)
    meta = _read_meta(ACTIVE_META)
    meta.setdefault("created", datetime.now(timezone.utc).isoformat())
    meta.setdefault("source", "saved")
    dst.with_suffix(".json").write_text(json.dumps(meta))
    logger.info("antispoofing_checkpoint_saved", name=name)
    return Checkpoint(name=name, val_acc=meta.get("val_acc"), epoch=meta.get("epoch"),
                      created=meta.get("created"), source=meta.get("source"), active=True)


@router.post("/checkpoints/{name}/activate")
async def activate_checkpoint(name: str, current_user: ActiveUser) -> dict:
    """Make a saved checkpoint the active model (cdcn_pp.pt) and re-sign it.
    Cameras load it on their next restart."""
    current_user.require_role("admin")
    name = _safe_slug(name)
    src = CKPT_DIR / f"{name}.pt"
    if not src.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Checkpoint not found.")
    shutil.copy2(src, ACTIVE_CKPT)
    sidecar = src.with_suffix(".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, ACTIVE_META)
    # cdcn_pp.pt is a signed model (verify-on-startup, fail-closed) — swapping
    # the bytes without re-signing would brick the next boot, so re-sign now.
    try:
        from src.core.model_registry import DEFAULT_MANIFEST_PATH, ModelRegistry
        reg = ModelRegistry()
        if DEFAULT_MANIFEST_PATH.exists():
            reg.load_manifest(DEFAULT_MANIFEST_PATH)
        reg.sign("cdcn_pp", ACTIVE_CKPT)
        reg.save_manifest(DEFAULT_MANIFEST_PATH)
    except Exception as exc:
        logger.error("antispoofing_checkpoint_resign_failed", error=str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Activated, but re-signing failed: {exc}")
    logger.info("antispoofing_checkpoint_activated", name=name)
    return {"name": name, "active": True,
            "note": "Restart the camera(s) to load the activated model."}


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str, current_user: ActiveUser) -> Response:
    current_user.require_role("admin")
    name = _safe_slug(name)
    pt = CKPT_DIR / f"{name}.pt"
    if pt.is_file():
        pt.unlink()
    js = pt.with_suffix(".json")
    if js.is_file():
        js.unlink()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
