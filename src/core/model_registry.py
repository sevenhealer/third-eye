from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.exceptions import ModelChecksumError, ModelLoadError
from src.core.logging import get_logger

logger = get_logger(__name__)

_CHUNK = 65536  # 64 KiB read chunks for large weight files


def compute_sha256(path: Path) -> str:
    """Return hex SHA-256 digest of *path*, streaming to avoid loading it all into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class ModelEntry:
    name: str
    path: str  # absolute or relative path as stored
    sha256: str
    signed_at: str  # ISO-8601 UTC
    version: str = "1.0.0"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class ModelRegistry:
    """
    Tracks SHA-256 hashes for all model weight files.

    Sign once after training/downloading. Verify at every startup.
    Any mismatch raises ModelChecksumError — never silently continues.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    # ── signing ───────────────────────────────────────────────────────────────

    def sign(self, name: str, path: Path, version: str = "1.0.0") -> ModelEntry:
        """
        Compute SHA-256 for *path* and register it under *name*.

        Returns the new ModelEntry. Overwrites any existing entry for *name*.
        Raises ModelLoadError if the file does not exist.
        """
        if not path.exists():
            raise ModelLoadError(f"Cannot sign: file not found: {path}")
        digest = compute_sha256(path)
        entry = ModelEntry(
            name=name,
            path=str(path),
            sha256=digest,
            signed_at=datetime.now(timezone.utc).isoformat(),
            version=version,
        )
        self._entries[name] = entry
        logger.info("model_signed", name=name, sha256=digest[:16] + "…", path=str(path))
        return entry

    # ── verification ──────────────────────────────────────────────────────────

    def verify(self, name: str) -> bool:
        """
        Verify the current file SHA-256 matches the registered hash.

        Returns True on match. Raises ModelChecksumError on mismatch or missing file.
        Raises KeyError if *name* is not registered.
        """
        entry = self._entries[name]
        p = Path(entry.path)
        if not p.exists():
            raise ModelChecksumError(f"Model file missing: {p}")
        actual = compute_sha256(p)
        if actual != entry.sha256:
            raise ModelChecksumError(
                f"SHA-256 mismatch for '{name}': expected {entry.sha256[:16]}… "
                f"got {actual[:16]}…  — possible tampering or corruption"
            )
        logger.info("model_verified", name=name, ok=True)
        return True

    def verify_all(self) -> dict[str, bool]:
        """
        Verify every registered model. Returns {name: True} for all passing entries.

        Raises ModelChecksumError immediately on the first failure.
        """
        results: dict[str, bool] = {}
        for name in list(self._entries):
            results[name] = self.verify(name)
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save_manifest(self, path: Path) -> None:
        """Persist all entries to a JSON manifest file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: entry.as_dict() for name, entry in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))
        logger.info("manifest_saved", path=str(path), count=len(data))

    def load_manifest(self, path: Path) -> None:
        """
        Load entries from a JSON manifest file.

        Entries are merged into the registry — existing entries with the same name
        are overwritten.
        Raises ModelLoadError if the file is missing or malformed.
        """
        if not path.exists():
            raise ModelLoadError(f"Model manifest not found: {path}")
        try:
            data: dict = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ModelLoadError(f"Malformed model manifest at {path}: {exc}") from exc
        for name, raw in data.items():
            self._entries[name] = ModelEntry(**raw)
        logger.info("manifest_loaded", path=str(path), count=len(data))

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def entries(self) -> list[ModelEntry]:
        return list(self._entries.values())


# ── module-level singleton ────────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def startup_verify(manifest_path: Path) -> None:
    """
    Load manifest and verify all registered weights.

    Called during API lifespan startup. Raises ModelChecksumError on tamper/corruption.
    No-ops if the manifest file does not exist (first boot before any model is signed).
    """
    if not manifest_path.exists():
        logger.info("model_manifest_absent_skipping_verification", path=str(manifest_path))
        return
    reg = get_registry()
    reg.load_manifest(manifest_path)
    results = reg.verify_all()
    logger.info("startup_model_verification_complete", verified=len(results))
