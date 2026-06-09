from __future__ import annotations

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import torchreid as _torchreid
    _TORCHREID_AVAILABLE = True
except ImportError:
    _torchreid = None  # type: ignore[assignment]
    _TORCHREID_AVAILABLE = False

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class OSNetReID:
    """
    OSNet-x0.75 appearance feature extractor for person Re-ID.

    Extracts a normalized 512-dim L2 embedding from a BGR person crop.
    StrongSortTracker uses these embeddings to link tracks across occlusions
    and across cameras via CrossCameraCoordinator.

    When torchreid is unavailable (test/CI without the ML stack), every call
    returns np.zeros(512) — callers with zero embeddings automatically fall
    back to pure IoU matching.
    """

    EMBED_DIM = 512
    INPUT_W = 128
    INPUT_H = 256

    def __init__(self, device: str = "cpu", weights_path: str | None = None) -> None:
        self.device = device
        self._model = None
        if _TORCHREID_AVAILABLE and _TORCH_AVAILABLE:
            self._load(weights_path)
        else:
            logger.warning("osnet_unavailable", reason="torchreid or torch not installed")

    def _load(self, weights_path: str | None) -> None:
        try:
            model = _torchreid.models.build_model(
                name="osnet_x0_75",
                num_classes=1,
                pretrained=(weights_path is None),
            )
            if weights_path:
                _torchreid.utils.load_pretrained_weights(model, weights_path)
            model.eval()
            model.to(self.device)
            self._model = model
            logger.info("osnet_loaded", device=self.device)
        except Exception as exc:
            logger.warning("osnet_load_failed", error=str(exc))

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Return a normalized 512-dim embedding for a BGR person crop.
        Returns np.zeros(512) when the model is unavailable.
        """
        if self._model is None or not _CV2_AVAILABLE or not _TORCH_AVAILABLE:
            return np.zeros(self.EMBED_DIM, dtype=np.float32)
        try:
            img = _cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array(_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(_STD, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std
            t = _torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            with _torch.no_grad():
                feat = self._model(t)
            vec = feat.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-6 else vec
        except Exception as exc:
            logger.warning("osnet_extract_error", error=str(exc))
            return np.zeros(self.EMBED_DIM, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


_reid: OSNetReID | None = None


def get_reid(device: str = "cpu") -> OSNetReID:
    global _reid
    if _reid is None:
        _reid = OSNetReID(device=device)
    return _reid
