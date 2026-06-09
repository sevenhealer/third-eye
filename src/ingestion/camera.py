from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


class CameraReader:
    """
    OpenCV-backed camera reader for RTSP streams and USB devices.

    source: int for USB (/dev/video0 = 0) or str for RTSP/file URL.
    Reconnection uses exponential backoff up to reconnect_max_seconds.
    """

    def __init__(
        self,
        source: str | int,
        reconnect_max_seconds: int = 60,
    ) -> None:
        self.source = source
        self.reconnect_max_seconds = reconnect_max_seconds
        self._cap: Any = None

    def open(self) -> bool:
        """Open the capture device. Returns True on success."""
        if not _CV2_AVAILABLE:
            logger.error(
                "camera_open_failed",
                reason="opencv not installed — run: pip install opencv-python-headless",
            )
            return False
        cap = _cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            logger.warning("camera_open_failed", source=str(self.source))
            return False
        self._cap = cap
        logger.info("camera_opened", source=str(self.source), **self._current_props())
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read one BGR frame.
        Returns (True, frame) on success, (False, None) on failure or when closed.
        """
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    def props(self) -> dict:
        """Return width, height, fps, backend as reported by the driver."""
        if self._cap is None or not _CV2_AVAILABLE:
            return {"source": str(self.source), "opened": False}
        return {
            "source": str(self.source),
            "width": int(self._cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._cap.get(_cv2.CAP_PROP_FPS)),
            "backend": self._cap.getBackendName(),
        }

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("camera_released", source=str(self.source))

    def reopen_with_backoff(self) -> bool:
        """
        Retry open() with exponential backoff until reconnect_max_seconds elapses.
        Returns True if successfully reopened.
        """
        self.release()
        delay = 2.0
        deadline = time.monotonic() + self.reconnect_max_seconds
        while time.monotonic() < deadline:
            logger.info("camera_reconnect_attempt", source=str(self.source), backoff_s=delay)
            time.sleep(delay)
            if self.open():
                return True
            delay = min(delay * 2, 30.0)
        logger.error("camera_reconnect_exhausted", source=str(self.source))
        return False

    def _current_props(self) -> dict:
        p = self.props()
        p.pop("source", None)
        return p

    def __enter__(self) -> "CameraReader":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()
