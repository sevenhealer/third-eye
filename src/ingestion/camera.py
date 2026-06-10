from __future__ import annotations

import os
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

try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _av = None  # type: ignore[assignment]
    _AV_AVAILABLE = False


def _is_url(source: str | int) -> bool:
    return isinstance(source, str) and (
        source.startswith("rtsp://")
        or source.startswith("rtsps://")
        or source.startswith("http://")
        or source.startswith("https://")
    )


class _PyAVReader:
    """
    PyAV-backed RTSP/URL reader.

    Used in place of OpenCV's VideoCapture for network streams on macOS because
    OpenCV ships its own bundled FFmpeg whose network stack can't reach hosts
    that Homebrew FFmpeg (used by PyAV) connects to without issue.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._container: Any = None
        self._video_stream: Any = None
        self._frame_gen: Any = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0

    def open(self) -> bool:
        try:
            container = _av.open(
                self._url,
                options={"rtsp_transport": "tcp", "stimeout": "10000000"},
            )
            video_streams = container.streams.video
            if not video_streams:
                container.close()
                return False
            vs = video_streams[0]
            vs.thread_type = "AUTO"
            self._container = container
            self._video_stream = vs
            self._width = vs.codec_context.width
            self._height = vs.codec_context.height
            self._fps = float(vs.average_rate or vs.guessed_rate or 25)
            self._frame_gen = container.decode(video=0)
            return True
        except Exception as exc:
            logger.warning("pyav_open_failed", url=self._url, error=str(exc))
            return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._frame_gen is None:
            return False, None
        try:
            av_frame = next(self._frame_gen)
            bgr = av_frame.to_ndarray(format="bgr24")
            return True, bgr
        except StopIteration:
            return False, None
        except Exception as exc:
            logger.warning("pyav_read_error", error=str(exc))
            return False, None

    def isOpened(self) -> bool:
        return self._container is not None

    def getBackendName(self) -> str:
        return "PyAV/FFmpeg"

    def get(self, prop_id: int) -> float:
        if not _CV2_AVAILABLE:
            return 0.0
        if prop_id == _cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == _cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == _cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def release(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._frame_gen = None


class CameraReader:
    """
    Camera reader that supports RTSP/URL streams and USB devices.

    For network URLs (rtsp://, rtsps://, http://):
      - Uses PyAV when available — links against Homebrew FFmpeg on macOS,
        which correctly handles network streams that OpenCV's bundled FFmpeg fails on.
      - Falls back to OpenCV CAP_FFMPEG if PyAV is not installed.

    For integer sources (USB cameras): always uses OpenCV.

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
        self._use_pyav: bool = False

    def open(self) -> bool:
        """Open the capture device. Returns True on success."""
        if _is_url(self.source):
            return self._open_url(str(self.source))
        return self._open_cv(self.source)

    def _open_url(self, url: str) -> bool:
        if _AV_AVAILABLE:
            reader = _PyAVReader(url)
            if reader.open():
                self._cap = reader
                self._use_pyav = True
                logger.info("camera_opened", source=url, backend="PyAV/FFmpeg",
                            width=int(reader.get(_cv2.CAP_PROP_FRAME_WIDTH)) if _CV2_AVAILABLE else 0,
                            height=int(reader.get(_cv2.CAP_PROP_FRAME_HEIGHT)) if _CV2_AVAILABLE else 0,
                            fps=reader.get(_cv2.CAP_PROP_FPS) if _CV2_AVAILABLE else 0)
                return True
            logger.warning("pyav_open_failed_falling_back_to_opencv", source=url)

        # PyAV unavailable or failed — try OpenCV with explicit FFmpeg backend
        return self._open_cv(url)

    def _open_cv(self, source: str | int) -> bool:
        if not _CV2_AVAILABLE:
            logger.error("camera_open_failed",
                         reason="opencv not installed — run: pip install opencv-python-headless")
            return False
        if isinstance(source, str):
            cap = _cv2.VideoCapture(source, _cv2.CAP_FFMPEG)
            cap.set(_cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        else:
            cap = _cv2.VideoCapture(source)
            # keep the driver queue at one frame so reads stay near-live;
            # ignored by backends that don't support it
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            logger.warning("camera_open_failed", source=str(source))
            return False
        self._cap = cap
        self._use_pyav = False
        logger.info("camera_opened", source=str(source), **self._current_props())
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read one BGR frame.
        Returns (True, frame) on success, (False, None) on failure or when closed.
        """
        if self._cap is None:
            return False, None
        if self._use_pyav:
            return self._cap.read_frame()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        if self._cap is None:
            return False
        if self._use_pyav:
            return self._cap.isOpened()
        return bool(self._cap.isOpened())

    def props(self) -> dict:
        """Return width, height, fps, backend as reported by the driver."""
        if self._cap is None:
            return {"source": str(self.source), "opened": False}
        if _CV2_AVAILABLE:
            return {
                "source": str(self.source),
                "width": int(self._cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(self._cap.get(_cv2.CAP_PROP_FPS)),
                "backend": self._cap.getBackendName(),
            }
        return {"source": str(self.source), "opened": True}

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._use_pyav = False
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
