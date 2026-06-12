from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None   # keep name defined so patch() can find it
    _ULTRALYTICS_AVAILABLE = False


class ModelNotLoadedError(Exception):
    pass


@dataclass
class ObjectDetection:
    bbox: np.ndarray       # [x1, y1, x2, y2] float32
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray | None = None   # appearance vector (e.g. face embedding) for ReID-assisted tracking


class YOLODetector:
    """
    Object detector backed by ultralytics YOLO (v8/v9/v10).

    Guard: requires `ultralytics` package. Call load() before detect().
    class_filter: if set, only return detections for those class IDs.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.25,
        class_filter: list[int] | None = None,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._conf = conf_threshold
        self._class_filter = set(class_filter) if class_filter else None
        self._imgsz = imgsz
        self._device = device
        self._model = None

    def load(self) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ModelNotLoadedError(
                "ultralytics is not installed — run: pip install ultralytics"
            )
        self._model = _YOLO(self._model_name)
        if self._device:
            self._model.to(self._device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        if self._model is None:
            raise ModelNotLoadedError("Call load() before detect()")

        results = self._model(frame, conf=self._conf, imgsz=self._imgsz, verbose=False)
        detections: list[ObjectDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                if self._class_filter is not None and class_id not in self._class_filter:
                    continue
                conf = float(box.conf[0])
                if conf < self._conf:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype("float32")
                class_name = result.names.get(class_id, str(class_id))
                detections.append(
                    ObjectDetection(
                        bbox=xyxy,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                    )
                )
        return detections
