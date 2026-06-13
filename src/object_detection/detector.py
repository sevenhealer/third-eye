from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    from ultralytics import YOLOWorld as _YOLOWorld
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None   # keep name defined so patch() can find it
    _YOLOWorld = None
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
    Object detector backed by ultralytics YOLO (v8 → YOLO26).

    Two modes:
      - Closed-set (default): the model's trained classes, e.g. COCO's 80.
        Anything outside them is mapped to the nearest class or missed —
        an almirah becomes "fridge", an iPad "laptop".
      - Open-vocabulary: pass `vocab=[...]` of free-text class names; loads
        YOLO-World and detects exactly those, so domain objects (almirah,
        iPad, water bottle, ...) are found and labelled correctly without
        retraining. Slower per frame than closed-set.

    Guard: requires `ultralytics`. Call load() before detect().
    class_filter (closed-set only): keep just these class IDs.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.25,
        class_filter: list[int] | None = None,
        imgsz: int = 640,
        device: str | None = None,
        vocab: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._conf = conf_threshold
        self._class_filter = set(class_filter) if class_filter else None
        self._imgsz = imgsz
        self._device = device
        self._vocab = vocab
        self._model = None

    @property
    def is_open_vocab(self) -> bool:
        return self._vocab is not None

    def load(self) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ModelNotLoadedError(
                "ultralytics is not installed — run: pip install ultralytics"
            )
        if self._vocab is not None:
            # default to a YOLO-World checkpoint if the caller didn't name one
            name = self._model_name
            if "world" not in name.lower():
                name = "yolov8x-worldv2.pt"
            self._model = _YOLOWorld(name)
            self._model.set_classes(self._vocab)
        else:
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
