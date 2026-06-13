from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.object_detection.detector import ModelNotLoadedError, ObjectDetection, YOLODetector


def _make_box(x1=10, y1=20, x2=100, y2=200, cls_id=0, conf=0.9):
    """Build a mock ultralytics box."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value.numpy.return_value = np.array(
        [x1, y1, x2, y2], dtype="float32"
    )
    return box


def _make_result(boxes=None, names=None):
    result = MagicMock()
    result.boxes = boxes if boxes is not None else []
    result.names = names or {0: "person", 1: "bicycle"}
    return result


def _make_yolo_model(results=None):
    model = MagicMock()
    model.return_value = results or []
    return model


# ── dataclass ────────────────────────────────────────────────────────────────

def test_object_detection_dataclass():
    bbox = np.array([0, 0, 50, 50], dtype="float32")
    d = ObjectDetection(bbox=bbox, class_id=0, class_name="person", confidence=0.9)
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.confidence == pytest.approx(0.9)


# ── ModelNotLoadedError before load ─────────────────────────────────────────

def test_detect_raises_before_load():
    detector = YOLODetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ModelNotLoadedError):
        detector.detect(frame)


# ── load raises when ultralytics absent ─────────────────────────────────────

def test_load_raises_when_ultralytics_unavailable():
    detector = YOLODetector()
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", False):
        with pytest.raises(ModelNotLoadedError):
            detector.load()


# ── is_loaded flag ───────────────────────────────────────────────────────────

def test_is_loaded_false_before_load():
    assert YOLODetector().is_loaded is False


def test_is_loaded_true_after_load():
    detector = YOLODetector()
    mock_model = _make_yolo_model()
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLO", return_value=mock_model):
        detector.load()
    assert detector.is_loaded is True


# ── detection results ────────────────────────────────────────────────────────

def test_detect_returns_empty_when_no_boxes():
    detector = YOLODetector()
    mock_model = _make_yolo_model([_make_result(boxes=[])])
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLO", return_value=mock_model):
        detector.load()
    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result == []


def test_detect_returns_single_detection():
    detector = YOLODetector(conf_threshold=0.25)
    box = _make_box(conf=0.9, cls_id=0)
    mock_model = _make_yolo_model([_make_result(boxes=[box])])
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLO", return_value=mock_model):
        detector.load()
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].confidence == pytest.approx(0.9)


# ── class filter ─────────────────────────────────────────────────────────────

def test_class_filter_excludes_unwanted():
    detector = YOLODetector(class_filter=[1])   # only bicycle
    box = _make_box(cls_id=0, conf=0.9)         # person — should be excluded
    mock_model = _make_yolo_model([_make_result(boxes=[box])])
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLO", return_value=mock_model):
        detector.load()
    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result == []


def test_class_filter_includes_wanted():
    detector = YOLODetector(class_filter=[0])   # only person
    boxes = [_make_box(cls_id=0, conf=0.9), _make_box(cls_id=1, conf=0.8)]
    mock_model = _make_yolo_model([_make_result(boxes=boxes)])
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLO", return_value=mock_model):
        detector.load()
    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0].class_id == 0


# ── open-vocabulary mode (YOLO-World) ────────────────────────────────────────

def test_open_vocab_loads_yoloworld_and_sets_classes():
    vocab = ["person", "almirah", "ipad"]
    mock_model = _make_yolo_model()
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLOWorld",
               return_value=mock_model) as world, \
         patch("src.object_detection.detector._YOLO") as plain:
        detector = YOLODetector(vocab=vocab)
        assert detector.is_open_vocab
        detector.load()
        # YOLO-World path, not the closed-set YOLO path
        world.assert_called_once()
        plain.assert_not_called()
        mock_model.set_classes.assert_called_once_with(vocab)


def test_open_vocab_detects_prompted_classes():
    vocab = ["person", "almirah"]
    boxes = [_make_box(cls_id=1, conf=0.8)]   # index 1 -> "almirah"
    result = _make_result(boxes=boxes, names={0: "person", 1: "almirah"})
    mock_model = _make_yolo_model([result])
    with patch("src.object_detection.detector._ULTRALYTICS_AVAILABLE", True), \
         patch("src.object_detection.detector._YOLOWorld", return_value=mock_model):
        detector = YOLODetector(vocab=vocab)
        detector.load()
        dets = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].class_name == "almirah"


def test_closed_set_default_is_not_open_vocab():
    assert YOLODetector().is_open_vocab is False
