"""
Test aggiuntivi per detection.py — coverage righe mancanti.

Copre: get_window_patches (grid <= 0), run_sliding_window (con/senza motion),
detect_and_rescale, run_multiscale_inference (TTA flip), run_full_detection.
Mock: ultralytics YOLO model, cv2.dnn.NMSBoxes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.detection import (
    detect_and_rescale,
    get_window_patches,
    run_full_detection,
    run_multiscale_inference,
    run_sliding_window,
)


# ============================================================
# Helpers
# ============================================================


def _make_fake_model(boxes_xyxy=None, conf=None):
    """Restituisce un mock YOLO model con output preconfigurato."""
    if boxes_xyxy is None:
        boxes_xyxy = []
    if conf is None:
        conf = [0.9] * len(boxes_xyxy)

    model = MagicMock()
    result = MagicMock()
    result_list = MagicMock()
    result_list.__iter__ = MagicMock(return_value=iter([]))
    result.__getitem__ = MagicMock(return_value=result_list)

    boxes_mock = MagicMock()
    fake_boxes = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        box = MagicMock()
        box.xyxy = [np.array([x1, y1, x2, y2])]
        box.conf = [conf[i]]
        fake_boxes.append(box)

    result_list.boxes = fake_boxes
    model.return_value = [result_list]
    return model


# ============================================================
# get_window_patches — grid <= 0 (riga 27)
# ============================================================


class TestGetWindowPatchesEdgeCases:
    """Verifica i casi limite di get_window_patches."""

    def test_grid_zero_raises_value_error(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="grid deve essere > 0"):
            get_window_patches(frame_w=640, frame_h=480, grid=0, overlap=0.0)

    def test_grid_negative_raises_value_error(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="grid deve essere > 0"):
            get_window_patches(frame_w=640, frame_h=480, grid=-1, overlap=0.0)

    def test_grid_1_returns_single_patch(self):
        # Arrange
        patches = get_window_patches(frame_w=320, frame_h=240, grid=1, overlap=0.0)
        # Assert — una sola patch che copre l'intero frame (o subset)
        assert len(patches) == 1


# ============================================================
# run_sliding_window — righe 53-66
# ============================================================


class TestRunSlidingWindow:
    """Verifica run_sliding_window con e senza motion_regions."""

    def test_no_motion_regions_processes_all_patches(self):
        # Arrange — 4 patch, nessuna motion region
        model = _make_fake_model(boxes_xyxy=[(10, 10, 50, 50)], conf=[0.8])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patches = [(0, 0, 50, 50), (50, 0, 100, 50), (0, 50, 50, 100), (50, 50, 100, 100)]

        # Act
        all_boxes, hits = run_sliding_window(model, frame, patches, conf=0.5, motion_regions=None)

        # Assert — ogni patch viene elaborata, ogni box viene rilevato
        assert len(all_boxes) == 4  # 1 box per patch
        assert hits == 4

    def test_empty_motion_regions_processes_all_patches(self):
        # Arrange — lista vuota equivale a nessun filtro
        model = _make_fake_model(boxes_xyxy=[(5, 5, 20, 20)], conf=[0.7])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patches = [(0, 0, 50, 50), (50, 50, 100, 100)]

        # Act
        all_boxes, hits = run_sliding_window(model, frame, patches, conf=0.5, motion_regions=[])

        # Assert — lista vuota → len==0 → filtro saltato → tutte le patch processate
        assert len(all_boxes) == 2  # una detection per ogni patch
        assert hits == 2

    def test_motion_regions_filters_patches(self):
        # Arrange — solo la prima patch interseca la motion region
        model = _make_fake_model(boxes_xyxy=[(5, 5, 20, 20)], conf=[0.9])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        patches = [
            (0, 0, 50, 50),      # interseca motion region (25,25)-(75,75)
            (100, 100, 200, 200),  # non interseca
        ]
        motion_regions = [(25, 25, 75, 75)]

        # Act
        all_boxes, hits = run_sliding_window(model, frame, patches, conf=0.5, motion_regions=motion_regions)

        # Assert — solo prima patch processata
        assert len(all_boxes) == 1
        assert hits == 1

    def test_boxes_offset_by_patch_position(self):
        # Arrange — modello rileva box a (5, 5, 20, 20) dentro patch (100, 100, 200, 200)
        # → coordinate assolute devono essere (105, 105, 120, 120)
        model = _make_fake_model(boxes_xyxy=[(5, 5, 20, 20)], conf=[0.85])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        patches = [(100, 100, 200, 200)]

        # Act
        all_boxes, hits = run_sliding_window(model, frame, patches, conf=0.5, motion_regions=None)

        # Assert
        assert len(all_boxes) == 1
        x1, y1, x2, y2, score = all_boxes[0]
        assert x1 == pytest.approx(105.0)
        assert y1 == pytest.approx(105.0)
        assert x2 == pytest.approx(120.0)
        assert y2 == pytest.approx(120.0)

    def test_no_detections_returns_empty_and_zero_hits(self):
        # Arrange — modello non rileva nulla
        model = _make_fake_model(boxes_xyxy=[], conf=[])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patches = [(0, 0, 50, 50)]

        # Act
        all_boxes, hits = run_sliding_window(model, frame, patches, conf=0.5, motion_regions=None)

        # Assert
        assert all_boxes == []
        assert hits == 0


# ============================================================
# detect_and_rescale — righe 69-83
# ============================================================


class TestDetectAndRescale:
    """Verifica detect_and_rescale: imgsz adattivo e riscalatura coordinate."""

    def test_scale_1_returns_coordinates_unchanged(self):
        # Arrange — scale=1.0, base_imgsz=640 → effective_imgsz=640
        model = _make_fake_model(boxes_xyxy=[(10, 20, 50, 80)], conf=[0.9])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        boxes = detect_and_rescale(model, frame, conf=0.5, scale=1.0)

        # Assert — x/scale = x/1.0 → invariato
        assert len(boxes) == 1
        x1, y1, x2, y2, score = boxes[0]
        assert x1 == pytest.approx(10.0)
        assert y1 == pytest.approx(20.0)
        assert x2 == pytest.approx(50.0)
        assert y2 == pytest.approx(80.0)

    def test_scale_2_divides_coordinates_by_2(self):
        # Arrange — scale=2.0 → le coordinate sono divise per 2
        model = _make_fake_model(boxes_xyxy=[(20, 40, 100, 160)], conf=[0.8])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Act
        boxes = detect_and_rescale(model, frame, conf=0.5, scale=2.0)

        # Assert
        assert len(boxes) == 1
        x1, y1, x2, y2, _ = boxes[0]
        assert x1 == pytest.approx(10.0)
        assert y1 == pytest.approx(20.0)
        assert x2 == pytest.approx(50.0)
        assert y2 == pytest.approx(80.0)

    def test_no_detections_returns_empty(self):
        # Arrange
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        boxes = detect_and_rescale(model, frame, conf=0.5, scale=1.0)

        # Assert
        assert boxes == []

    def test_high_scale_caps_imgsz_at_1280(self):
        # Arrange — scale=3.0 → effective_imgsz = min(640*3, 1280) = 1280
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act — verifica che venga chiamato con imgsz corretto
        detect_and_rescale(model, frame, conf=0.5, scale=3.0, base_imgsz=640)

        # Assert — il modello viene chiamato con imgsz=1280 (capped)
        call_kwargs = model.call_args[1]
        assert call_kwargs["imgsz"] == 1280

    def test_scale_less_than_1_uses_base_imgsz(self):
        # Arrange — scale=0.5 → max(1.0, 0.5)=1.0 → effective_imgsz=640
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        detect_and_rescale(model, frame, conf=0.5, scale=0.5, base_imgsz=640)

        # Assert
        call_kwargs = model.call_args[1]
        assert call_kwargs["imgsz"] == 640


# ============================================================
# run_multiscale_inference — righe 86-109
# ============================================================


class TestRunMultiscaleInference:
    """Verifica run_multiscale_inference con TTA flip."""

    def test_single_scale_no_augmentation(self):
        # Arrange
        model = _make_fake_model(boxes_xyxy=[(10, 10, 50, 50)], conf=[0.9])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_CUBIC", 2), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, hits = run_multiscale_inference(
                model, frame, scales=[1.0], augmentations=[], conf=0.5,
                orig_w=100, orig_h=100
            )

        # Assert — 1 box, 1 hit
        assert len(all_boxes) == 1
        assert hits == 1

    def test_flip_h_augmentation_mirrors_coordinates(self):
        # Arrange — box a (10, 10, 50, 50) in frame 100x100 con scale=1.0
        # Dopo flip_h: x1_new = orig_w - x2 = 100 - 50 = 50
        #              x2_new = orig_w - x1 = 100 - 10 = 90
        model = _make_fake_model(boxes_xyxy=[(10, 10, 50, 50)], conf=[0.9])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.flip", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_CUBIC", 2), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, hits = run_multiscale_inference(
                model, frame, scales=[1.0], augmentations=["flip_h"], conf=0.5,
                orig_w=100, orig_h=100
            )

        # Assert — 2 hit (scala + flip), 2 box
        assert hits == 2
        assert len(all_boxes) == 2
        # Verifica che le coordinate del flip siano invertite
        flip_box = all_boxes[1]
        assert flip_box[0] == pytest.approx(50.0)  # 100 - 50
        assert flip_box[2] == pytest.approx(90.0)  # 100 - 10

    def test_multiple_scales_returns_all_boxes(self):
        # Arrange — modello rileva 1 box per chiamata
        model = _make_fake_model(boxes_xyxy=[(5, 5, 30, 30)], conf=[0.7])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_CUBIC", 2), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, hits = run_multiscale_inference(
                model, frame, scales=[0.5, 1.0, 2.0], augmentations=[], conf=0.5,
                orig_w=100, orig_h=100
            )

        # Assert — 3 scale × 1 box = 3 box, 3 hits
        assert len(all_boxes) == 3
        assert hits == 3

    def test_no_detections_returns_empty(self):
        # Arrange
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, hits = run_multiscale_inference(
                model, frame, scales=[1.0], augmentations=[], conf=0.5,
                orig_w=100, orig_h=100
            )

        # Assert
        assert all_boxes == []
        assert hits == 0


# ============================================================
# run_full_detection — righe 187-204
# ============================================================


class TestRunFullDetection:
    """Verifica run_full_detection — orchestrazione sliding window + multi-scale."""

    def _make_config(self, sliding_window=True):
        return PipelineConfig(
            enable_sliding_window=sliding_window,
            inference_scales=[1.0],
            tta_augmentations=[],
            nms_iou_internal=0.5,
            nms_iou_threshold=0.55,
        )

    def test_with_sliding_window_disabled(self):
        # Arrange — sliding window disabilitata
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        config = self._make_config(sliding_window=False)
        patches = [(0, 0, 50, 50)]

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, sw_hits, ms_hits = run_full_detection(
                model, frame, conf=0.5,
                frame_w=100, frame_h=100,
                motion_regions=None,
                patches=patches,
                config=config,
            )

        # Assert — sw_hits=0 (non eseguito), ms_hits=0 (nessuna detection)
        assert sw_hits == 0

    def test_with_sliding_window_enabled_no_patches(self):
        # Arrange — patches vuota → sliding window non eseguita
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        config = self._make_config(sliding_window=True)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            all_boxes, sw_hits, ms_hits = run_full_detection(
                model, frame, conf=0.5,
                frame_w=100, frame_h=100,
                motion_regions=None,
                patches=[],
                config=config,
            )

        # Assert — patches vuota → sliding window saltata
        assert sw_hits == 0

    def test_returns_tuple_of_three(self):
        # Arrange
        model = _make_fake_model()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        config = self._make_config(sliding_window=False)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1):
            result = run_full_detection(
                model, frame, conf=0.5,
                frame_w=100, frame_h=100,
                motion_regions=None,
                patches=[],
                config=config,
            )

        # Assert
        assert len(result) == 3

    def test_with_detections_returns_boxes(self):
        # Arrange — modello rileva 1 persona
        model = _make_fake_model(boxes_xyxy=[(10, 10, 50, 50)], conf=[0.9])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        config = self._make_config(sliding_window=False)

        with patch("person_anonymizer.detection.cv2.resize", return_value=frame), \
             patch("person_anonymizer.detection.cv2.INTER_LINEAR", 1), \
             patch("person_anonymizer.detection.cv2.dnn") as mock_dnn:
            mock_dnn.NMSBoxes.return_value = np.array([[0]])
            all_boxes, sw_hits, ms_hits = run_full_detection(
                model, frame, conf=0.5,
                frame_w=100, frame_h=100,
                motion_regions=None,
                patches=[],
                config=config,
            )

        # Assert — almeno 1 box rilevato
        assert len(all_boxes) >= 1
        assert ms_hits == 1
