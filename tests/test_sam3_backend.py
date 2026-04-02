"""Test per sam3_backend — funziona senza SAM3 installato (usa mock)."""

import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from person_anonymizer.sam3_backend import (  # noqa: I001
    Sam3ImageRefiner,
    Sam3VideoDetector,
    check_sam3_available,
    mask_to_polygons,
)

# ─── check_sam3_available ───────────────────────────────────────


class TestCheckSam3Available:
    def test_returns_false_when_module_missing(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert check_sam3_available() is False

    def test_returns_false_when_python_too_old(self):
        with patch.object(sys, "version_info", (3, 11, 0)):
            assert check_sam3_available() is False

    def test_returns_true_when_available(self):
        mock_spec = MagicMock()
        with (
            patch("importlib.util.find_spec", return_value=mock_spec),
            patch.object(sys, "version_info", (3, 12, 0)),
        ):
            assert check_sam3_available() is True


# ─── mask_to_polygons ──────────────────────────────────────────


class TestMaskToPolygons:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = mask_to_polygons(mask)
        assert result == []

    def test_small_mask_filtered_by_min_area(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:12, 10:12] = 255  # area = ~4 px, sotto min_area=100
        result = mask_to_polygons(mask, min_area=100)
        assert result == []

    def test_circle_mask_produces_polygon(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        result = mask_to_polygons(mask, min_area=100)
        assert len(result) == 1
        assert len(result[0]) >= 3  # almeno un triangolo

    def test_rectangle_mask_produces_4_points(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        result = mask_to_polygons(mask, epsilon_ratio=0.02, min_area=100)
        assert len(result) == 1
        # Con epsilon alto un rettangolo dà ~4 punti
        assert len(result[0]) >= 4

    def test_binary_01_mask_normalized(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 1  # valori 0/1 invece di 0/255
        result = mask_to_polygons(mask, min_area=100)
        assert len(result) == 1

    def test_multiple_blobs_produce_multiple_polygons(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (75, 75), 40, 255, -1)
        cv2.circle(mask, (225, 225), 40, 255, -1)
        result = mask_to_polygons(mask, min_area=100)
        assert len(result) == 2

    def test_epsilon_ratio_affects_simplification(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, -1)
        detailed = mask_to_polygons(mask, epsilon_ratio=0.001, min_area=100)
        simple = mask_to_polygons(mask, epsilon_ratio=0.05, min_area=100)
        assert len(detailed[0]) >= len(simple[0])


# ─── Sam3ImageRefiner ──────────────────────────────────────────


class TestSam3ImageRefiner:
    def test_init_defaults(self):
        refiner = Sam3ImageRefiner()
        assert refiner.model_path == "sam3_hiera_large.pt"
        assert refiner.device == "cuda"
        assert refiner._predictor is None

    @pytest.mark.skipif(
        check_sam3_available(), reason="SAM3 è installato, test non applicabile"
    )
    def test_refine_boxes_raises_without_sam3(self):
        refiner = Sam3ImageRefiner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ImportError, match="SAM3"):
            refiner.refine_boxes(frame, [(10, 10, 50, 50)])

    def test_release_noop_when_no_predictor(self):
        refiner = Sam3ImageRefiner()
        refiner.release()  # non deve sollevare errori


# ─── Sam3VideoDetector ─────────────────────────────────────────


class TestSam3VideoDetector:
    def test_init_defaults(self):
        detector = Sam3VideoDetector()
        assert detector.text_prompt == "person"
        assert detector.device == "cuda"

    @pytest.mark.skipif(
        check_sam3_available(), reason="SAM3 è installato, test non applicabile"
    )
    def test_detect_video_raises_without_sam3(self):
        detector = Sam3VideoDetector()
        config = MagicMock()
        with pytest.raises(ImportError, match="SAM3"):
            detector.detect_video("fake.mp4", config)
