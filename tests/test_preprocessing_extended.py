"""
Test aggiuntivi per preprocessing.py — coverage righe mancanti.

Copre: build_undistortion_maps (righe 23, 26, 29, 34),
undistort_frame (riga 34), MotionDetector.get_motion_regions edge cases (riga 108).
"""

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

from person_anonymizer.preprocessing import (
    MotionDetector,
    build_undistortion_maps,
    undistort_frame,
)

# ============================================================
# build_undistortion_maps — righe 23, 26, 29
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestBuildUndistortionMaps:
    """Verifica build_undistortion_maps — costruisce le mappe di undistortion."""

    def test_returns_two_maps(self):
        # Arrange — camera matrix e distorsione identity (nessuna distorsione)
        camera_matrix = np.eye(3, dtype=np.float64)
        camera_matrix[0, 0] = 500.0  # fx
        camera_matrix[1, 1] = 500.0  # fy
        camera_matrix[0, 2] = 160.0  # cx
        camera_matrix[1, 2] = 120.0  # cy
        dist_coefficients = np.zeros((4, 1), dtype=np.float64)
        frame_w, frame_h = 320, 240

        # Act
        map1, map2 = build_undistortion_maps(camera_matrix, dist_coefficients, frame_w, frame_h)

        # Assert — restituisce due array
        assert map1 is not None
        assert map2 is not None
        assert isinstance(map1, np.ndarray)
        assert isinstance(map2, np.ndarray)

    def test_maps_have_correct_shape(self):
        # Arrange
        camera_matrix = np.array([
            [400.0, 0.0, 200.0],
            [0.0, 400.0, 150.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coefficients = np.zeros((5, 1), dtype=np.float64)
        frame_w, frame_h = 400, 300

        # Act
        map1, map2 = build_undistortion_maps(camera_matrix, dist_coefficients, frame_w, frame_h)

        # Assert — le mappe hanno le dimensioni del frame
        assert map1.shape[:2] == (frame_h, frame_w)

    def test_maps_with_distortion_differ_from_identity(self):
        # Arrange — distorsione radiale significativa
        camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_no_distortion = np.zeros((5, 1), dtype=np.float64)
        dist_with_distortion = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float64).reshape(5, 1)
        frame_w, frame_h = 640, 480

        # Act
        map1_no, map2_no = build_undistortion_maps(
            camera_matrix, dist_no_distortion, frame_w, frame_h
        )
        map1_yes, map2_yes = build_undistortion_maps(
            camera_matrix, dist_with_distortion, frame_w, frame_h
        )

        # Assert — le mappe con distorsione sono diverse da quelle senza
        assert not np.array_equal(map1_no, map1_yes)


# ============================================================
# undistort_frame — riga 34
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestUndistortFrame:
    """Verifica undistort_frame."""

    def test_returns_array_same_shape(self):
        # Arrange
        camera_matrix = np.array([
            [400.0, 0.0, 160.0],
            [0.0, 400.0, 120.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coefficients = np.zeros((5, 1), dtype=np.float64)
        frame_w, frame_h = 320, 240
        map1, map2 = build_undistortion_maps(camera_matrix, dist_coefficients, frame_w, frame_h)

        frame = np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)

        # Act
        result = undistort_frame(frame, map1, map2)

        # Assert — shape preservata
        assert result.shape == frame.shape

    def test_identity_maps_return_identical_frame(self):
        # Arrange — mappe identity (nessuna distorsione)
        camera_matrix = np.array([
            [500.0, 0.0, 100.0],
            [0.0, 500.0, 100.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coefficients = np.zeros((5, 1), dtype=np.float64)
        frame_w, frame_h = 200, 200
        map1, map2 = build_undistortion_maps(camera_matrix, dist_coefficients, frame_w, frame_h)

        frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 128

        # Act
        result = undistort_frame(frame, map1, map2)

        # Assert — il risultato è un ndarray
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_uses_cv2_remap(self):
        # Arrange — verifica che venga chiamato cv2.remap internamente
        map1 = np.zeros((100, 100), dtype=np.float32)
        map2 = np.zeros((100, 100), dtype=np.float32)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        expected = np.ones((100, 100, 3), dtype=np.uint8) * 42

        with patch(
            "person_anonymizer.preprocessing.cv2.remap", return_value=expected
        ) as mock_remap:
            # Act
            result = undistort_frame(frame, map1, map2)

        # Assert — cv2.remap chiamato con i parametri corretti
        assert mock_remap.called
        call_args = mock_remap.call_args[0]
        assert call_args[0] is frame
        assert call_args[1] is map1
        assert call_args[2] is map2
        assert result is expected


# ============================================================
# MotionDetector — edge cases riga 108 (contourArea < min_area)
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestMotionDetectorExtended:
    """Verifica MotionDetector — filtering per area minima."""

    def test_small_contours_filtered_by_min_area(self):
        # Arrange — frame con piccolo movimento (1x1 pixel) che non supera min_area
        md = MotionDetector(threshold=10, min_area=5000, padding=0)
        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Cambia solo 1 pixel — area del contorno < min_area
        frame2[100, 100] = [255, 255, 255]

        md.get_motion_regions(frame1)  # inizializza prev_gray

        # Act
        result = md.get_motion_regions(frame2)

        # Assert — nessuna regione (troppo piccola)
        assert result == []

    def test_large_contours_above_min_area_detected(self):
        # Arrange — blocco 50x50 di movimento certamente > min_area=100
        md = MotionDetector(threshold=10, min_area=100, padding=0)
        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[50:100, 50:100] = 200  # 50x50 = 2500 pixel

        md.get_motion_regions(frame1)

        # Act
        result = md.get_motion_regions(frame2)

        # Assert — almeno una regione rilevata
        assert len(result) >= 1

    def test_region_format_x1_y1_x2_y2(self):
        # Arrange
        md = MotionDetector(threshold=10, min_area=50, padding=5)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[20:50, 20:50] = 200

        md.get_motion_regions(frame1)

        # Act
        result = md.get_motion_regions(frame2)

        # Assert — ogni regione è (x1, y1, x2, y2)
        for region in result:
            assert len(region) == 4
            x1, y1, x2, y2 = region
            assert x1 < x2
            assert y1 < y2
