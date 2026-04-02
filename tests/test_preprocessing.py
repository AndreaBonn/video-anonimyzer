"""Test per il modulo preprocessing."""

import numpy as np
import pytest

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from person_anonymizer.preprocessing import interpolate_frames, should_interpolate

if CV2_AVAILABLE:
    from person_anonymizer.preprocessing import MotionDetector, enhance_frame


class TestShouldInterpolate:
    """Test per should_interpolate — funzione pura."""

    def test_low_fps_returns_true(self):
        # Arrange
        fps = 10
        threshold = 15
        # Act
        result = should_interpolate(fps, threshold)
        # Assert
        assert result is True

    def test_high_fps_returns_false(self):
        # Arrange
        fps = 30
        threshold = 15
        # Act
        result = should_interpolate(fps, threshold)
        # Assert
        assert result is False

    def test_equal_fps_returns_false(self):
        # Arrange
        fps = 15
        threshold = 15
        # Act
        result = should_interpolate(fps, threshold)
        # Assert
        assert result is False

    def test_zero_fps_returns_true(self):
        # Arrange / Act / Assert
        assert should_interpolate(0, 15) is True


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestInterpolateFrames:
    """Test per interpolate_frames — richiede cv2 (addWeighted)."""

    def test_single_step_returns_one_frame(self):
        # Arrange
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Act
        result = interpolate_frames(frame_a, frame_b, 1)
        # Assert
        assert len(result) == 1
        assert result[0].shape == (100, 100, 3)

    def test_multiple_steps(self):
        # Arrange
        frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_b = np.ones((50, 50, 3), dtype=np.uint8) * 200
        # Act
        result = interpolate_frames(frame_a, frame_b, 3)
        # Assert
        assert len(result) == 3

    def test_interpolated_values_increase_monotonically(self):
        # Arrange
        frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_b = np.ones((10, 10, 3), dtype=np.uint8) * 255
        # Act
        result = interpolate_frames(frame_a, frame_b, 3)
        # Assert — ogni frame interpolato dovrebbe avere media crescente
        means = [f.mean() for f in result]
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1]


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestEnhanceFrame:
    """Test per enhance_frame — CLAHE condizionale."""

    def test_dark_frame_is_enhanced(self):
        # Arrange
        dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 30
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Act
        result = enhance_frame(dark_frame, clahe, darkness_threshold=60)
        # Assert — il frame deve essere diverso (CLAHE applicata)
        assert not np.array_equal(result, dark_frame)

    def test_bright_frame_unchanged(self):
        # Arrange
        bright_frame = np.ones((100, 100, 3), dtype=np.uint8) * 180
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Act
        result = enhance_frame(bright_frame, clahe, darkness_threshold=60)
        # Assert — il frame deve restare identico
        assert np.array_equal(result, bright_frame)

    def test_returns_same_shape(self):
        # Arrange
        frame = np.random.randint(0, 50, (200, 300, 3), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Act
        result = enhance_frame(frame, clahe, darkness_threshold=60)
        # Assert
        assert result.shape == frame.shape


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestMotionDetector:
    """Test per MotionDetector — frame differencing."""

    def test_first_frame_returns_none(self):
        # Arrange
        md = MotionDetector(threshold=25, min_area=500, padding=60)
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        # Act
        result = md.get_motion_regions(frame)
        # Assert
        assert result is None

    def test_identical_frames_return_empty(self):
        # Arrange
        md = MotionDetector(threshold=25, min_area=500, padding=60)
        frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
        md.get_motion_regions(frame)  # primo frame
        # Act
        result = md.get_motion_regions(frame.copy())
        # Assert
        assert result == []

    def test_different_frames_return_regions(self):
        # Arrange
        md = MotionDetector(threshold=25, min_area=100, padding=10)
        frame1 = np.zeros((200, 300, 3), dtype=np.uint8)
        frame2 = np.zeros((200, 300, 3), dtype=np.uint8)
        # Disegna un rettangolo bianco grande nel secondo frame
        frame2[50:150, 50:250] = 255
        md.get_motion_regions(frame1)  # primo frame
        # Act
        result = md.get_motion_regions(frame2)
        # Assert
        assert len(result) > 0
        # Ogni regione è una tupla (x1, y1, x2, y2)
        for region in result:
            assert len(region) == 4
            x1, y1, x2, y2 = region
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 300 and y2 <= 200

    def test_padding_clamped_to_frame_bounds(self):
        # Arrange — motion vicino ai bordi
        md = MotionDetector(threshold=25, min_area=100, padding=100)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[0:30, 0:30] = 255  # angolo in alto a sinistra
        md.get_motion_regions(frame1)
        # Act
        result = md.get_motion_regions(frame2)
        # Assert — il padding non deve superare i bordi del frame
        if result:
            for x1, y1, x2, y2 in result:
                assert x1 >= 0
                assert y1 >= 0
                assert x2 <= 100
                assert y2 <= 100
