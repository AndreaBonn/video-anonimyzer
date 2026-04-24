"""
Test aggiuntivi per anonymization.py — coverage righe mancanti.

Copre: obscure_polygon (righe 96-97: w==0 o h==0 dopo clamping),
box_to_polygon (righe 145-147: frame_w/frame_h None o falsy).
"""

import numpy as np
import pytest

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from person_anonymizer.anonymization import box_to_polygon, obscure_polygon
from person_anonymizer.config import PipelineConfig


# ============================================================
# obscure_polygon — righe 96-97 (w == 0 o h == 0)
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestObscurePolygonEdgeCases:
    """Verifica i casi in cui il bounding rect ha larghezza o altezza zero."""

    def test_zero_width_polygon_returns_unchanged_frame(self):
        # Arrange — poligono con punti x tutti uguali → w = 0 dopo boundingRect
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        original = frame.copy()
        # Tutti i punti sulla stessa colonna x=50 → w=0
        points = [(50, 10), (50, 50), (50, 80)]

        # Act
        result = obscure_polygon(frame, points, "pixelation", 10)

        # Assert — frame non modificato
        assert np.array_equal(result, original)
        assert result is frame

    def test_zero_height_polygon_returns_unchanged_frame(self):
        # Arrange — poligono con punti y tutti uguali → h = 0
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 64
        original = frame.copy()
        # Tutti i punti sulla stessa riga y=30 → h=0
        points = [(10, 30), (50, 30), (80, 30)]

        # Act
        result = obscure_polygon(frame, points, "blur", 11)

        # Assert — frame non modificato
        assert np.array_equal(result, original)
        assert result is frame

    def test_single_point_polygon_returns_unchanged(self):
        # Arrange — un singolo punto → w=0, h=0
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 200
        original = frame.copy()
        points = [(25, 25)]

        # Act
        result = obscure_polygon(frame, points, "pixelation", 5)

        # Assert
        assert np.array_equal(result, original)

    def test_polygon_at_frame_edge_clamped_still_processed(self):
        # Arrange — poligono valido anche se a bordo frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        # Poligono nell'angolo in basso a destra
        points = [(90, 90), (100, 90), (100, 100), (90, 100)]

        # Act
        result = obscure_polygon(frame, points, "pixelation", 5)

        # Assert — il frame è stato modificato (o identico se area = 0 dopo clamp)
        assert result is frame  # sempre stesso oggetto


# ============================================================
# box_to_polygon — righe 145-147 (frame_w=0, frame_h=0 o None)
# ============================================================


class TestBoxToPolygonEdgeCases:
    """Verifica box_to_polygon con frame_w/frame_h falsy."""

    def test_frame_w_zero_clamps_x2(self):
        # frame_w=0 falsy → pad uniforme, poi x2 = min(55, 0) = 0
        # Poligono: [(5,5), (0,5), (0,55), (5,55)]
        x1, y1, x2, y2 = 10, 10, 50, 50
        result = box_to_polygon(x1, y1, x2, y2, padding=5, frame_w=0, frame_h=100)
        # x2 clamped a 0, x1 padded a 5
        assert result[0][0] == 5   # x1 padded
        assert result[1][0] == 0   # x2 clamped

    def test_frame_h_zero_clamps_y2(self):
        # frame_h=0 falsy → pad uniforme, poi y2 = min(55, 0) = 0
        x1, y1, x2, y2 = 10, 10, 50, 50
        result = box_to_polygon(x1, y1, x2, y2, padding=5, frame_w=100, frame_h=0)
        assert result[0][1] == 5   # y1 padded
        assert result[2][1] == 0   # y2 clamped

    def test_both_frame_dimensions_none_uniform_padding(self):
        # Arrange — frame_w=None, frame_h=None → padding uniforme
        x1, y1, x2, y2 = 20, 20, 60, 60
        padding = 8

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=padding, frame_w=None, frame_h=None)

        # Assert — padding uniforme in tutte le direzioni
        xs = [p[0] for p in result]
        ys = [p[1] for p in result]
        assert min(xs) == max(0, x1 - padding)
        assert max(xs) == x2 + padding
        assert min(ys) == max(0, y1 - padding)
        assert max(ys) == y2 + padding

    def test_x2_not_clamped_when_frame_w_none(self):
        # Arrange — frame_w=None → x2 non viene clamped
        x1, y1, x2, y2 = 80, 10, 110, 50  # x2 > frame_w ipotetico
        padding = 0

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=0, frame_w=None, frame_h=None)

        # Assert — x2 = 110 (non clamped)
        xs = [p[0] for p in result]
        assert max(xs) == 110

    def test_y2_not_clamped_when_frame_h_none(self):
        # Arrange — frame_h=None → y2 non viene clamped
        x1, y1, x2, y2 = 10, 80, 50, 120  # y2 > frame_h ipotetico
        padding = 0

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=0, frame_w=None, frame_h=None)

        # Assert — y2 = 120 (non clamped)
        ys = [p[1] for p in result]
        assert max(ys) == 120

    def test_x2_clamped_to_frame_w_when_provided(self):
        # Arrange — x2+padding > frame_w → clamped
        frame_w, frame_h = 100, 100
        x1, y1, x2, y2 = 10, 10, 95, 50
        padding = 10  # x2+padding = 105 > 100

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=padding, frame_w=frame_w, frame_h=frame_h)

        # Assert — x2 clamped a 100
        xs = [p[0] for p in result]
        assert max(xs) <= frame_w

    def test_y2_clamped_to_frame_h_when_provided(self):
        # Arrange — y2+padding > frame_h → clamped
        frame_w, frame_h = 100, 100
        x1, y1, x2, y2 = 10, 10, 50, 95
        padding = 10  # y2+padding = 105 > 100

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=padding, frame_w=frame_w, frame_h=frame_h)

        # Assert — y2 clamped a 100
        ys = [p[1] for p in result]
        assert max(ys) <= frame_h
