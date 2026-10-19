"""
Test aggiuntivi per anonymization.py — coverage righe mancanti.

Copre: obscure_polygon su poligoni degeneri e ai bordi del frame,
box_to_polygon con frame_w/frame_h None o falsy.
"""

import numpy as np
import pytest

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from person_anonymizer.anonymization import (
    box_to_polygon,
    compute_adaptive_intensity,
    obscure_polygon,
)
from person_anonymizer.config import PipelineConfig

# ============================================================
# obscure_polygon — poligoni degeneri e bordi del frame
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestObscurePolygonEdgeCases:
    """Poligoni degeneri (linee, punti) e poligoni sul bordo del frame."""

    @pytest.mark.parametrize("axis", ["vertical", "horizontal"])
    def test_obscure_polygon_line_changes_inside_preserves_outside(self, axis: str) -> None:
        # Arrange
        frame = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        inside = np.zeros((100, 100), dtype=bool)
        if axis == "vertical":
            points = [(50, 10), (50, 50), (50, 80)]
            inside[10:81, 50] = True
        else:
            points = [(10, 30), (50, 30), (80, 30)]
            inside[30, 10:81] = True
        # Act
        result = obscure_polygon(frame=frame, points=points, method="pixelation", intensity=10)
        # Assert
        assert result is frame
        assert np.any(result[inside] != original[inside])
        np.testing.assert_array_equal(result[~inside], original[~inside])

    def test_obscure_polygon_single_point_preserves_pixel_and_obscures_control(self) -> None:
        # Arrange
        frame = np.random.default_rng(0).integers(0, 256, (50, 50, 3), dtype=np.uint8)
        original = frame.copy()
        # Act
        result = obscure_polygon(
            frame=frame,
            points=[(25, 25)],
            method="pixelation",
            intensity=5,
        )
        control = obscure_polygon(
            frame=original.copy(),
            points=[(20, 20), (29, 20), (29, 29), (20, 29)],
            method="pixelation",
            intensity=5,
        )
        # Assert
        np.testing.assert_array_equal(result, original)
        assert np.any(control[20:30, 20:30] != original[20:30, 20:30])

    @pytest.mark.parametrize("edge", ["right", "bottom", "corner"])
    def test_obscure_polygon_frame_edge_obscures_last_pixels(self, edge: str) -> None:
        # Arrange
        frame = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        inside = np.zeros((100, 100), dtype=bool)
        bounds = {
            "right": (90, 40, 100, 59),
            "bottom": (40, 90, 59, 100),
            "corner": (90, 90, 100, 100),
        }
        x1, y1, x2, y2 = bounds[edge]
        inside[y1 : min(y2 + 1, 100), x1 : min(x2 + 1, 100)] = True
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        # Act
        result = obscure_polygon(frame=frame, points=points, method="pixelation", intensity=5)
        # Assert
        assert result is frame
        np.testing.assert_array_equal(result[~inside], original[~inside])
        if edge in ("right", "corner"):
            assert np.all(np.any(result[y1:y2, -1] != original[y1:y2, -1], axis=1))
        if edge in ("bottom", "corner"):
            assert np.all(np.any(result[-1, x1:x2] != original[-1, x1:x2], axis=1))

    def test_obscure_polygon_pixelation_blocks_are_ten_pixels_uniform_and_distinct(self) -> None:
        # Arrange
        frame = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        original = frame.copy()
        points = [(10, 10), (29, 10), (29, 29), (10, 29)]
        outside = np.ones((40, 40), dtype=bool)
        outside[10:30, 10:30] = False
        # Act
        result = obscure_polygon(frame=frame, points=points, method="pixelation", intensity=10)
        # Assert
        for y, x in [(10, 10), (10, 20), (20, 10), (20, 20)]:
            assert np.all(result[y : y + 10, x : x + 10] == result[y, x])
        assert np.any(result[10, 10] != result[10, 20])
        assert np.any(result[10, 10] != result[20, 10])
        assert np.any(result[20, 10] != result[20, 20])
        assert np.any(result[10, 20] != result[20, 20])
        np.testing.assert_array_equal(result[outside], original[outside])


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
        assert result[0][0] == 5  # x1 padded
        assert result[1][0] == 0  # x2 clamped

    def test_frame_h_zero_clamps_y2(self):
        # frame_h=0 falsy → pad uniforme, poi y2 = min(55, 0) = 0
        x1, y1, x2, y2 = 10, 10, 50, 50
        result = box_to_polygon(x1, y1, x2, y2, padding=5, frame_w=100, frame_h=0)
        assert result[0][1] == 5  # y1 padded
        assert result[2][1] == 0  # y2 clamped

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

        # Act
        result = box_to_polygon(x1, y1, x2, y2, padding=0, frame_w=None, frame_h=None)

        # Assert — x2 = 110 (non clamped)
        xs = [p[0] for p in result]
        assert max(xs) == 110

    def test_y2_not_clamped_when_frame_h_none(self):
        # Arrange — frame_h=None → y2 non viene clamped
        x1, y1, x2, y2 = 10, 80, 50, 120  # y2 > frame_h ipotetico

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


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((40, 200, 200, 400), [(15, 190), (210, 190), (210, 410), (15, 410)]),
        ((800, 200, 960, 400), [(790, 190), (985, 190), (985, 410), (790, 410)]),
        ((200, 40, 400, 200), [(190, 15), (410, 15), (410, 210), (190, 210)]),
        ((200, 800, 400, 960), [(190, 790), (410, 790), (410, 985), (190, 985)]),
        ((200, 200, 400, 400), [(190, 190), (410, 190), (410, 410), (190, 410)]),
    ],
    ids=["left", "right", "top", "bottom", "center"],
)
def test_box_to_polygon_directional_padding_returns_exact_corners(
    box: tuple[int, int, int, int],
    expected: list[tuple[int, int]],
) -> None:
    # Arrange
    config = PipelineConfig(edge_threshold=0.05, edge_padding_multiplier=2.5)
    # Act
    result = box_to_polygon(
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        padding=10,
        frame_w=1000,
        frame_h=1000,
        config=config,
    )
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("height", "base", "expected"),
    [(160, 30, 60), (160, 10, 40), (8, 1, 3)],
    ids=["scaled", "height_floor", "absolute_floor"],
)
def test_compute_adaptive_intensity_dominant_term_returns_exact_value(
    height: int,
    base: int,
    expected: int,
) -> None:
    # Arrange
    reference_height = 80
    # Act
    result = compute_adaptive_intensity(
        box_height=height,
        base_intensity=base,
        reference_height=reference_height,
    )
    # Assert
    assert result == expected


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestObscurePolygonOutsideFrame:
    """Polygons that fall outside the frame (e.g. annotations from a larger video)."""

    @pytest.mark.parametrize(
        "points",
        [
            [(110, 10), (130, 10), (130, 50), (110, 50)],
            [(10, 110), (50, 110), (50, 130), (10, 130)],
            [(100, 10), (120, 10), (120, 50), (100, 50)],
            [(-30, 10), (-10, 10), (-10, 50), (-30, 50)],
        ],
        ids=["past_right", "past_bottom", "on_right_border", "past_left"],
    )
    def test_obscure_polygon_entirely_outside_frame_leaves_frame_unchanged(self, points):
        frame = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()

        result = obscure_polygon(frame=frame, points=points, method="pixelation", intensity=10)

        np.testing.assert_array_equal(result, original)

    def test_obscure_polygon_partly_outside_frame_obscures_visible_part(self):
        frame = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()

        obscure_polygon(
            frame=frame,
            points=[(80, 10), (130, 10), (130, 50), (80, 50)],
            method="blur",
            intensity=11,
        )

        assert np.any(frame[10:50, 80:100] != original[10:50, 80:100])
        np.testing.assert_array_equal(frame[:, :80], original[:, :80])
