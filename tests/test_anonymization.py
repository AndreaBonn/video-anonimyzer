"""
Test per anonymization.py — funzioni pure senza cv2.

compute_adaptive_intensity, box_to_polygon e polygon_to_bbox
operano su geometria pura e non richiedono frame reali.
"""

import pytest

from anonymization import (
    box_to_polygon,
    compute_adaptive_intensity,
    polygon_to_bbox,
)
from config import PipelineConfig


# ============================================================
# compute_adaptive_intensity
# ============================================================


class TestComputeAdaptiveIntensity:
    """Verifica il calcolo dell'intensità di oscuramento adattiva."""

    def test_reference_height_returns_base_intensity(self):
        # Arrange — altezza uguale alla reference → scale_factor = 1.0
        base_intensity = 10
        reference_height = 80

        # Act
        result = compute_adaptive_intensity(
            box_height=80,
            base_intensity=base_intensity,
            reference_height=reference_height,
        )

        # Assert — adaptive = 10 * 1.0 = 10; min_intensity = max(3, 80//4) = 20
        # Il risultato è max(20, 10) = 20 (la soglia minima vince)
        assert result == max(3, 80 // 4)

    def test_small_height_minimum_guaranteed(self):
        # Arrange — figura piccola: 30px, base 10, ref 80
        # adaptive = int(10 * 30/80) = int(3.75) = 3
        # min_intensity = max(3, 30//4) = max(3, 7) = 7
        # risultato = max(7, 3) = 7

        # Act
        result = compute_adaptive_intensity(
            box_height=30, base_intensity=10, reference_height=80
        )

        # Assert
        assert result == 7

    def test_large_height_approximately_double(self):
        # Arrange — altezza doppia della reference → scala circa 2x
        base_intensity = 10
        reference_height = 80

        # Act
        result_base = compute_adaptive_intensity(80, base_intensity, reference_height)
        result_double = compute_adaptive_intensity(160, base_intensity, reference_height)

        # Assert — risultato con altezza doppia >= risultato con altezza base
        assert result_double >= result_base

    def test_zero_reference_height_returns_base(self):
        # Arrange — reference_height = 0 → guardia nel codice → restituisce base
        base_intensity = 10

        # Act
        result = compute_adaptive_intensity(
            box_height=50, base_intensity=base_intensity, reference_height=0
        )

        # Assert
        assert result == base_intensity

    def test_minimum_intensity_never_below_3(self):
        # Arrange — scatola di 1px, base 1
        # min_intensity = max(3, 1//4) = max(3, 0) = 3

        # Act
        result = compute_adaptive_intensity(
            box_height=1, base_intensity=1, reference_height=80
        )

        # Assert
        assert result >= 3

    def test_intensity_increases_with_height(self):
        # Arrange
        heights = [20, 40, 80, 160]
        results = [
            compute_adaptive_intensity(h, base_intensity=10, reference_height=80)
            for h in heights
        ]

        # Assert — l'intensità è non-decrescente all'aumentare dell'altezza
        for i in range(len(results) - 1):
            assert results[i + 1] >= results[i]


# ============================================================
# box_to_polygon
# ============================================================


class TestBoxToPolygon:
    """Verifica la conversione bounding box → poligono 4 punti."""

    def test_no_padding_returns_four_corners(self):
        # Arrange
        x1, y1, x2, y2 = 10, 20, 50, 80

        # Act
        polygon = box_to_polygon(x1, y1, x2, y2, padding=0)

        # Assert — 4 punti nei 4 angoli del box
        assert len(polygon) == 4
        assert (10, 20) in polygon
        assert (50, 20) in polygon
        assert (50, 80) in polygon
        assert (10, 80) in polygon

    def test_with_padding_expands_box(self):
        # Arrange
        x1, y1, x2, y2 = 20, 20, 60, 60
        padding = 5

        # Act
        polygon = box_to_polygon(x1, y1, x2, y2, padding=padding)

        # Assert — i punti devono essere espansi di padding in ogni direzione
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        assert min(xs) == x1 - padding
        assert max(xs) == x2 + padding
        assert min(ys) == y1 - padding
        assert max(ys) == y2 + padding

    def test_clamped_to_frame_borders(self):
        # Arrange — box vicino ai bordi con padding che uscirebbe dal frame
        frame_w, frame_h = 100, 100
        x1, y1, x2, y2 = 2, 2, 98, 98
        padding = 10

        # Act
        polygon = box_to_polygon(
            x1, y1, x2, y2,
            padding=padding,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        # Assert — nessun punto supera i limiti del frame
        for px, py in polygon:
            assert px >= 0
            assert py >= 0
            assert px <= frame_w
            assert py <= frame_h

    def test_edge_padding_multiplier_applied_near_left_border(self):
        # Arrange — x1 molto vicino al bordo sinistro (< edge_threshold * frame_w)
        config = PipelineConfig(edge_threshold=0.05, edge_padding_multiplier=2.5)
        frame_w, frame_h = 1000, 1000
        # x1=10 < 1000*0.05=50 → bordo sinistro attivo
        x1, y1, x2, y2 = 10, 200, 200, 400
        padding = 10

        # Act
        polygon_with_edge = box_to_polygon(
            x1, y1, x2, y2,
            padding=padding,
            frame_w=frame_w,
            frame_h=frame_h,
            config=config,
        )
        polygon_no_edge = box_to_polygon(
            x1, y1, x2, y2,
            padding=padding,
            frame_w=None,
            frame_h=None,
        )

        # Assert — il padding sinistro con edge è maggiore di quello senza
        min_x_edge = min(p[0] for p in polygon_with_edge)
        min_x_no_edge = min(p[0] for p in polygon_no_edge)
        assert min_x_edge <= min_x_no_edge

    def test_returns_list_of_tuples(self):
        polygon = box_to_polygon(0, 0, 100, 100, padding=0)
        assert isinstance(polygon, list)
        for point in polygon:
            assert isinstance(point, tuple)
            assert len(point) == 2

    def test_zero_padding_no_edge_detection(self):
        # Arrange — padding=0, nessuna espansione attesa
        x1, y1, x2, y2 = 50, 50, 150, 150

        # Act
        polygon = box_to_polygon(x1, y1, x2, y2, padding=0)

        # Assert
        assert (x1, y1) in polygon
        assert (x2, y2) in polygon


# ============================================================
# polygon_to_bbox
# ============================================================


class TestPolygonToBbox:
    """Verifica la conversione poligono → bounding box."""

    def test_triangle_bbox(self):
        # Arrange
        triangle = [(0, 0), (100, 0), (50, 80)]

        # Act
        bbox = polygon_to_bbox(triangle)

        # Assert
        assert bbox == [0, 0, 100, 80]

    def test_rectangle_bbox_identical(self):
        # Arrange — 4 punti di un rettangolo → bbox identico al rettangolo
        rectangle = [(10, 20), (60, 20), (60, 80), (10, 80)]

        # Act
        bbox = polygon_to_bbox(rectangle)

        # Assert
        assert bbox == [10, 20, 60, 80]

    def test_single_point_bbox(self):
        # Arrange
        polygon = [(5, 7)]

        # Act
        bbox = polygon_to_bbox(polygon)

        # Assert
        assert bbox == [5, 7, 5, 7]

    def test_bbox_format(self):
        # Arrange
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Act
        bbox = polygon_to_bbox(polygon)

        # Assert — formato [x1, y1, x2, y2] con x1 <= x2 e y1 <= y2
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert x1 <= x2
        assert y1 <= y2

    def test_pentagon_bbox(self):
        # Arrange
        pentagon = [(2, 0), (4, 0), (5, 3), (3, 5), (0, 3)]

        # Act
        bbox = polygon_to_bbox(pentagon)

        # Assert
        assert bbox == [0, 0, 5, 5]

    def test_box_to_polygon_and_back_consistency(self):
        # Arrange — round-trip: box → polygon → bbox deve restituire il box originale
        x1, y1, x2, y2 = 10, 20, 50, 80
        polygon = box_to_polygon(x1, y1, x2, y2, padding=0)

        # Act
        bbox = polygon_to_bbox(polygon)

        # Assert
        assert bbox == [x1, y1, x2, y2]
