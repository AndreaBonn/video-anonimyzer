"""
Test per detection.py — funzioni pure senza YOLO/modelli pesanti.

Le funzioni che richiedono cv2.dnn.NMSBoxes sono skipplate se cv2
non è disponibile. Tutte le altre funzioni usano solo numpy.
"""

import pytest

try:
    import cv2 as _cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from person_anonymizer.detection import (
    compute_iou_boxes,
    get_window_patches,
    patch_intersects_motion,
)

if CV2_AVAILABLE:
    from person_anonymizer.detection import apply_nms


# ============================================================
# compute_iou_boxes
# ============================================================


class TestComputeIouBoxes:
    """Verifica il calcolo IoU tra bounding box."""

    def test_boxes_identical(self):
        # Arrange
        box = [10, 10, 50, 50]

        # Act
        iou = compute_iou_boxes(box, box)

        # Assert
        assert iou == pytest.approx(1.0)

    def test_boxes_no_overlap(self):
        # Arrange
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]

        # Act
        iou = compute_iou_boxes(box_a, box_b)

        # Assert
        assert iou == pytest.approx(0.0)

    def test_boxes_partial_overlap(self):
        # Arrange — due box 10x10 sovrapposti per 5x5 (25px su 175px di unione)
        box_a = [0, 0, 10, 10]
        box_b = [5, 5, 15, 15]

        # Act
        iou = compute_iou_boxes(box_a, box_b)

        # Assert — intersezione 5*5=25, unione 100+100-25=175
        assert 0.0 < iou < 1.0
        assert iou == pytest.approx(25 / 175)

    def test_boxes_contained(self):
        # Arrange — box_b contenuto in box_a
        box_a = [0, 0, 100, 100]
        box_b = [10, 10, 40, 40]

        # Act
        iou = compute_iou_boxes(box_a, box_b)

        # Assert — intersezione = area di box_b = 30*30 = 900, unione = 10000+900-900 = 10000
        assert iou > 0.0
        assert iou == pytest.approx(900 / 10000)

    def test_boxes_touching_edge(self):
        # Arrange — box adiacenti che si toccano su un bordo (nessun overlap reale)
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]

        # Act
        iou = compute_iou_boxes(box_a, box_b)

        # Assert — intersezione = 0 (bordi a contatto, area nulla)
        assert iou == pytest.approx(0.0)

    def test_symmetry(self):
        # Arrange
        box_a = [0, 0, 20, 20]
        box_b = [10, 10, 30, 30]

        # Act
        iou_ab = compute_iou_boxes(box_a, box_b)
        iou_ba = compute_iou_boxes(box_b, box_a)

        # Assert — IoU è simmetrica
        assert iou_ab == pytest.approx(iou_ba)

    def test_zero_area_box(self):
        """Box con area zero (x1 == x2) non deve causare ZeroDivisionError."""
        # Arrange
        box_a = [10, 10, 10, 20]  # larghezza zero
        box_b = [5, 5, 15, 25]
        # Act
        result = compute_iou_boxes(box_a, box_b)
        # Assert
        assert result == 0.0


# ============================================================
# apply_nms
# ============================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 non disponibile")
class TestApplyNms:
    """Verifica Non-Maximum Suppression."""

    def test_empty_list(self):
        # Arrange / Act
        result = apply_nms([], iou_threshold=0.5)

        # Assert
        assert result == []

    def test_single_box(self):
        # Arrange
        boxes = [[10.0, 10.0, 50.0, 50.0, 0.9]]

        # Act
        result = apply_nms(boxes, iou_threshold=0.5)

        # Assert
        assert len(result) == 1
        assert result[0] == boxes[0]

    def test_overlapping_boxes_kept_highest_score(self):
        # Arrange — due box quasi identici, score diverso
        boxes = [
            [10.0, 10.0, 50.0, 50.0, 0.9],
            [11.0, 11.0, 51.0, 51.0, 0.5],
        ]

        # Act
        result = apply_nms(boxes, iou_threshold=0.5)

        # Assert — deve restare solo il box con score più alto
        assert len(result) == 1
        assert result[0][4] == pytest.approx(0.9)

    def test_non_overlapping_boxes_all_kept(self):
        # Arrange — due box distanti
        boxes = [
            [0.0, 0.0, 10.0, 10.0, 0.9],
            [100.0, 100.0, 110.0, 110.0, 0.8],
        ]

        # Act
        result = apply_nms(boxes, iou_threshold=0.5)

        # Assert — entrambi devono essere mantenuti
        assert len(result) == 2


# ============================================================
# get_window_patches
# ============================================================


class TestGetWindowPatches:
    """Verifica la generazione di patch per sliding window."""

    def test_grid_3x3_count(self):
        # Arrange
        frame_w, frame_h, grid, overlap = 600, 600, 3, 0.0

        # Act
        patches = get_window_patches(frame_w, frame_h, grid, overlap)

        # Assert — griglia 3x3 = 9 patch
        assert len(patches) == 9

    def test_grid_2x2_count(self):
        patches = get_window_patches(400, 400, 2, 0.0)
        assert len(patches) == 4

    def test_patches_format(self):
        # Arrange
        patches = get_window_patches(300, 300, 2, 0.0)

        # Assert — ogni patch è una tupla (x1, y1, x2, y2) con x2 > x1 e y2 > y1
        for x1, y1, x2, y2 in patches:
            assert x2 > x1
            assert y2 > y1

    def test_patches_coverage_top_left(self):
        # Arrange
        frame_w, frame_h, grid, overlap = 300, 300, 3, 0.0

        # Act
        patches = get_window_patches(frame_w, frame_h, grid, overlap)

        # Assert — la prima patch parte da (0, 0)
        x1, y1, _, _ = patches[0]
        assert x1 == 0
        assert y1 == 0

    def test_patches_do_not_exceed_frame(self):
        # Arrange
        frame_w, frame_h = 320, 240

        # Act
        patches = get_window_patches(frame_w, frame_h, 3, 0.3)

        # Assert — nessuna patch supera i limiti del frame
        for x1, y1, x2, y2 in patches:
            assert x1 >= 0
            assert y1 >= 0
            assert x2 <= frame_w
            assert y2 <= frame_h

    def test_patches_with_overlap_more_coverage(self):
        # Arrange — con overlap le patch coprono aree maggiori
        patches_no_overlap = get_window_patches(300, 300, 3, 0.0)
        patches_overlap = get_window_patches(300, 300, 3, 0.3)

        # Act
        def total_area(patches):
            return sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in patches)

        # Assert — overlap produce patch singole più grandi
        assert total_area(patches_overlap) >= total_area(patches_no_overlap)


# ============================================================
# patch_intersects_motion
# ============================================================


class TestPatchIntersectsMotion:
    """Verifica l'intersezione tra una patch e le zone di movimento."""

    def test_intersection_true(self):
        # Arrange — patch [0,0,50,50], regione di movimento [25,25,75,75]
        motion_regions = [(25, 25, 75, 75)]

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert
        assert result is True

    def test_intersection_false_separated(self):
        # Arrange — patch [0,0,50,50], regione [100,100,150,150]
        motion_regions = [(100, 100, 150, 150)]

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert
        assert result is False

    def test_intersection_touching_border_false(self):
        # Arrange — patch [0,0,50,50], regione adiacente [50,0,100,50]
        # Bordi a contatto ma non sovrapposti (px2 == mx1 → no overlap)
        motion_regions = [(50, 0, 100, 50)]

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert — bordi a contatto non costituiscono intersezione
        assert result is False

    def test_intersection_multiple_regions_one_matching(self):
        # Arrange — due regioni, solo la seconda interseca
        motion_regions = [
            (200, 200, 300, 300),  # distante
            (10, 10, 40, 40),  # interseca la patch [0,0,50,50]
        ]

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert
        assert result is True

    def test_intersection_empty_regions(self):
        # Arrange — nessuna regione di movimento
        motion_regions = []

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert
        assert result is False

    def test_intersection_contained_region(self):
        # Arrange — regione completamente dentro la patch
        motion_regions = [(10, 10, 30, 30)]

        # Act
        result = patch_intersects_motion(0, 0, 50, 50, motion_regions)

        # Assert
        assert result is True
