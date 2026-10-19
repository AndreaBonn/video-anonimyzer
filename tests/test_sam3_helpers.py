"""Pure helpers extracted from the SAM3 backend."""

from types import SimpleNamespace

import numpy as np

from person_anonymizer.sam3_backend import (
    build_frame_annotation,
    masks_to_polygons,
    polygon_or_box,
    sam3_report_row,
)


class _FakeTensor:
    """Minimal stand-in for a torch tensor: only ``cpu().numpy()``."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _square_mask(x0: int, y0: int, size: int, shape=(60, 60)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    return mask


def test_masks_to_polygons_empty_input_returns_empty_list():
    assert masks_to_polygons([], epsilon_ratio=0.005, min_area=10) == []


def test_masks_to_polygons_flattens_masks_in_order():
    masks = [_FakeTensor(_square_mask(5, 5, 20)), _FakeTensor(_square_mask(35, 35, 20))]

    polygons = masks_to_polygons(masks, epsilon_ratio=0.005, min_area=10)

    assert len(polygons) == 2
    assert min(p[0] for p in polygons[0]) == 5
    assert min(p[0] for p in polygons[1]) == 35


def test_masks_to_polygons_squeezes_leading_channel_axis():
    mask = _FakeTensor(_square_mask(5, 5, 20)[np.newaxis, :, :])

    polygons = masks_to_polygons([mask], epsilon_ratio=0.005, min_area=10)

    assert len(polygons) == 1


def test_masks_to_polygons_drops_masks_below_min_area():
    masks = [_FakeTensor(_square_mask(5, 5, 4))]

    assert masks_to_polygons(masks, epsilon_ratio=0.005, min_area=100) == []


def test_polygon_or_box_returns_mask_polygon_when_present():
    polygon = polygon_or_box(
        _square_mask(10, 10, 20), (0, 0, 50, 50), epsilon_ratio=0.005, min_area=10
    )

    assert min(p[0] for p in polygon) == 10
    assert polygon != [(0, 0), (50, 0), (50, 50), (0, 50)]


def test_polygon_or_box_falls_back_to_box_corners_on_empty_mask():
    polygon = polygon_or_box(
        np.zeros((60, 60), dtype=np.uint8), (1, 2, 30, 40), epsilon_ratio=0.005, min_area=10
    )

    assert polygon == [(1, 2), (30, 2), (30, 40), (1, 40)]


def test_build_frame_annotation_one_intensity_per_polygon(monkeypatch):
    heights = []

    def fake_resolve(config, box_h):
        heights.append(box_h)
        return box_h * 2

    monkeypatch.setattr("person_anonymizer.anonymization.resolve_intensity", fake_resolve)
    polygons = [[(0, 10), (5, 30), (9, 20)], [(1, 1), (2, 1), (2, 4)]]

    annotation = build_frame_annotation(polygons, SimpleNamespace())

    assert annotation == {"auto": polygons, "manual": [], "intensities": [40, 6]}
    assert heights == [20, 3]


def test_build_frame_annotation_no_polygons_gives_empty_lists():
    annotation = build_frame_annotation([], SimpleNamespace())

    assert annotation == {"auto": [], "manual": [], "intensities": []}


def test_sam3_report_row_has_fixed_confidence_and_counts():
    row = sam3_report_row(frame_idx=7, n_polygons=3)

    assert row["frame_number"] == 7
    assert row["persons_detected"] == 3
    assert (row["avg_confidence"], row["min_confidence"], row["max_confidence"]) == (1.0, 1.0, 1.0)
    assert row["motion_zones"] == row["sliding_window_hits"] == 0
    assert row["multiscale_hits"] == row["post_check_alerts"] == 0
