"""Test delle funzioni di disegno pure: ritornano nuovi array e non mutano l'input."""

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.review_drawing import (
    HELP_BAR_HEIGHT,
    INFO_BAR_HEIGHT,
    OverlayStyle,
    draw_annotations,
    draw_pending_polygon,
    draw_status_bars,
    format_info_text,
)

SIZE = 200
SQUARE = [(40, 40), (120, 40), (120, 120), (40, 120)]
STYLE = OverlayStyle(
    auto_color=(0, 255, 0),
    manual_color=(0, 120, 255),
    drawing_color=(255, 255, 0),
    fill_alpha=0.5,
)


@pytest.fixture()
def black() -> np.ndarray:
    return np.zeros((SIZE, SIZE, 3), dtype=np.uint8)


def test_overlay_style_from_config_copies_review_fields() -> None:
    config = PipelineConfig(
        review_auto_color=(1, 2, 3),
        review_manual_color=(4, 5, 6),
        review_drawing_color=(7, 8, 9),
        review_fill_alpha=0.25,
    )
    style = OverlayStyle.from_config(config)
    assert style == OverlayStyle(
        auto_color=(1, 2, 3), manual_color=(4, 5, 6), drawing_color=(7, 8, 9), fill_alpha=0.25
    )


def test_draw_annotations_blends_auto_fill_without_mutating_input(black: np.ndarray) -> None:
    annotation = {"auto": [SQUARE], "manual": [], "intensities": [1]}
    result = draw_annotations(black, annotation=annotation, style=STYLE)
    assert result is not black
    assert black.sum() == 0
    np.testing.assert_allclose(result[80, 80], [0, 128, 0], atol=1, rtol=0)
    np.testing.assert_array_equal(result[180, 180], [0, 0, 0])


def test_draw_annotations_uses_manual_color_for_manual_polygons(black: np.ndarray) -> None:
    annotation = {"auto": [], "manual": [SQUARE], "intensities": []}
    result = draw_annotations(black, annotation=annotation, style=STYLE)
    np.testing.assert_allclose(result[80, 80], [0, 60, 128], atol=1, rtol=0)


def test_draw_annotations_marks_vertices_white(black: np.ndarray) -> None:
    annotation = {"auto": [SQUARE], "manual": [], "intensities": []}
    result = draw_annotations(black, annotation=annotation, style=STYLE)
    np.testing.assert_allclose(result[40, 40], [128, 255, 128], atol=1, rtol=0)  # blended


def test_draw_annotations_without_polygons_returns_equal_copy(black: np.ndarray) -> None:
    result = draw_annotations(black, annotation={}, style=STYLE)
    assert result is not black
    np.testing.assert_array_equal(result, black)


def test_draw_pending_polygon_draws_segment_to_cursor_without_mutating_input(
    black: np.ndarray,
) -> None:
    result = draw_pending_polygon(
        black, points=[(20, 100), (60, 100)], cursor=(60, 160), color=STYLE.drawing_color
    )
    assert black.sum() == 0
    np.testing.assert_array_equal(result[100, 40], [255, 255, 0])
    np.testing.assert_allclose(result[130, 60], [255, 255, 0], atol=30, rtol=0)  # anti-aliased
    np.testing.assert_array_equal(result[104, 20], [255, 255, 255])


def test_draw_pending_polygon_without_points_returns_equal_copy(black: np.ndarray) -> None:
    result = draw_pending_polygon(black, points=[], cursor=(50, 50), color=STYLE.drawing_color)
    assert result is not black
    np.testing.assert_array_equal(result, black)


def test_format_info_text_matches_expected_line() -> None:
    annotation = {"auto": [SQUARE, SQUARE], "manual": [SQUARE], "intensities": [1, 1]}
    text = format_info_text(0, 10, annotation, True)
    assert text == "  Frame 1 / 10  |  Poligoni: 3 (2 auto, 1 man)  |  [DEL]"


def test_format_info_text_normal_mode_and_empty_annotation() -> None:
    assert format_info_text(4, 5, {}, False) == (
        "  Frame 5 / 5  |  Poligoni: 0 (0 auto, 0 man)  |  [NORM]"
    )


def test_draw_status_bars_paints_both_bars_without_mutating_input(black: np.ndarray) -> None:
    result = draw_status_bars(black, info_text="")
    assert black.sum() == 0
    assert result.shape == black.shape
    np.testing.assert_array_equal(result[0, 150], [40, 40, 40])
    np.testing.assert_array_equal(result[INFO_BAR_HEIGHT - 1, 190], [40, 40, 40])
    np.testing.assert_array_equal(result[INFO_BAR_HEIGHT + 5, 190], [0, 0, 0])
    np.testing.assert_array_equal(result[SIZE - HELP_BAR_HEIGHT, 199], [40, 40, 40])
    np.testing.assert_array_equal(result[SIZE - HELP_BAR_HEIGHT - 1, 199], [0, 0, 0])
