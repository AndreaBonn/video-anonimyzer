"""Test puri per i passi estratti da render_video: obscure_frame e build_debug_frame."""

from unittest.mock import patch

import numpy as np

from person_anonymizer.config import PipelineConfig
from person_anonymizer.rendering import build_debug_frame, obscure_frame

POLY = [(10, 10), (40, 10), (40, 40), (10, 40)]


def _noise_frame():
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def _recorded_intensities(ann, method, config):
    calls = []

    def fake(frame, poly, meth, intensity):
        calls.append(intensity)
        return frame

    with patch("person_anonymizer.rendering.obscure_polygon", side_effect=fake):
        obscure_frame(np.zeros((8, 8, 3), np.uint8), ann, method, config)
    return calls


class TestObscureFrame:
    def test_obscure_frame_changes_polygon_area_and_leaves_input_untouched(self):
        frame = _noise_frame()
        original = frame.copy()
        ann = {"auto": [POLY], "manual": [], "intensities": [8]}

        result = obscure_frame(frame, ann, "pixelation", PipelineConfig())

        assert np.array_equal(frame, original)
        assert not np.array_equal(result[10:40, 10:40], original[10:40, 10:40])
        assert np.array_equal(result[50:, 50:], original[50:, 50:])

    def test_obscure_frame_without_annotations_returns_equal_copy(self):
        frame = _noise_frame()

        result = obscure_frame(frame, {}, "pixelation", PipelineConfig())

        assert result is not frame
        assert np.array_equal(result, frame)

    def test_obscure_frame_adaptive_uses_per_polygon_intensity(self):
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=10)
        ann = {"auto": [POLY, POLY], "manual": [], "intensities": [6]}

        # Second auto polygon has no per-polygon value: falls back to the configured one.
        assert _recorded_intensities(ann, "pixelation", config) == [6, 10]

    def test_obscure_frame_non_adaptive_ignores_per_polygon_intensity(self):
        config = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)
        ann = {"auto": [POLY], "manual": [], "intensities": [6]}

        assert _recorded_intensities(ann, "pixelation", config) == [10]

    def test_obscure_frame_manual_uses_configured_intensity(self):
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=12)
        ann = {"auto": [], "manual": [POLY], "intensities": [3]}

        assert _recorded_intensities(ann, "pixelation", config) == [12]

    def test_obscure_frame_blur_makes_even_intensity_odd_for_auto_and_manual(self):
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=10)
        ann = {"auto": [POLY], "manual": [POLY], "intensities": [6]}

        assert _recorded_intensities(ann, "blur", config) == [7, 11]

    def test_obscure_frame_pixelation_keeps_even_intensity(self):
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=10)
        ann = {"auto": [POLY], "manual": [POLY], "intensities": [6]}

        assert _recorded_intensities(ann, "pixelation", config) == [6, 10]


class TestBuildDebugFrame:
    def test_build_debug_frame_passes_auto_and_manual_polygons(self):
        frame = _noise_frame()
        config = PipelineConfig()
        ann = {"auto": [POLY], "manual": [[(1, 1), (5, 1), (5, 5)]]}

        with patch("person_anonymizer.rendering.draw_debug_polygons") as draw:
            draw.return_value = "debug"
            result = build_debug_frame(frame, ann, config)

        draw.assert_called_once_with(frame, ann["auto"], ann["manual"], config)
        assert result == "debug"

    def test_build_debug_frame_missing_keys_default_to_empty(self):
        with patch("person_anonymizer.rendering.draw_debug_polygons") as draw:
            build_debug_frame(_noise_frame(), {}, PipelineConfig())

        assert draw.call_args.args[1:3] == ([], [])
