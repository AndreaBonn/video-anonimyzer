"""Test dei passi puri estratti da stage_detection: clamp, ghost, report row, totali."""

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.detection_report import DetectionTotals, report_row
from person_anonymizer.models import FrameDetectionResult
from person_anonymizer.stage_detection import _ghost_polygons, clamp_box

FRAME_W, FRAME_H = 640, 480


def _config():
    return PipelineConfig(
        person_padding=0, enable_adaptive_intensity=False, anonymization_intensity=10
    )


def _result(tracked=(), polygons=(), motion_count=0, sw_hits=0, ms_hits=0):
    return FrameDetectionResult(
        polygons=list(polygons),
        intensities=[10] * len(polygons),
        tracked=list(tracked),
        sw_hits=sw_hits,
        ms_hits=ms_hits,
        active_ids={tid for tid, *_ in tracked},
        prev_interp_frame=None,
        motion_count=motion_count,
    )


class TestClampBox:
    def test_box_inside_frame_is_unchanged(self):
        assert clamp_box((10, 20, 30, 40), FRAME_W, FRAME_H) == (10, 20, 30, 40)

    def test_box_on_exact_borders_is_unchanged(self):
        assert clamp_box((0, 0, FRAME_W, FRAME_H), FRAME_W, FRAME_H) == (0, 0, FRAME_W, FRAME_H)

    def test_negative_origin_is_clamped_to_zero(self):
        assert clamp_box((-5, -1, 30, 40), FRAME_W, FRAME_H) == (0, 0, 30, 40)

    def test_far_corner_is_clamped_to_frame_size(self):
        assert clamp_box((10, 20, FRAME_W + 1, FRAME_H + 50), FRAME_W, FRAME_H) == (
            10,
            20,
            FRAME_W,
            FRAME_H,
        )


class TestGhostPolygons:
    def test_zero_height_ghost_is_excluded(self):
        polygons, intensities = _ghost_polygons(
            [(1, 10, 100, 50, 100)], _config(), FRAME_W, FRAME_H
        )

        assert polygons == []
        assert intensities == []

    def test_one_pixel_height_ghost_is_included(self):
        polygons, intensities = _ghost_polygons(
            [(1, 10, 100, 50, 101)], _config(), FRAME_W, FRAME_H
        )

        assert polygons == [[(10, 100), (50, 100), (50, 101), (10, 101)]]
        assert intensities == [10]

    def test_ghost_below_the_frame_collapses_and_is_excluded(self):
        polygons, _ = _ghost_polygons(
            [(1, 10, FRAME_H + 20, 50, FRAME_H + 80)], _config(), FRAME_W, FRAME_H
        )

        assert polygons == []

    def test_ghost_is_clamped_and_order_is_preserved(self):
        polygons, intensities = _ghost_polygons(
            [(1, -10, 0, 40, 60), (2, 100, 100, 200, 200)], _config(), FRAME_W, FRAME_H
        )

        assert polygons == [
            [(0, 0), (40, 0), (40, 60), (0, 60)],
            [(100, 100), (200, 100), (200, 200), (100, 200)],
        ]
        assert intensities == [10, 10]

    def test_no_ghosts_gives_empty_lists(self):
        assert _ghost_polygons([], _config(), FRAME_W, FRAME_H) == ([], [])


class TestReportRow:
    def test_empty_confidences_give_zero(self):
        row = report_row(7, _result(motion_count=2, sw_hits=3, ms_hits=4))

        assert row == {
            "frame_number": 7,
            "persons_detected": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "motion_zones": 2,
            "sliding_window_hits": 3,
            "multiscale_hits": 4,
            "post_check_alerts": 0,
        }

    def test_confidences_are_aggregated(self):
        tracked = [(1, 0, 0, 10, 10, 0.5), (2, 0, 0, 10, 10, 0.9)]

        row = report_row(0, _result(tracked=tracked, polygons=[[], []]))

        assert row["persons_detected"] == 2
        assert row["avg_confidence"] == pytest.approx(0.7)
        assert row["min_confidence"] == 0.5
        assert row["max_confidence"] == 0.9


class TestDetectionTotals:
    def test_record_accumulates_ids_confidences_and_instances(self):
        totals = DetectionTotals()

        totals.record(_result(tracked=[(1, 0, 0, 1, 1, 0.5)], polygons=[[]]))
        totals.record(
            _result(tracked=[(1, 0, 0, 1, 1, 0.7), (2, 0, 0, 1, 1, 0.9)], polygons=[[], []])
        )

        assert totals.unique_ids == {1, 2}
        assert totals.all_confs == [0.5, 0.7, 0.9]
        assert totals.total_instances == 3
        assert totals.frames_zero_det == 0

    def test_record_counts_frames_without_polygons(self):
        totals = DetectionTotals()

        totals.record(_result())

        assert totals.frames_zero_det == 1
        assert totals.total_instances == 0

    def test_summary_lines_match_expected_text(self):
        totals = DetectionTotals()
        totals.record(_result(tracked=[(1, 0, 0, 1, 1, 0.5)], polygons=[[]] * 1500))
        totals.record(_result())

        assert totals.summary_lines(total_frames=4) == [
            "  Persone tracciate (ID unici):   1",
            "  Istanze totali rilevate:        1,500",
            "  Frame con 0 rilevamenti:        1  (25.0%)",
            "  Confidenza media:               0.50",
        ]

    def test_summary_with_no_frames_has_zero_percentage_and_no_corrupted_line(self):
        lines = DetectionTotals().summary_lines(total_frames=0)

        assert lines[2] == "  Frame con 0 rilevamenti:        0  (0.0%)"
        assert len(lines) == 4

    def test_summary_truncates_corrupted_list_after_ten(self):
        totals = DetectionTotals(corrupted=list(range(11)))

        last = totals.summary_lines(total_frames=20)[-1]

        assert last == (
            "  ATTENZIONE: 11 frame corrotti (saltati): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]..."
        )

    def test_summary_lists_all_corrupted_up_to_ten(self):
        totals = DetectionTotals(corrupted=[3, 4])

        assert totals.summary_lines(total_frames=20)[-1] == (
            "  ATTENZIONE: 2 frame corrotti (saltati): [3, 4]"
        )
