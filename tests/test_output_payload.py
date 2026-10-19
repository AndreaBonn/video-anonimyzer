"""Pure tests for the payload/row builders extracted from save_outputs."""

import json

from person_anonymizer.config import VERSION, PipelineConfig
from person_anonymizer.models import PipelineResult, VideoMeta
from person_anonymizer.output import (
    _frame_payload,
    build_annotations_payload,
    report_rows,
)


def _result(**kwargs):
    defaults = dict(
        annotations={},
        report_data={},
        review_stats={"reviewed": 2},
        method="pixelation",
        mode="manual",
        enable_debug=False,
        enable_report=False,
        ffmpeg_available=False,
        actual_refinement_passes=1,
        refinement_annotations_added=3,
    )
    defaults.update(kwargs)
    return PipelineResult(**defaults)


def _build(result, input_path="/videos/cam1.mp4"):
    return build_annotations_payload(
        config=PipelineConfig(),
        result=result,
        input_path=input_path,
        meta=VideoMeta(fps=25.0, frame_w=640, frame_h=480, total_frames=10),
    )


class TestReportRows:
    def test_report_rows_sorts_by_frame_index(self):
        data = {5: {"frame_number": 5}, 1: {"frame_number": 1}, 3: {"frame_number": 3}}

        assert [r["frame_number"] for r in report_rows(data)] == [1, 3, 5]

    def test_report_rows_empty_report_returns_empty_list(self):
        assert report_rows({}) == []


class TestFramePayload:
    def test_frame_payload_converts_points_to_lists(self):
        ann = {"auto": [[(1, 2), (3, 4)]], "manual": [[(5, 6)]], "intensities": [7]}

        assert _frame_payload(ann) == {
            "auto": [[[1, 2], [3, 4]]],
            "manual": [[[5, 6]]],
            "intensities": [7],
        }

    def test_frame_payload_without_intensities_omits_key(self):
        assert _frame_payload({"auto": [[(0, 0)]]}) == {"auto": [[[0, 0]]], "manual": []}

    def test_frame_payload_empty_intensities_omits_key(self):
        assert "intensities" not in _frame_payload({"auto": [], "intensities": []})

    def test_frame_payload_empty_manual_gives_empty_list(self):
        assert _frame_payload({"auto": [[(1, 1)]], "manual": []})["manual"] == []


class TestBuildAnnotationsPayload:
    def test_payload_video_and_header_fields(self):
        payload = _build(_result())

        assert payload["schema_version"] == "2.0"
        assert payload["tool_version"] == VERSION
        assert payload["video"] == {
            "filename": "cam1.mp4",
            "total_frames": 10,
            "fps": 25.0,
            "resolution": [640, 480],
        }
        assert payload["mode"] == "manual"
        assert payload["review_stats"] == {"reviewed": 2}

    def test_payload_refinement_and_method_come_from_result(self):
        payload = _build(_result(method="blur"))
        config = PipelineConfig()

        assert payload["pipeline_config"]["anonymization_method"] == "blur"
        assert payload["pipeline_config"]["padding"] == config.person_padding
        assert payload["refinement"] == {
            "max_passes": config.max_refinement_passes,
            "actual_passes": 1,
            "annotations_added": 3,
            "overlap_threshold": config.refinement_overlap_threshold,
        }

    def test_payload_empty_annotations_gives_empty_frames(self):
        assert _build(_result())["frames"] == {}

    def test_payload_frames_keyed_by_string_with_optional_intensities(self):
        annotations = {
            2: {"auto": [[(1, 2), (3, 4)]], "manual": [], "intensities": [9]},
            7: {"auto": [], "manual": [[(5, 6)]]},
        }

        frames = _build(_result(annotations=annotations))["frames"]

        assert frames["2"] == {"auto": [[[1, 2], [3, 4]]], "manual": [], "intensities": [9]}
        assert frames["7"] == {"auto": [], "manual": [[[5, 6]]]}

    def test_payload_is_json_serializable(self):
        annotations = {0: {"auto": [[(1, 2)]], "manual": [], "intensities": [1]}}

        json.dumps(_build(_result(annotations=annotations)))
