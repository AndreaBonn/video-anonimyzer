"""Tests for the pure path and report helpers of the pipeline."""

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import OutputPaths, PipelineResult, VideoMeta
from person_anonymizer.pipeline_report import (
    RunLabels,
    build_output_paths,
    format_banner,
    format_summary,
    resolution_label,
)

META = VideoMeta(fps=25.0, frame_w=1280, frame_h=720, total_frames=100)
LABELS = RunLabels(input_name="a.mp4", output_name="a_anonymized.mp4", mode="auto", method="blur")


def _result(**overrides) -> PipelineResult:
    values = {
        "annotations": {},
        "report_data": {},
        "review_stats": {},
        "method": "blur",
        "mode": "auto",
        "enable_debug": True,
        "enable_report": True,
        "ffmpeg_available": True,
        "actual_refinement_passes": 0,
        "refinement_annotations_added": 0,
    }
    values.update(overrides)
    return PipelineResult(**values)


def _paths() -> OutputPaths:
    return build_output_paths(input_path="/v/a.mp4", output=None)


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (479, "640x479"),
        (480, "480p"),
        (719, "480p"),
        (720, "720p"),
        (1079, "720p"),
        (1080, "1080p"),
        (2159, "1080p"),
        (2160, "4K"),
    ],
)
def test_resolution_label_boundaries(height, expected):
    assert resolution_label(frame_w=640, frame_h=height) == expected


def test_build_output_paths_without_output_uses_input_directory():
    paths = build_output_paths(input_path="/v/a.mp4", output=None)

    assert paths == OutputPaths(
        output="/v/a_anonymized.mp4",
        temp_video="/v/a_temp_noaudio.avi",
        temp_debug="/v/a_temp_debug.avi",
        debug="/v/a_debug.mp4",
        report="/v/a_report.csv",
        json="/v/a_annotations.json",
    )


def test_build_output_paths_with_output_places_side_files_next_to_it():
    paths = build_output_paths(input_path="/v/a.mp4", output="/out/final.mp4")

    assert paths.output == "/out/final.mp4"
    assert paths.temp_video == "/out/a_temp_noaudio.avi"
    assert paths.json == "/out/a_annotations.json"


def test_format_banner_tracking_off_says_disabled():
    config = PipelineConfig(enable_tracking=False)

    banner = format_banner(meta=META, labels=LABELS, config=config, do_interpolation=False)

    assert "Tracking:       disabilitato" in banner


def test_format_banner_tracking_on_shows_max_age():
    config = PipelineConfig(enable_tracking=True, track_max_age=7)

    banner = format_banner(meta=META, labels=LABELS, config=config, do_interpolation=False)

    assert "Tracking:       ByteTrack (max_age: 7)" in banner
    assert "Tracking:       disabilitato" not in banner


def test_format_banner_header_and_interpolation_lines():
    config = PipelineConfig(interpolation_fps_threshold=15)

    on = format_banner(meta=META, labels=LABELS, config=config, do_interpolation=True)
    off = format_banner(meta=META, labels=LABELS, config=config, do_interpolation=False)

    assert "Input:          a.mp4  (100 frame, 25fps, 720p)" in on
    assert "Interpolazione: abilitata (25fps < 15fps)" in on
    assert "Interpolazione: disabilitata (fps >= 15)" in off
    assert on.startswith("\nPerson Anonymizer v")


def test_format_summary_lists_optional_rows_only_when_enabled():
    full = format_summary(
        elapsed_s=65.0, paths=_paths(), result=_result(mode="manual"), normalized=False
    )
    bare = format_summary(
        elapsed_s=65.0,
        paths=_paths(),
        result=_result(enable_debug=False, enable_report=False),
        normalized=False,
    )

    assert "Tempo totale: 1m 5s" in full
    assert "  Debug:        a_debug.mp4" in full
    assert "  Report:       a_report.csv" in full
    assert "  Annotazioni:  a_annotations.json" in full
    assert "Debug:" not in bare
    assert "Report:" not in bare
    assert "Annotazioni:" not in bare
    assert full.endswith("\n")


def test_format_summary_normalized_auto_run_lists_annotations():
    summary = format_summary(
        elapsed_s=0.0, paths=_paths(), result=_result(mode="auto"), normalized=True
    )

    assert "  Annotazioni:  a_annotations.json" in summary
