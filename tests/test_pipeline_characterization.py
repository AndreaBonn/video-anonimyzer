import logging
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import (
    OutputPaths,
    PipelineContext,
    PipelineStoppedError,
    PostCheckOutcome,
)
from person_anonymizer.pipeline import run_pipeline


def _make_ctx(tmp_path, **kwargs):
    """Create a minimal PipelineContext with a fake video file."""
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"\x00" * 16)
    defaults = dict(
        input=str(video_file),
        mode="auto",
        method="pixelation",
        output=None,
        no_debug=True,
        no_report=True,
        review=None,
        normalize=False,
        stop_event=None,
        review_state=None,
        sse_manager=None,
        job_id=None,
    )
    defaults.update(kwargs)
    return PipelineContext(**defaults)


def _mock_cap(total_frames=10, fps=25.0, w=640, h=480):
    """Create a VideoCapture mock returning valid metadata."""
    cap = MagicMock()
    cap.isOpened.return_value = True

    def cap_get(prop):
        import cv2

        mapping = {
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
        }
        return mapping.get(prop, 0)

    cap.get.side_effect = cap_get
    cap.release = MagicMock()
    return cap


@pytest.fixture
def stages():
    annotations = {0: {"auto": [], "manual": [], "intensities": []}}
    stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
    with (
        patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=_mock_cap()) as cap,
        patch("person_anonymizer.pipeline.shutil.which", return_value="ffmpeg"),
        patch("person_anonymizer.pipeline.time.time", side_effect=[100.0, 165.0]),
        patch("person_anonymizer.pipeline.load_detection_backend"),
        patch("person_anonymizer.pipeline.build_undistortion_maps") as maps,
        patch(
            "person_anonymizer.pipeline.run_detection_loop", return_value=(annotations, {}, {})
        ) as detection,
        patch(
            "person_anonymizer.pipeline.run_refinement_loop",
            return_value=(annotations, 0, 0, PostCheckOutcome("clean")),
        ),
        patch(
            "person_anonymizer.pipeline.run_manual_review_stage", return_value=(annotations, stats)
        ) as review,
        patch("person_anonymizer.pipeline.render_video") as render,
        patch("person_anonymizer.pipeline.save_outputs") as save,
    ):
        yield dict(cap=cap, maps=maps, detection=detection, review=review, render=render, save=save)


def test_stdout_default_config_matches_snapshot(tmp_path, stages, capsys):
    ctx = _make_ctx(tmp_path, no_debug=False, no_report=False)
    run_pipeline(ctx=ctx, config=PipelineConfig())
    assert (
        capsys.readouterr().out
        == """
Person Anonymizer v7.1.0
----------------------------------------
Input:          test.mp4  (10 frame, 25fps, 480p)
Output:         test_anonymized.mp4
Modalità:      auto
Metodo:         pixelation adattivo (base: 10px, ref: 80px)
Backend:        yolo
Modello:        yolov8x.pt  |  Confidenza: 0.2
Scale:          [1.0x, 1.5x, 2.0x, 2.5x] + flip_h
Sliding window: griglia 3x3, overlap 30%
Fish-eye:       abilitato
Tracking:       ByteTrack (max_age: 45)
Smoothing:      EMA (alpha: 0.35)
Interpolazione: disabilitata (fps >= 15)
----------------------------------------

[FASE 3/5] Revisione manuale — saltata (modalità auto)

[FASE 4/5] Rendering finale...

[FASE 5/5] Post-processing...
----------------------------------------
Completato  —  Tempo totale: 1m 5s
  Output:       test_anonymized.mp4
  Debug:        test_debug.mp4
  Report:       test_report.csv

"""
    )


def test_stdout_interpolated_config_matches_snapshot(tmp_path, stages, capsys):
    ctx = _make_ctx(tmp_path, no_debug=False, no_report=False)
    config = PipelineConfig(
        enable_fisheye_correction=True,
        enable_sliding_window=True,
        enable_tracking=False,
        enable_temporal_smoothing=False,
        enable_adaptive_intensity=True,
        enable_subframe_interpolation=True,
    )
    stages["cap"].return_value = _mock_cap(fps=10.0)
    run_pipeline(ctx=ctx, config=config)
    assert (
        capsys.readouterr().out
        == """
Person Anonymizer v7.1.0
----------------------------------------
Input:          test.mp4  (10 frame, 10fps, 480p)
Output:         test_anonymized.mp4
Modalità:      auto
Metodo:         pixelation adattivo (base: 10px, ref: 80px)
Backend:        yolo
Modello:        yolov8x.pt  |  Confidenza: 0.2
Scale:          [1.0x, 1.5x, 2.0x, 2.5x] + flip_h
Sliding window: griglia 3x3, overlap 30%
Fish-eye:       abilitato
Tracking:       disabilitato
Smoothing:      disabilitato
Interpolazione: abilitata (10fps < 15fps)
----------------------------------------

[FASE 3/5] Revisione manuale — saltata (modalità auto)

[FASE 4/5] Rendering finale...

[FASE 5/5] Post-processing...
----------------------------------------
Completato  —  Tempo totale: 1m 5s
  Output:       test_anonymized.mp4
  Debug:        test_debug.mp4
  Report:       test_report.csv

"""
    )


def test_fisheye_calibrated_config_passes_maps_to_detection(tmp_path, stages):
    ctx = _make_ctx(tmp_path)
    camera_matrix, coefficients = np.eye(3), np.zeros(4)
    config = PipelineConfig(camera_matrix=camera_matrix, dist_coefficients=coefficients)
    map1, map2 = np.zeros((480, 640)), np.ones((480, 640))
    stages["maps"].return_value = (map1, map2)
    run_pipeline(ctx=ctx, config=config)
    stages["maps"].assert_called_once_with(camera_matrix, coefficients, 640, 480)
    fisheye = stages["detection"].call_args.args[4]
    assert fisheye.enabled is True
    assert fisheye.undist_map1 is map1
    assert fisheye.undist_map2 is map2


def test_output_without_override_uses_input_directory_for_all_paths(tmp_path, stages):
    ctx = _make_ctx(tmp_path, no_debug=False, no_report=False)
    run_pipeline(ctx=ctx, config=PipelineConfig())
    stages["save"].assert_called_once()
    paths = stages["save"].call_args.args[4]
    assert paths == OutputPaths(
        output=str(tmp_path / "test_anonymized.mp4"),
        temp_video=str(tmp_path / "test_temp_noaudio.avi"),
        temp_debug=str(tmp_path / "test_temp_debug.avi"),
        debug=str(tmp_path / "test_debug.mp4"),
        report=str(tmp_path / "test_report.csv"),
        json=str(tmp_path / "test_annotations.json"),
    )
    assert stages["render"].call_args.args[1] == paths.temp_video
    assert stages["render"].call_args.kwargs["debug_path"] == paths.temp_debug


def test_output_flags_disabled_omit_debug_render_and_summary_rows(tmp_path, stages, capsys):
    # Arrange / Act: pair the disabled flags with the enabled output in the same test.
    for no_debug, no_report in ((False, False), (True, False), (False, True), (True, True)):
        ctx = _make_ctx(tmp_path, no_debug=no_debug, no_report=no_report)
        with patch("person_anonymizer.pipeline.time.time", side_effect=[100.0, 165.0]):
            run_pipeline(ctx=ctx, config=PipelineConfig())
        # Assert
        stdout = capsys.readouterr().out
        assert ("  Debug:        test_debug.mp4\n" in stdout) is not no_debug
        assert ("  Report:       test_report.csv\n" in stdout) is not no_report
        expected_debug = None if no_debug else str(tmp_path / "test_temp_debug.avi")
        assert stages["render"].call_args.kwargs["debug_path"] == expected_debug
        result = stages["save"].call_args.args[1]
        assert result.enable_debug is not no_debug
        assert result.enable_report is not no_report


def test_context_missing_mode_and_method_uses_config_values(tmp_path, stages):
    ctx = _make_ctx(tmp_path, mode=None, method=None)
    config = PipelineConfig(operation_mode="manual", anonymization_method="blur")
    run_pipeline(ctx=ctx, config=config)
    stages["review"].assert_called_once()
    assert stages["render"].call_args.args[6] == "blur"
    result = stages["save"].call_args.args[1]
    assert (result.mode, result.method) == ("manual", "blur")


def test_interpolation_very_low_fps_emits_warning(tmp_path, stages, capsys):
    ctx = _make_ctx(tmp_path)
    stages["cap"].return_value = _mock_cap(fps=2.0)
    config = PipelineConfig(enable_subframe_interpolation=True)
    run_pipeline(ctx=ctx, config=config)
    assert capsys.readouterr().out.startswith(
        "ATTENZIONE: fps molto basso (< 3). L'interpolazione potrebbe produrre artefatti. "
        "Considera di disabilitarla.\n\nPerson Anonymizer v7.1.0\n"
    )


@pytest.mark.parametrize("checkpoint", ["review", "render"])
def test_stop_after_stage_prevents_next_stage_and_removes_temps(tmp_path, stages, checkpoint):
    # Arrange
    event = threading.Event()
    ctx = _make_ctx(tmp_path, mode="manual", stop_event=event)
    temps = [tmp_path / "test_temp_noaudio.avi", tmp_path / "test_temp_debug.avi"]
    for temp in temps:
        temp.write_bytes(b"partial render")
    stage_result = stages[checkpoint].return_value

    def stop_stage(*args, **kwargs):
        event.set()
        return stage_result

    stages[checkpoint].side_effect = stop_stage
    # Act / Assert
    with pytest.raises(PipelineStoppedError):
        run_pipeline(ctx=ctx, config=PipelineConfig())
    stages[checkpoint].assert_called_once()
    stages["save"].assert_not_called()
    if checkpoint == "review":
        stages["render"].assert_not_called()
    assert all(not temp.exists() for temp in temps)
    # Positive pair: the same stages without a stop reach output persistence.
    event.clear()
    stages[checkpoint].side_effect = None
    stages["render"].reset_mock()
    with patch("person_anonymizer.pipeline.time.time", side_effect=[100.0, 165.0]):
        run_pipeline(ctx=ctx, config=PipelineConfig())
    stages["render"].assert_called_once()
    stages["save"].assert_called_once()


def test_stop_during_refinement_skips_manual_review(tmp_path, stages):
    # A stopped web job must not open the manual review, which would wait for
    # a reviewer that is no longer there.
    event = threading.Event()
    ctx = _make_ctx(tmp_path, mode="manual", stop_event=event)
    annotations = {0: {"auto": [], "manual": [], "intensities": []}}

    def stop_refinement(*args, **kwargs):
        event.set()
        return annotations, 1, 0, PostCheckOutcome("clean")

    with patch("person_anonymizer.pipeline.run_refinement_loop", side_effect=stop_refinement):
        with pytest.raises(PipelineStoppedError):
            run_pipeline(ctx=ctx, config=PipelineConfig())
    stages["review"].assert_not_called()

    # Positive pair: without a stop the same run reaches the manual review.
    event.clear()
    with patch("person_anonymizer.pipeline.time.time", side_effect=[100.0, 165.0]):
        run_pipeline(ctx=ctx, config=PipelineConfig())
    stages["review"].assert_called_once()


def test_stop_cleanup_oserror_preserves_stop_and_logs_warning(tmp_path, stages, caplog):
    # Arrange
    event = threading.Event()
    ctx = _make_ctx(tmp_path, stop_event=event)
    temp = tmp_path / "test_temp_noaudio.avi"

    def stop_render(*args, **kwargs):
        temp.write_bytes(b"partial render")
        event.set()

    stages["render"].side_effect = stop_render
    # Act
    with (
        patch(
            "person_anonymizer.pipeline.os.remove", side_effect=OSError("cleanup failed")
        ) as remove,
        caplog.at_level(logging.WARNING, logger="person_anonymizer.pipeline"),
        pytest.raises(PipelineStoppedError),
    ):
        run_pipeline(ctx=ctx, config=PipelineConfig())
    # Assert
    remove.assert_called_once_with(str(temp))
    assert temp.read_bytes() == b"partial render"
    record = next(
        record for record in caplog.records if record.name == "person_anonymizer.pipeline"
    )
    assert record.levelno == logging.WARNING
    assert record.getMessage() == f"Impossibile rimuovere il file temporaneo {temp}"
    assert isinstance(record.exc_info[1], OSError)
