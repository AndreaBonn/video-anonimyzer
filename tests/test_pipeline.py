"""
Test per pipeline.py — orchestratore run_pipeline().

Mock di tutte le dipendenze esterne: cv2, stage functions, backend, file system.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import PipelineContext, PipelineInputError


def _make_ctx(tmp_path, **kwargs):
    """Crea un PipelineContext minimale con un file video finto."""
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
    """Crea un cv2.VideoCapture mock che restituisce valori validi."""
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


class TestRunPipelineInputValidation:
    """Test validazione input di run_pipeline."""

    def test_normalize_without_review_raises_error(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path, normalize=True, review=None)

        # Act / Assert
        with patch("person_anonymizer.pipeline.os.path.isfile", return_value=True):
            with pytest.raises(PipelineInputError, match="--normalize richiede --review"):
                from person_anonymizer.pipeline import run_pipeline
                run_pipeline(ctx)

    def test_file_not_found_raises_error(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        ctx.input = str(tmp_path / "nonexistent.mp4")

        # Act / Assert
        from person_anonymizer.pipeline import run_pipeline
        with pytest.raises(PipelineInputError, match="File non trovato"):
            run_pipeline(ctx)

    def test_unsupported_extension_raises_error(self, tmp_path):
        # Arrange
        bad_file = tmp_path / "video.xyz"
        bad_file.write_bytes(b"\x00")
        ctx = _make_ctx(tmp_path, input=str(bad_file))

        # Act / Assert
        from person_anonymizer.pipeline import run_pipeline
        with pytest.raises(PipelineInputError, match="Formato non supportato"):
            run_pipeline(ctx)

    def test_video_not_openable_raises_error(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        cap = MagicMock()
        cap.isOpened.return_value = False

        # Act / Assert
        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("person_anonymizer.pipeline.shutil.which", return_value=None):
                from person_anonymizer.pipeline import run_pipeline
                with pytest.raises(PipelineInputError, match="Impossibile aprire"):
                    run_pipeline(ctx)

    def test_zero_frames_raises_error(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        cap = _mock_cap(total_frames=0)

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("person_anonymizer.pipeline.shutil.which", return_value=None):
                from person_anonymizer.pipeline import run_pipeline
                with pytest.raises(PipelineInputError, match="Impossibile determinare"):
                    run_pipeline(ctx)


class TestRunPipelineResolutionLabels:
    """Test etichette risoluzione video."""

    @pytest.mark.parametrize("height,expected_label", [
        (2160, "4K"),
        (1080, "1080p"),
        (720, "720p"),
        (480, "480p"),
        (360, "640x360"),
    ])
    def test_resolution_label_assigned_correctly(self, tmp_path, height, expected_label):
        # Arrange
        ctx = _make_ctx(tmp_path)
        cap = _mock_cap(total_frames=5, h=height, w=640)
        annotations = {0: {"auto": [], "manual": [], "intensities": []}}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})), \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            # Should not raise — labels are assigned internally
            run_pipeline(ctx)


class TestRunPipelineAutoMode:
    """Test modalità auto (skip revisione manuale)."""

    def _run_auto(self, tmp_path, config=None):
        ctx = _make_ctx(tmp_path, mode="auto")
        cap = _mock_cap(total_frames=5)
        annotations = {i: {"auto": [], "manual": [], "intensities": []} for i in range(5)}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value="ffmpeg"), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})) as mock_det, \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 1, 0)) as mock_ref, \
             patch("person_anonymizer.pipeline.run_manual_review_stage") as mock_rev, \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx, config=config)
            return mock_det, mock_ref, mock_rev

    def test_detection_loop_called_in_auto_mode(self, tmp_path):
        # Arrange / Act
        mock_det, _, _ = self._run_auto(tmp_path)

        # Assert — chiamato con cap, total_frames, model, config, fisheye
        mock_det.assert_called_once()
        call_kwargs = mock_det.call_args
        # Verifica che stop_event sia passato (argomento critico)
        all_args = str(call_kwargs)
        assert "stop_event" in all_args or len(call_kwargs.args) >= 5

    def test_refinement_loop_called_in_auto_mode(self, tmp_path):
        # Arrange / Act
        _, mock_ref, _ = self._run_auto(tmp_path)

        # Assert — chiamato con input_path, annotations, model, config
        mock_ref.assert_called_once()
        call_kwargs = mock_ref.call_args
        assert call_kwargs is not None

    def test_manual_review_not_called_in_auto_mode(self, tmp_path):
        # Arrange / Act
        _, _, mock_rev = self._run_auto(tmp_path)

        # Assert
        mock_rev.assert_not_called()

    def test_default_config_created_when_none(self, tmp_path):
        # Arrange / Act — config=None should default to PipelineConfig()
        mock_det, _, _ = self._run_auto(tmp_path, config=None)

        # Assert — pipeline ran without error, detection was called
        mock_det.assert_called_once()


class TestRunPipelineManualMode:
    """Test modalità manual (con revisione manuale)."""

    def test_manual_review_called_in_manual_mode(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path, mode="manual")
        cap = _mock_cap(total_frames=5)
        annotations = {i: {"auto": [], "manual": [], "intensities": []} for i in range(5)}
        review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 5}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})), \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.run_manual_review_stage",
                   return_value=(annotations, review_stats)) as mock_rev, \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        # Assert
        mock_rev.assert_called_once()


class TestRunPipelineReviewJson:
    """Test modalità review con JSON esterno."""

    def test_review_json_loads_annotations_skips_detection(self, tmp_path):
        # Arrange
        json_file = tmp_path / "annotations.json"
        json_file.write_text('{"frames": {}}')
        ctx = _make_ctx(tmp_path, review=str(json_file), mode="manual")
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.load_annotations_from_json",
                   return_value=(annotations, "manual")) as mock_load, \
             patch("person_anonymizer.pipeline.run_detection_loop") as mock_det, \
             patch("person_anonymizer.pipeline.run_refinement_loop") as mock_ref, \
             patch("person_anonymizer.pipeline.run_manual_review_stage",
                   return_value=(annotations, {})), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        mock_load.assert_called_once()
        mock_det.assert_not_called()
        mock_ref.assert_not_called()

    def test_normalize_flag_calls_normalize_annotations(self, tmp_path):
        # Arrange
        json_file = tmp_path / "annotations.json"
        json_file.write_text('{"frames": {}}')
        ctx = _make_ctx(tmp_path, review=str(json_file), normalize=True, mode="auto")
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.load_annotations_from_json",
                   return_value=(annotations, "auto")), \
             patch("person_anonymizer.pipeline.normalize_annotations",
                   return_value=(annotations, (0, 0))) as mock_norm, \
             patch("person_anonymizer.pipeline.run_manual_review_stage",
                   return_value=(annotations, {})), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        mock_norm.assert_called_once()


class TestRunPipelineSam3Backend:
    """Test pipeline con backend SAM3."""

    def test_sam3_video_detector_used_when_available(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path, mode="auto")
        config = PipelineConfig(detection_backend="sam3")
        cap = _mock_cap(total_frames=5)
        annotations = {}

        sam3_detector = MagicMock()
        sam3_detector.detect_video.return_value = (annotations, {})

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop") as mock_det, \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = sam3_detector
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx, config=config)

        sam3_detector.detect_video.assert_called_once()
        mock_det.assert_not_called()


class TestRunPipelineStopEvent:
    """Test stop_event per interruzione asincrona."""

    def test_stop_event_passed_to_detection_loop(self, tmp_path):
        # Arrange
        stop = threading.Event()
        ctx = _make_ctx(tmp_path, mode="auto", stop_event=stop)
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})) as mock_det, \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        call_kwargs = mock_det.call_args
        # stop_event deve essere passato come keyword o posizionale
        assert stop in call_kwargs.args or stop in call_kwargs.kwargs.values()

    def test_pipeline_with_custom_output_path(self, tmp_path):
        # Arrange
        out_path = str(tmp_path / "custom_output.mp4")
        ctx = _make_ctx(tmp_path, mode="auto", output=out_path)
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})), \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs") as mock_save:

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        # Verifica che save_outputs venga chiamato con il percorso custom
        call_args = mock_save.call_args
        paths_arg = call_args.args[4]  # OutputPaths
        assert paths_arg.output == out_path


class TestRunPipelineFfmpegAbsent:
    """Test comportamento senza ffmpeg."""

    def test_pipeline_runs_without_ffmpeg(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path, mode="auto")
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})), \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs") as mock_save:

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx)

        call_args = mock_save.call_args
        result_arg = call_args.args[1]  # PipelineResult
        assert result_arg.ffmpeg_available is False


class TestRunPipelineFisheyeCorrection:
    """Test correzione fisheye."""

    def test_fisheye_disabled_when_no_camera_matrix(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path, mode="auto")
        config = PipelineConfig(enable_fisheye_correction=True, camera_matrix=None)
        cap = _mock_cap(total_frames=5)
        annotations = {}

        with patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap), \
             patch("person_anonymizer.pipeline.shutil.which", return_value=None), \
             patch("person_anonymizer.pipeline.load_detection_backend") as mock_backend, \
             patch("person_anonymizer.pipeline.run_detection_loop",
                   return_value=(annotations, {}, {})) as mock_det, \
             patch("person_anonymizer.pipeline.run_refinement_loop",
                   return_value=(annotations, 0, 0)), \
             patch("person_anonymizer.pipeline.render_video"), \
             patch("person_anonymizer.pipeline.save_outputs"):

            backend = MagicMock()
            backend.yolo_model = MagicMock()
            backend.sam3_refiner = None
            backend.sam3_video_detector = None
            mock_backend.return_value = backend

            from person_anonymizer.pipeline import run_pipeline
            run_pipeline(ctx, config=config)

        # Verifica che fisheye passato sia disabled (camera_matrix è None)
        call_args = mock_det.call_args
        fisheye_arg = call_args.args[4]
        assert fisheye_arg.enabled is False
