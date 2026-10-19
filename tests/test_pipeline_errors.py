"""
Test per le eccezioni custom della pipeline.

Verifica che PipelineError e PipelineInputError siano correttamente
definite e utilizzabili come eccezioni standard Python.
"""

from unittest.mock import MagicMock, patch

import cv2
import pytest

from person_anonymizer.models import (
    PipelineContext,
    PipelineError,
    PipelineInputError,
    PostCheckOutcome,
)
from person_anonymizer.pipeline import run_pipeline


@pytest.fixture
def pipeline_capture(tmp_path):
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"video")
    ctx = PipelineContext(input=str(video_file), mode="auto", no_debug=True, no_report=True)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = {
        cv2.CAP_PROP_FRAME_COUNT: 10,
        cv2.CAP_PROP_FPS: 25.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
    }.get
    with (
        patch("person_anonymizer.pipeline.cv2.VideoCapture", return_value=cap),
        patch("person_anonymizer.pipeline.shutil.which", return_value="ffmpeg"),
        patch("person_anonymizer.pipeline.load_detection_backend"),
        patch("person_anonymizer.pipeline.run_detection_loop", return_value=({}, {}, {})),
        patch(
            "person_anonymizer.pipeline.run_refinement_loop",
            return_value=({}, 0, 0, PostCheckOutcome(status="clean")),
        ),
        patch("person_anonymizer.pipeline.render_video"),
        patch("person_anonymizer.pipeline.save_outputs"),
    ):
        yield ctx, cap


class TestPipelineCaptureRelease:
    def test_zero_frames_releases_capture(self, pipeline_capture):
        ctx, cap = pipeline_capture
        cap.get.return_value = 0
        cap.get.side_effect = None

        with pytest.raises(PipelineInputError, match="numero di frame"):
            run_pipeline(ctx)

        cap.release.assert_called_once_with()

    def test_backend_failure_releases_capture(self, pipeline_capture):
        ctx, cap = pipeline_capture
        error = RuntimeError("Backend failed")

        with patch("person_anonymizer.pipeline.load_detection_backend", side_effect=error):
            with pytest.raises(RuntimeError) as exc_info:
                run_pipeline(ctx)

        assert exc_info.value is error
        cap.release.assert_called_once_with()

    def test_review_failure_releases_capture(self, pipeline_capture):
        ctx, cap = pipeline_capture
        ctx.review = "invalid.json"
        error = PipelineInputError("Invalid annotations")

        with patch("person_anonymizer.pipeline.load_annotations_from_json", side_effect=error):
            with pytest.raises(PipelineInputError) as exc_info:
                run_pipeline(ctx)

        assert exc_info.value is error
        cap.release.assert_called_once_with()

    def test_normal_pipeline_releases_capture(self, pipeline_capture):
        ctx, cap = pipeline_capture

        result = run_pipeline(ctx)

        assert result.status == "clean"
        cap.release.assert_called_once_with()


class TestPipelineExceptions:
    """Verifica le eccezioni custom della pipeline."""

    def test_pipeline_error_is_exception(self):
        # Arrange / Act / Assert
        assert issubclass(PipelineError, Exception)

    def test_pipeline_input_error_is_pipeline_error(self):
        # Arrange / Act / Assert
        assert issubclass(PipelineInputError, PipelineError)

    def test_pipeline_input_error_carries_message(self):
        # Arrange / Act
        err = PipelineInputError("file non trovato")

        # Assert
        assert str(err) == "file non trovato"

    def test_pipeline_error_catchable_as_exception(self):
        # Arrange / Act / Assert
        with pytest.raises(Exception):
            raise PipelineError("test")

    def test_pipeline_input_error_catchable_as_pipeline_error(self):
        # Arrange / Act / Assert
        with pytest.raises(PipelineError):
            raise PipelineInputError("test")
