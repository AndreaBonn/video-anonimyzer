"""
Test per cli.py — parsing argomenti e gestione errori.
"""

from unittest.mock import patch

import pytest

from person_anonymizer.cli import main, parse_args
from person_anonymizer.models import PipelineContext, PipelineError, PipelineInputError


class TestParseArgs:
    """Test per parse_args."""

    def test_requires_input_argument(self):
        # Arrange / Act / Assert
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["cli"]):
                parse_args()

    def test_parses_input_path(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4"]):
            args = parse_args()

        # Assert
        assert args.input == "video.mp4"

    def test_default_mode_is_none(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4"]):
            args = parse_args()

        # Assert
        assert args.mode is None

    def test_parses_mode_auto(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "-M", "auto"]):
            args = parse_args()

        # Assert
        assert args.mode == "auto"

    def test_parses_mode_manual(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "-M", "manual"]):
            args = parse_args()

        # Assert
        assert args.mode == "manual"

    def test_rejects_invalid_mode(self):
        # Arrange / Act / Assert
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["cli", "video.mp4", "-M", "invalid"]):
                parse_args()

    def test_parses_output_path(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "-o", "out.mp4"]):
            args = parse_args()

        # Assert
        assert args.output == "out.mp4"

    def test_parses_method_blur(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "-m", "blur"]):
            args = parse_args()

        # Assert
        assert args.method == "blur"

    def test_parses_no_debug_flag(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "--no-debug"]):
            args = parse_args()

        # Assert
        assert args.no_debug is True

    def test_parses_no_report_flag(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "--no-report"]):
            args = parse_args()

        # Assert
        assert args.no_report is True

    def test_parses_review_path(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "--review", "ann.json"]):
            args = parse_args()

        # Assert
        assert args.review == "ann.json"

    def test_parses_normalize_flag(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "--normalize"]):
            args = parse_args()

        # Assert
        assert args.normalize is True

    def test_defaults_are_correct(self):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4"]):
            args = parse_args()

        # Assert
        assert args.mode is None
        assert args.output is None
        assert args.method is None
        assert args.no_debug is False
        assert args.no_report is False
        assert args.review is None
        assert args.normalize is False
        assert args.backend == "yolo"

    def test_parses_backend_yolo(self):
        with patch("sys.argv", ["cli", "video.mp4", "--backend", "yolo"]):
            args = parse_args()
        assert args.backend == "yolo"

    def test_parses_backend_yolo_sam3(self):
        with patch("sys.argv", ["cli", "video.mp4", "--backend", "yolo+sam3"]):
            args = parse_args()
        assert args.backend == "yolo+sam3"

    def test_parses_backend_sam3(self):
        with patch("sys.argv", ["cli", "video.mp4", "--backend", "sam3"]):
            args = parse_args()
        assert args.backend == "sam3"

    def test_rejects_invalid_backend(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["cli", "video.mp4", "--backend", "invalid"]):
                parse_args()


class TestMain:
    """Test per main() — gestione errori."""

    @patch(
        "person_anonymizer.pipeline.run_pipeline",
        side_effect=PipelineInputError("file non trovato"),
    )
    def test_pipeline_input_error_exits_1(self, mock_run):
        # Arrange / Act / Assert
        with patch("sys.argv", ["cli", "video.mp4"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("person_anonymizer.pipeline.run_pipeline", side_effect=PipelineError("errore generico"))
    def test_pipeline_error_exits_1(self, mock_run):
        # Arrange / Act / Assert
        with patch("sys.argv", ["cli", "video.mp4"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("person_anonymizer.pipeline.run_pipeline", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits_1(self, mock_run):
        # Arrange / Act / Assert
        with patch("sys.argv", ["cli", "video.mp4"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("person_anonymizer.pipeline.run_pipeline")
    def test_main_creates_pipeline_context(self, mock_run):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "-M", "auto", "-m", "blur"]):
            main()

        # Assert
        mock_run.assert_called_once()
        ctx = mock_run.call_args[0][0]
        assert isinstance(ctx, PipelineContext)
        assert ctx.input == "video.mp4"
        assert ctx.mode == "auto"
        assert ctx.method == "blur"

    @patch("person_anonymizer.pipeline.run_pipeline")
    def test_main_passes_backend_to_config(self, mock_run):
        # Arrange / Act
        with patch("sys.argv", ["cli", "video.mp4", "--backend", "yolo+sam3"]):
            main()

        # Assert
        mock_run.assert_called_once()
        config = mock_run.call_args[1].get("config") or mock_run.call_args[0][1]
        assert config.detection_backend == "yolo+sam3"
