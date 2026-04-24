"""
Test per web/pipeline_runner.py — PipelineRunner e _build_config.

Copre: start, stop, get_status, _run (successo, PipelineError, Exception generico,
config invalida, yolo path traversal, stop_event), _build_config.
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from person_anonymizer.web.pipeline_runner import PipelineRunner, _build_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sse():
    sse = MagicMock()
    return sse


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "outputs"


@pytest.fixture
def runner(mock_sse, tmp_output):
    tmp_output.mkdir()
    return PipelineRunner(sse=mock_sse, output_dir=tmp_output)


MINIMAL_CONFIG = {
    "operation_mode": "auto",
    "anonymization_method": "pixelation",
}


# ---------------------------------------------------------------------------
# TestBuildConfig
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """Verifica _build_config — costruzione PipelineConfig da dict web."""

    def test_valid_config_returns_pipeline_config(self):
        # Arrange
        from person_anonymizer.config import PipelineConfig

        # Act
        config = _build_config(MINIMAL_CONFIG)

        # Assert
        assert isinstance(config, PipelineConfig)

    def test_invalid_config_raises_value_error(self):
        # Arrange — detection_confidence fuori range
        bad_config = {"detection_confidence": 99.9}

        # Act / Assert
        with pytest.raises(ValueError):
            _build_config(bad_config)

    def test_quality_clahe_grid_list_converted_to_tuple(self):
        # Arrange — la web form invia liste, non tuple
        config_dict = {**MINIMAL_CONFIG, "quality_clahe_grid": [4, 4]}

        # Act
        config = _build_config(config_dict)

        # Assert
        assert isinstance(config.quality_clahe_grid, tuple)
        assert config.quality_clahe_grid == (4, 4)

    def test_unknown_fields_ignored(self):
        # Arrange — campo sconosciuto non in _ALLOWED_FIELDS
        config_dict = {**MINIMAL_CONFIG, "nonexistent_field": "value"}

        # Act — non deve sollevare eccezioni
        config = _build_config(config_dict)

        # Assert
        assert not hasattr(config, "nonexistent_field")


# ---------------------------------------------------------------------------
# TestPipelineRunnerStart
# ---------------------------------------------------------------------------


class TestPipelineRunnerStart:
    """Verifica start() di PipelineRunner."""

    def test_start_returns_true_when_idle(self, runner):
        # Arrange
        with patch.object(runner, "_run"):  # evita thread reale
            with patch("threading.Thread") as mock_thread_cls:
                mock_thread = MagicMock()
                mock_thread.is_alive.return_value = False
                mock_thread_cls.return_value = mock_thread

                # Act
                success, msg = runner.start(
                    job_id="aabbccddeeff",
                    video_path="/fake/video.mp4",
                    config_dict=MINIMAL_CONFIG,
                )

        # Assert
        assert success is True
        assert "avviata" in msg.lower()

    def test_start_returns_false_when_already_running(self, runner):
        # Arrange — simula thread già vivo
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runner._thread = mock_thread

        # Act
        success, msg = runner.start(
            job_id="aabbccddeeff",
            video_path="/fake/video.mp4",
            config_dict=MINIMAL_CONFIG,
        )

        # Assert
        assert success is False
        assert "già in esecuzione" in msg

    def test_start_sets_current_job_id(self, runner):
        # Arrange
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_cls.return_value = mock_thread

            # Act
            runner.start(
                job_id="testjobid123",
                video_path="/fake/video.mp4",
                config_dict=MINIMAL_CONFIG,
            )

        # Assert
        assert runner._current_job_id == "testjobid123"

    def test_start_clears_stop_event(self, runner):
        # Arrange — imposta lo stop event prima dello start
        runner._stop_event.set()

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_cls.return_value = mock_thread

            # Act
            runner.start(
                job_id="aabbccddeeff",
                video_path="/fake/video.mp4",
                config_dict=MINIMAL_CONFIG,
            )

        # Assert — stop event deve essere cleared
        assert not runner._stop_event.is_set()


# ---------------------------------------------------------------------------
# TestPipelineRunnerStop
# ---------------------------------------------------------------------------


class TestPipelineRunnerStop:
    """Verifica stop() di PipelineRunner."""

    def test_stop_returns_false_when_no_thread(self, runner):
        # Arrange — nessun thread
        runner._thread = None

        # Act
        result = runner.stop()

        # Assert
        assert result is False

    def test_stop_returns_false_when_thread_not_alive(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        runner._thread = mock_thread

        # Act
        result = runner.stop()

        # Assert
        assert result is False

    def test_stop_returns_true_and_sets_event_when_running(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runner._thread = mock_thread
        runner._current_job_id = "aabbccddeeff"

        # Act
        result = runner.stop()

        # Assert
        assert result is True
        assert runner._stop_event.is_set()

    def test_stop_returns_false_for_wrong_job_id(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runner._thread = mock_thread
        runner._current_job_id = "aabbccddeeff"

        # Act — job_id diverso
        result = runner.stop(job_id="111111111111")

        # Assert
        assert result is False

    def test_stop_without_job_id_stops_any_running(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runner._thread = mock_thread
        runner._current_job_id = "aabbccddeeff"

        # Act — job_id=None deve fermare comunque
        result = runner.stop(job_id=None)

        # Assert
        assert result is True


# ---------------------------------------------------------------------------
# TestPipelineRunnerGetStatus
# ---------------------------------------------------------------------------


class TestPipelineRunnerGetStatus:
    """Verifica get_status() di PipelineRunner."""

    def test_get_status_not_running_initially(self, runner):
        # Arrange / Act
        status = runner.get_status()

        # Assert
        assert status["running"] is False
        assert status["job_id"] is None

    def test_get_status_running_when_thread_alive(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runner._thread = mock_thread
        runner._current_job_id = "aabbccddeeff"

        # Act
        status = runner.get_status()

        # Assert
        assert status["running"] is True
        assert status["job_id"] == "aabbccddeeff"

    def test_get_status_job_id_none_when_not_running(self, runner):
        # Arrange
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        runner._thread = mock_thread
        runner._current_job_id = "aabbccddeeff"

        # Act
        status = runner.get_status()

        # Assert
        assert status["running"] is False
        assert status["job_id"] is None


# ---------------------------------------------------------------------------
# TestPipelineRunnerRun
# ---------------------------------------------------------------------------


class TestPipelineRunnerRun:
    """Verifica _run() — esecuzione effettiva nella pipeline."""

    def _run_sync(self, runner, job_id, video_path, config_dict, review_json=None):
        """Esegue _run in un thread e attende completamento."""
        t = threading.Thread(
            target=runner._run,
            args=(job_id, video_path, config_dict, review_json),
        )
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "_run non è terminato entro il timeout"

    def test_run_emits_started_event(self, runner, mock_sse, tmp_output):
        # Arrange
        with patch("person_anonymizer.web.pipeline_runner._build_config") as mock_cfg, patch(
            "person_anonymizer.pipeline.run_pipeline"
        ), patch("person_anonymizer.web.pipeline_runner.TqdmCapture"), patch(
            "person_anonymizer.web.pipeline_runner.StdoutCapture"
        ), patch(
            "person_anonymizer.web.pipeline_runner.Path"
        ) as mock_path_cls:
            from person_anonymizer.config import PipelineConfig

            mock_cfg.return_value = PipelineConfig(operation_mode="auto")

            # Evita che Path(__file__) fallisca
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.return_value = mock_path_instance
            mock_path_instance.parent = mock_path_instance
            mock_path_instance.__truediv__ = lambda s, o: mock_path_instance
            mock_path_instance.exists.return_value = False
            mock_path_instance.__str__ = lambda s: "/fake/pa_dir"
            mock_path_cls.return_value = mock_path_instance

            # Act
            self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", MINIMAL_CONFIG)

        # Assert — deve aver emesso "started"
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "started" in events

    def test_run_emits_error_on_invalid_config(self, runner, mock_sse):
        # Arrange — config invalida causa ValueError in _build_config
        bad_config = {"detection_confidence": 999.0}

        # Act
        self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", bad_config)

        # Assert
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "error" in events

    def test_run_emits_error_on_invalid_config_calls_sse_close(self, runner, mock_sse):
        # Arrange
        bad_config = {"detection_confidence": 999.0}

        # Act
        self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", bad_config)

        # Assert — SSE deve essere chiuso
        mock_sse.close.assert_called_with("aabbccddeeff")

    def test_run_clears_job_id_after_completion(self, runner, mock_sse):
        # Arrange — config invalida per evitare setup complesso
        bad_config = {"detection_confidence": 999.0}

        # Act
        self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", bad_config)

        # Assert — job_id viene azzerato
        with runner._lock:
            assert runner._current_job_id is None

    def test_run_emits_pipeline_error_event(self, runner, mock_sse):
        # Arrange
        from person_anonymizer.models import PipelineError

        with patch("person_anonymizer.web.pipeline_runner._build_config") as mock_cfg, patch(
            "person_anonymizer.pipeline.run_pipeline",
            side_effect=PipelineError("pipeline failed"),
        ), patch("person_anonymizer.web.pipeline_runner.TqdmCapture"), patch(
            "person_anonymizer.web.pipeline_runner.StdoutCapture"
        ):
            from person_anonymizer.config import PipelineConfig

            cfg = PipelineConfig(operation_mode="auto")
            mock_cfg.return_value = cfg

            import person_anonymizer as pa

            pa_dir = Path(pa.__file__).resolve().parent
            with patch("person_anonymizer.web.pipeline_runner.Path") as mock_path_cls:
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = mock_path_instance
                mock_path_instance.parent = MagicMock(
                    __truediv__=lambda s, o: mock_path_instance,
                    resolve=lambda: mock_path_instance,
                    __str__=lambda s: str(pa_dir),
                )
                mock_path_instance.__truediv__ = lambda s, o: mock_path_instance
                mock_path_instance.exists.return_value = False
                mock_path_instance.stem = "video"
                mock_path_instance.__str__ = lambda s: str(pa_dir)
                mock_path_instance.mkdir = MagicMock()
                mock_path_instance.iterdir = MagicMock(return_value=[])
                mock_path_cls.return_value = mock_path_instance

                self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", MINIMAL_CONFIG)

        # Assert
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "error" in events

    def test_run_emits_stopped_event_when_stop_requested(self, runner, mock_sse, tmp_output):
        # Arrange — stop_event settato, ma con config valida per arrivare al finally
        runner._stop_event.set()

        with patch("person_anonymizer.web.pipeline_runner._build_config") as mock_cfg, patch(
            "person_anonymizer.pipeline.run_pipeline"
        ), patch("person_anonymizer.web.pipeline_runner.TqdmCapture"), patch(
            "person_anonymizer.web.pipeline_runner.StdoutCapture"
        ), patch(
            "person_anonymizer.web.pipeline_runner.Path"
        ) as mock_path_cls:
            from person_anonymizer.config import PipelineConfig

            mock_cfg.return_value = PipelineConfig(operation_mode="auto")
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.return_value = mock_path_instance
            mock_path_instance.parent = mock_path_instance
            mock_path_instance.__truediv__ = lambda s, o: mock_path_instance
            mock_path_instance.exists.return_value = False
            mock_path_instance.__str__ = lambda s: "/fake/pa_dir"
            mock_path_instance.mkdir = MagicMock()
            mock_path_instance.iterdir = MagicMock(return_value=[])
            mock_path_instance.stem = "video"
            mock_path_cls.return_value = mock_path_instance

            # Act
            self._run_sync(runner, "aabbccddeeff", "/fake/video.mp4", MINIMAL_CONFIG)

        # Assert — nel finally, stop_event.is_set() → emette "stopped"
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "stopped" in events
