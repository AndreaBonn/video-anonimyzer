"""
Test per web/output_capture.py — TqdmCapture e StdoutCapture.

Copre: install/uninstall, write/flush, context manager, sanitizzazione path,
phase_label detection, buffer residuo, rate-limit tqdm.
"""

import sys
from unittest.mock import MagicMock

import pytest

from person_anonymizer.web.output_capture import StdoutCapture, TqdmCapture

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sse():
    sse = MagicMock()
    return sse


# ---------------------------------------------------------------------------
# TestStdoutCaptureInstallUninstall
# ---------------------------------------------------------------------------


class TestStdoutCaptureInstallUninstall:
    """Verifica install/uninstall di StdoutCapture."""

    def test_install_replaces_stdout(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        original = sys.stdout

        # Act
        capture.install()

        # Assert
        assert sys.stdout is capture

        # Cleanup
        capture.uninstall()
        assert sys.stdout is original

    def test_uninstall_restores_stdout(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        original = sys.stdout
        capture.install()

        # Act
        capture.uninstall()

        # Assert
        assert sys.stdout is original

    def test_uninstall_without_install_does_not_raise(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")

        # Act / Assert — _original è None, non deve crashare
        capture.uninstall()


# ---------------------------------------------------------------------------
# TestStdoutCaptureWrite
# ---------------------------------------------------------------------------


class TestStdoutCaptureWrite:
    """Verifica il comportamento di write()."""

    def test_write_complete_line_emits_log_event(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()

        # Act
        capture.write("Hello world\n")
        capture.uninstall()

        # Assert
        mock_sse.emit.assert_called_with("job1", "log", {"message": "Hello world"})

    def test_write_partial_line_buffered_not_emitted(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act — nessun \n → nessun emit
        capture.write("partial")

        # Assert
        mock_sse.emit.assert_not_called()

        capture.uninstall()

    def test_write_empty_line_not_emitted(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act — riga vuota
        capture.write("\n")

        # Assert — riga vuota dopo strip non emessa
        log_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "log"]
        assert len(log_calls) == 0

        capture.uninstall()

    def test_write_multiple_lines_emits_each(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act
        capture.write("line1\nline2\n")
        capture.uninstall()

        # Assert
        log_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "log"]
        messages = [c[0][2]["message"] for c in log_calls]
        assert "line1" in messages
        assert "line2" in messages

    def test_write_phase_label_emits_phase_label_event(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act
        capture.write("[FASE 2/5] Detection\n")
        capture.uninstall()

        # Assert — deve emettere sia phase_label che log
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "phase_label" in events
        assert "log" in events

    def test_write_phase_label_correct_phase_number(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act
        capture.write("[FASE 3/5] Refinement\n")
        capture.uninstall()

        # Assert
        phase_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "phase_label"]
        assert len(phase_calls) == 1
        assert phase_calls[0][0][2]["phase"] == 3

    def test_write_forwards_to_original_stdout(self, mock_sse):
        # Arrange
        mock_original = MagicMock()
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture._original = mock_original

        # Act
        capture.write("test\n")

        # Assert
        mock_original.write.assert_called_once_with("test\n")


# ---------------------------------------------------------------------------
# TestStdoutCaptureSanitize
# ---------------------------------------------------------------------------


class TestStdoutCaptureSanitize:
    """Verifica sanitizzazione dei path assoluti."""

    def test_sanitize_removes_upload_path(self):
        # Arrange
        msg = "Processing /home/user/uploads/video.mp4"

        # Act
        result = StdoutCapture._sanitize_message(msg)

        # Assert
        assert "[FILE]" in result
        assert "/home/user/uploads/video.mp4" not in result

    def test_sanitize_removes_outputs_path(self):
        # Arrange
        msg = "Saved to /tmp/outputs/result.json"

        # Act
        result = StdoutCapture._sanitize_message(msg)

        # Assert
        assert "[FILE]" in result

    def test_sanitize_no_path_unchanged(self):
        # Arrange
        msg = "Processing frame 10 of 100"

        # Act
        result = StdoutCapture._sanitize_message(msg)

        # Assert
        assert result == msg

    def test_uninstall_emits_residual_buffer(self, mock_sse):
        # Arrange — testo senza \n finale rimane nel buffer
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()
        capture.write("residual text")  # no \n

        # Act
        capture.uninstall()

        # Assert — uninstall deve svuotare il buffer
        log_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "log"]
        assert len(log_calls) == 1
        assert log_calls[0][0][2]["message"] == "residual text"


# ---------------------------------------------------------------------------
# TestStdoutCaptureFlush
# ---------------------------------------------------------------------------


class TestStdoutCaptureFlush:
    """Verifica flush() di StdoutCapture."""

    def test_flush_calls_original_flush(self, mock_sse):
        # Arrange
        mock_original = MagicMock()
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        capture._original = mock_original

        # Act
        capture.flush()

        # Assert
        mock_original.flush.assert_called_once()

    def test_flush_without_original_does_not_raise(self, mock_sse):
        # Arrange
        capture = StdoutCapture(sse=mock_sse, job_id="job1")
        # _original è None di default

        # Act / Assert
        capture.flush()  # non deve sollevare eccezioni


# ---------------------------------------------------------------------------
# TestTqdmCaptureInstallUninstall
# ---------------------------------------------------------------------------


class TestTqdmCaptureInstallUninstall:
    """Verifica install/uninstall di TqdmCapture."""

    def test_install_stores_original_tqdm(self, mock_sse):
        # Arrange
        import tqdm as tqdm_module

        original = tqdm_module.tqdm
        capture = TqdmCapture(sse=mock_sse, job_id="job1")

        # Act
        capture.install()

        # Assert
        assert capture._original_tqdm is original

        # Cleanup
        capture.uninstall()

    def test_install_patches_tqdm_module(self, mock_sse):
        # Arrange
        import tqdm as tqdm_module

        original = tqdm_module.tqdm
        capture = TqdmCapture(sse=mock_sse, job_id="job1")

        # Act
        capture.install()

        # Assert — tqdm.tqdm è stato sostituito con PatchedTqdm
        assert tqdm_module.tqdm is not original

        # Cleanup
        capture.uninstall()

    def test_uninstall_restores_tqdm(self, mock_sse):
        # Arrange
        import tqdm as tqdm_module

        original = tqdm_module.tqdm
        capture = TqdmCapture(sse=mock_sse, job_id="job1")
        capture.install()

        # Act
        capture.uninstall()

        # Assert
        assert tqdm_module.tqdm is original

    def test_uninstall_without_install_does_not_raise(self, mock_sse):
        # Arrange
        capture = TqdmCapture(sse=mock_sse, job_id="job1")

        # Act / Assert — _original_tqdm è None, non deve crashare
        capture.uninstall()

    def test_install_patches_pipeline_stages(self, mock_sse):
        # Arrange
        import tqdm as tqdm_module

        import person_anonymizer.pipeline_stages as pa_stages

        original = tqdm_module.tqdm
        capture = TqdmCapture(sse=mock_sse, job_id="job1")

        # Act
        capture.install()

        # Assert — anche pipeline_stages.tqdm è stato patched
        assert pa_stages.tqdm is not original

        # Cleanup
        capture.uninstall()


# ---------------------------------------------------------------------------
# TestPatchedTqdmBehavior
# ---------------------------------------------------------------------------


class TestPatchedTqdmBehavior:
    """Verifica comportamento di PatchedTqdm (init, update, close)."""

    def test_patched_tqdm_init_emits_phase_event(self, mock_sse):
        # Arrange
        capture = TqdmCapture(sse=mock_sse, job_id="job1")
        capture.install()
        mock_sse.emit.reset_mock()

        # Act
        import tqdm as tqdm_module

        bar = tqdm_module.tqdm(total=10, desc="Test phase")
        bar.close()

        # Assert — deve emettere "phase" event all'init
        events = [c[0][1] for c in mock_sse.emit.call_args_list]
        assert "phase" in events

        capture.uninstall()

    def test_patched_tqdm_close_emits_progress_event(self, mock_sse):
        # Arrange
        capture = TqdmCapture(sse=mock_sse, job_id="job1")
        capture.install()

        import tqdm as tqdm_module

        bar = tqdm_module.tqdm(total=5, desc="Close test")
        mock_sse.emit.reset_mock()

        # Act
        bar.close()

        # Assert — close emette "progress" con current==total
        progress_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "progress"]
        assert len(progress_calls) >= 1
        last = progress_calls[-1][0][2]
        assert last["current"] == last["total"]

        capture.uninstall()

    def test_patched_tqdm_update_emits_progress_after_interval(self, mock_sse):
        # Arrange
        capture = TqdmCapture(sse=mock_sse, job_id="job1")
        capture.install()


        import tqdm as tqdm_module

        bar = tqdm_module.tqdm(total=100, desc="Progress test")
        # Forza il timestamp indietro per superare il rate-limit di 0.25s
        bar._last_emit = 0
        mock_sse.emit.reset_mock()

        # Act
        bar.update(10)
        bar.close()

        # Assert
        progress_calls = [c for c in mock_sse.emit.call_args_list if c[0][1] == "progress"]
        assert len(progress_calls) >= 1

        capture.uninstall()
