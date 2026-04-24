"""
Test aggiuntivi per rendering.py — coverage righe mancanti.

Copre: render_video con stop_event (riga 86-87), VideoWriter non apribile
(righe 71-79), frame corrotti (righe 91-95), compute_review_stats casi bordo.
"""

import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext
from person_anonymizer.rendering import compute_review_stats, render_video


# ============================================================
# render_video — righe 71-95, 111-132
# ============================================================


class TestRenderVideoExtended:
    """Casi limite di render_video."""

    def _create_test_video(self, path, num_frames=3, w=20, h=20, fps=25.0):
        """Crea un video AVI sintetico FFV1 con pattern non uniforme."""
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for i in range(num_frames):
            # Pattern a scacchiera per evitare frame uniformi (blur su uniforme = identico)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[::2, ::2] = 255  # pixel bianchi alternati
            frame[1::2, 1::2] = 128  # pixel grigi alternati
            frame += i * 10  # variazione tra frame
            writer.write(frame)
        writer.release()

    def test_stop_event_halts_rendering(self):
        # Arrange — stop_event già settato → nessun frame renderizzato
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=5)

            stop_event = threading.Event()
            stop_event.set()  # già interrotto

            # Act
            render_video(
                input_path, output_path,
                annotations={},
                fps=25.0, frame_w=20, frame_h=20,
                method="pixelation",
                fisheye=FisheyeContext(),
                config=config,
                stop_event=stop_event,
            )

            # Assert — output creato ma con 0 frame (stop immediato)
            assert os.path.exists(output_path)
            cap = cv2.VideoCapture(output_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert frame_count == 0

    def test_output_writer_not_openable_raises_runtime_error(self):
        # Arrange — mock VideoWriter che ritorna isOpened()=False
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            self._create_test_video(input_path, num_frames=2)

            fake_writer = MagicMock()
            fake_writer.isOpened.return_value = False

            with patch("person_anonymizer.rendering.cv2.VideoWriter", return_value=fake_writer):
                # Act / Assert
                with pytest.raises(RuntimeError, match="Impossibile aprire VideoWriter"):
                    render_video(
                        input_path, "/nonexistent/output.avi",
                        annotations={},
                        fps=25.0, frame_w=20, frame_h=20,
                        method="pixelation",
                        fisheye=FisheyeContext(),
                        config=config,
                    )

    def test_debug_writer_not_openable_raises_runtime_error(self):
        # Arrange — primo VideoWriter OK (output), secondo KO (debug)
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=2)

            writer_ok = MagicMock()
            writer_ok.isOpened.return_value = True

            writer_fail = MagicMock()
            writer_fail.isOpened.return_value = False

            writer_calls = [writer_ok, writer_fail]
            with patch("person_anonymizer.rendering.cv2.VideoWriter", side_effect=writer_calls):
                with pytest.raises(RuntimeError, match="Impossibile aprire VideoWriter"):
                    render_video(
                        input_path, output_path,
                        annotations={},
                        fps=25.0, frame_w=20, frame_h=20,
                        method="pixelation",
                        fisheye=FisheyeContext(),
                        config=config,
                        debug_path="/nonexistent/debug.avi",
                    )

    def test_blur_method_with_odd_intensity_modifies_frame(self):
        # Arrange — metodo blur con intensità dispari, area piena
        config = PipelineConfig(anonymization_intensity=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=1)
            annotations = {
                0: {
                    "auto": [[(0, 0), (20, 0), (20, 20), (0, 20)]],
                    "manual": [],
                    "intensities": [11],
                }
            }

            # Act
            render_video(
                input_path, output_path,
                annotations=annotations,
                fps=25.0, frame_w=20, frame_h=20,
                method="blur",
                fisheye=FisheyeContext(),
                config=config,
            )

            # Assert — output ha 1 frame e il blur ha modificato i pixel
            cap_out = cv2.VideoCapture(output_path)
            assert int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
            ret, out_frame = cap_out.read()
            cap_out.release()
            assert ret is True

            cap_in = cv2.VideoCapture(input_path)
            _, in_frame = cap_in.read()
            cap_in.release()
            # Il blur modifica i pixel — i frame non devono essere identici
            assert not np.array_equal(in_frame, out_frame)

    def test_blur_method_with_even_intensity_still_applies_blur(self):
        # Arrange — intensità pari viene aggiustata a dispari internamente
        config = PipelineConfig(anonymization_intensity=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=1)
            annotations = {
                0: {
                    "auto": [[(0, 0), (20, 0), (20, 20), (0, 20)]],
                    "manual": [],
                    "intensities": [10],
                }
            }

            # Act
            render_video(
                input_path, output_path,
                annotations=annotations,
                fps=25.0, frame_w=20, frame_h=20,
                method="blur",
                fisheye=FisheyeContext(),
                config=config,
            )

            # Assert — il blur viene applicato anche con intensità pari
            cap_out = cv2.VideoCapture(output_path)
            assert int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
            ret, out_frame = cap_out.read()
            cap_out.release()
            assert ret is True

            cap_in = cv2.VideoCapture(input_path)
            _, in_frame = cap_in.read()
            cap_in.release()
            assert not np.array_equal(in_frame, out_frame)

    def test_manual_polygons_rendered_modifies_pixels(self):
        # Arrange — poligoni manual coprono tutto il frame → pixel modificati
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=1)
            annotations = {
                0: {
                    "auto": [],
                    "manual": [[(0, 0), (20, 0), (20, 20), (0, 20)]],
                    "intensities": [],
                }
            }

            # Act
            render_video(
                input_path, output_path,
                annotations=annotations,
                fps=25.0, frame_w=20, frame_h=20,
                method="pixelation",
                fisheye=FisheyeContext(),
                config=config,
            )

            # Assert — manual polygon anonimizza i pixel
            cap_out = cv2.VideoCapture(output_path)
            assert int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
            ret, out_frame = cap_out.read()
            cap_out.release()
            assert ret is True

            cap_in = cv2.VideoCapture(input_path)
            _, in_frame = cap_in.read()
            cap_in.release()
            assert not np.array_equal(in_frame, out_frame)

    def test_adaptive_intensity_applied_per_polygon(self):
        # Arrange — adaptive_intensity con 2 poligoni e intensità diverse
        config = PipelineConfig(
            enable_adaptive_intensity=True,
            anonymization_intensity=10,
            adaptive_reference_height=80,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=1)
            annotations = {
                0: {
                    "auto": [
                        [(0, 0), (10, 0), (10, 10), (0, 10)],
                        [(10, 10), (20, 10), (20, 20), (10, 20)],
                    ],
                    "manual": [],
                    "intensities": [5, 15],
                }
            }

            # Act
            render_video(
                input_path, output_path,
                annotations=annotations,
                fps=25.0, frame_w=20, frame_h=20,
                method="pixelation",
                fisheye=FisheyeContext(),
                config=config,
            )

            # Assert — output ha 1 frame, pixel modificati dall'anonimizzazione
            cap_out = cv2.VideoCapture(output_path)
            assert int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
            ret, out_frame = cap_out.read()
            cap_out.release()
            assert ret is True

            cap_in = cv2.VideoCapture(input_path)
            _, in_frame = cap_in.read()
            cap_in.release()
            assert not np.array_equal(in_frame, out_frame)


# ============================================================
# compute_review_stats — casi bordo aggiuntivi
# ============================================================


class TestComputeReviewStatsExtended:
    """Casi limite aggiuntivi per compute_review_stats."""

    def test_frame_only_in_reviewed_counts_as_added(self):
        # Arrange — frame presente solo in reviewed (non in original)
        original = {}
        reviewed = {
            0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []}
        }

        # Act
        stats = compute_review_stats(original, reviewed, total_frames=1)

        # Assert — 1 poligono aggiunto
        assert stats["added"] == 1
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 1

    def test_frame_only_in_original_counts_as_removed(self):
        # Arrange — frame presente solo in original
        original = {
            3: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []}
        }
        reviewed = {}

        # Act
        stats = compute_review_stats(original, reviewed, total_frames=5)

        # Assert
        assert stats["added"] == 0
        assert stats["removed"] == 1
        assert stats["frames_modified"] == 1
        assert stats["frames_reviewed"] == 5

    def test_frames_reviewed_matches_total_frames(self):
        # Arrange
        original = {}
        reviewed = {}

        # Act
        stats = compute_review_stats(original, reviewed, total_frames=42)

        # Assert — frames_reviewed sempre uguale a total_frames passato
        assert stats["frames_reviewed"] == 42

    def test_manual_polygons_counted_in_diff(self):
        # Arrange — poligono aggiunto come manual
        original = {0: {"auto": [], "manual": []}}
        reviewed = {0: {"auto": [], "manual": [[(0, 0), (10, 0), (10, 10)]]}}

        # Act
        stats = compute_review_stats(original, reviewed, total_frames=1)

        # Assert
        assert stats["added"] == 1
