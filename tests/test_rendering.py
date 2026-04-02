"""
Test per rendering.py — render_video e compute_review_stats.

Usa frame sintetici numpy per evitare dipendenze da file video reali.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from person_anonymizer.rendering import render_video, compute_review_stats
from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext


class TestComputeReviewStats:
    """Verifica compute_review_stats con annotazioni note."""

    def test_no_changes(self):
        # Arrange
        original = {0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []}}
        reviewed = {0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []}}

        # Act
        stats = compute_review_stats(original, reviewed, 1)

        # Assert
        assert stats["added"] == 0
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 0

    def test_added_polygons(self):
        # Arrange
        original = {0: {"auto": [], "manual": []}}
        reviewed = {0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []}}

        # Act
        stats = compute_review_stats(original, reviewed, 1)

        # Assert
        assert stats["added"] == 1
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 1

    def test_removed_polygons(self):
        # Arrange
        original = {
            0: {"auto": [[(0, 0), (10, 0), (10, 10)], [(20, 20), (30, 20), (30, 30)]], "manual": []}
        }
        reviewed = {0: {"auto": [], "manual": []}}

        # Act
        stats = compute_review_stats(original, reviewed, 1)

        # Assert
        assert stats["added"] == 0
        assert stats["removed"] == 2
        assert stats["frames_modified"] == 1

    def test_mixed_changes_across_frames(self):
        # Arrange
        original = {
            0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": []},
            1: {"auto": [], "manual": []},
        }
        reviewed = {
            0: {"auto": [], "manual": []},  # rimosso 1
            1: {
                "auto": [[(5, 5), (15, 5), (15, 15)]],
                "manual": [[(20, 20), (30, 20), (30, 30)]],
            },  # aggiunti 2
        }

        # Act
        stats = compute_review_stats(original, reviewed, 2)

        # Assert
        assert stats["added"] == 2
        assert stats["removed"] == 1
        assert stats["frames_modified"] == 2
        assert stats["frames_reviewed"] == 2

    def test_empty_annotations(self):
        # Arrange
        original = {}
        reviewed = {}

        # Act
        stats = compute_review_stats(original, reviewed, 10)

        # Assert
        assert stats["added"] == 0
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 0


class TestRenderVideo:
    """Test render_video con video sintetico."""

    def _create_test_video(self, path, num_frames=5, w=20, h=20, fps=25.0):
        """Crea un video AVI con frame colorati sintetici."""
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for i in range(num_frames):
            frame = np.full((h, w, 3), fill_value=(i * 50) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_render_produces_output_file(self):
        # Arrange
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            self._create_test_video(input_path, num_frames=3)
            annotations = {
                0: {"auto": [], "manual": [], "intensities": []},
                1: {"auto": [], "manual": [], "intensities": []},
                2: {"auto": [], "manual": [], "intensities": []},
            }

            # Act
            render_video(
                input_path,
                output_path,
                annotations,
                25.0,
                20,
                20,
                "pixelation",
                FisheyeContext(),
                config,
            )

            # Assert
            assert os.path.exists(output_path)
            cap = cv2.VideoCapture(output_path)
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            cap.release()

    def test_render_applies_obscuring(self):
        # Arrange — frame con gradiente 20x20, poligono copre tutto il frame
        # Un frame uniforme (bianco) pixelato resta identico; il gradiente
        # garantisce che la pixelazione produca un risultato diverso.
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")

            fourcc = cv2.VideoWriter_fourcc(*"FFV1")
            writer = cv2.VideoWriter(input_path, fourcc, 25.0, (20, 20))
            gradient_frame = np.zeros((20, 20, 3), dtype=np.uint8)
            for row in range(20):
                gradient_frame[row, :, :] = row * 12  # gradiente verticale 0-228
            writer.write(gradient_frame)
            writer.release()

            annotations = {
                0: {
                    "auto": [[(0, 0), (20, 0), (20, 20), (0, 20)]],
                    "manual": [],
                    "intensities": [10],
                },
            }

            # Act
            render_video(
                input_path,
                output_path,
                annotations,
                25.0,
                20,
                20,
                "pixelation",
                FisheyeContext(),
                config,
            )

            # Assert — il frame di output non deve essere identico all'input
            cap = cv2.VideoCapture(output_path)
            ret, out_frame = cap.read()
            cap.release()
            assert ret
            assert not np.array_equal(out_frame, gradient_frame)

    def test_render_with_debug_output(self):
        # Arrange
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.avi")
            output_path = os.path.join(tmpdir, "output.avi")
            debug_path = os.path.join(tmpdir, "debug.avi")
            self._create_test_video(input_path, num_frames=2)
            annotations = {
                0: {
                    "auto": [[(0, 0), (10, 0), (10, 10), (0, 10)]],
                    "manual": [],
                    "intensities": [10],
                },
                1: {"auto": [], "manual": [], "intensities": []},
            }

            # Act
            render_video(
                input_path,
                output_path,
                annotations,
                25.0,
                20,
                20,
                "pixelation",
                FisheyeContext(),
                config,
                debug_path=debug_path,
            )

            # Assert
            assert os.path.exists(debug_path)
            cap = cv2.VideoCapture(debug_path)
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
            cap.release()
