"""
Test per output.py — load_annotations_from_json e salvataggio output.

Testa le funzioni di I/O senza dipendenze da cv2/YOLO.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import (
    OutputPaths,
    PipelineContext,
    PipelineInputError,
    PipelineResult,
    VideoMeta,
)
from person_anonymizer.output import load_annotations_from_json, save_outputs


def _make_result(**kwargs):
    defaults = dict(
        annotations={},
        report_data={},
        review_stats={},
        method="pixelation",
        mode="auto",
        enable_debug=False,
        enable_report=False,
        ffmpeg_available=False,
        actual_refinement_passes=0,
        refinement_annotations_added=0,
    )
    defaults.update(kwargs)
    return PipelineResult(**defaults)


def _make_paths(tmp_path):
    return OutputPaths(
        output=str(tmp_path / "out.mp4"),
        temp_video=str(tmp_path / "temp.avi"),
        temp_debug=str(tmp_path / "temp_debug.avi"),
        debug=str(tmp_path / "debug.mp4"),
        report=str(tmp_path / "report.csv"),
        json=str(tmp_path / "annotations.json"),
    )


def _make_meta():
    return VideoMeta(fps=25.0, frame_w=640, frame_h=480, total_frames=10)


def _make_ctx(**kwargs):
    defaults = dict(
        input="test.mp4",
        mode="auto",
        method="pixelation",
        output=None,
        no_debug=True,
        no_report=True,
        review=None,
        normalize=False,
    )
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class TestLoadAnnotationsFromJson:
    """Test per load_annotations_from_json."""

    def test_file_not_found_raises_error(self, tmp_path):
        # Arrange
        config = PipelineConfig()
        fake_path = str(tmp_path / "nonexistent.json")

        # Act / Assert
        with pytest.raises(PipelineInputError, match="File JSON non trovato"):
            load_annotations_from_json(fake_path, config)

    def test_loads_empty_annotations(self, tmp_path):
        # Arrange
        config = PipelineConfig()
        json_file = tmp_path / "empty.json"
        json_file.write_text(json.dumps({"frames": {}}))

        # Act
        annotations, mode = load_annotations_from_json(str(json_file), config)

        # Assert
        assert annotations == {}
        assert mode == "manual"

    def test_loads_annotations_with_auto_polygons(self, tmp_path):
        # Arrange
        config = PipelineConfig()
        data = {
            "frames": {
                "0": {
                    "auto": [[[0, 0], [10, 0], [10, 50], [0, 50]]],
                    "manual": [],
                },
                "5": {
                    "auto": [],
                    "manual": [[[20, 20], [30, 20], [30, 30]]],
                },
            }
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(data))

        # Act
        annotations, mode = load_annotations_from_json(str(json_file), config)

        # Assert
        assert len(annotations) == 2
        assert 0 in annotations
        assert 5 in annotations
        assert len(annotations[0]["auto"]) == 1
        assert len(annotations[5]["manual"]) == 1
        assert mode == "manual"

    def test_computes_intensities_for_auto_polygons(self, tmp_path):
        # Arrange
        config = PipelineConfig(enable_adaptive_intensity=True)
        data = {
            "frames": {
                "0": {
                    "auto": [[[0, 0], [10, 0], [10, 80], [0, 80]]],
                    "manual": [],
                }
            }
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(data))

        # Act
        annotations, _ = load_annotations_from_json(str(json_file), config)

        # Assert
        assert len(annotations[0]["intensities"]) == 1
        assert annotations[0]["intensities"][0] > 0

    def test_converts_points_to_tuples(self, tmp_path):
        # Arrange
        config = PipelineConfig()
        data = {
            "frames": {
                "0": {
                    "auto": [[[1, 2], [3, 4], [5, 6]]],
                    "manual": [],
                }
            }
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(data))

        # Act
        annotations, _ = load_annotations_from_json(str(json_file), config)

        # Assert
        poly = annotations[0]["auto"][0]
        for pt in poly:
            assert isinstance(pt, tuple)

    def test_handles_missing_frames_key(self, tmp_path):
        # Arrange
        config = PipelineConfig()
        json_file = tmp_path / "no_frames.json"
        json_file.write_text(json.dumps({"metadata": "test"}))

        # Act
        annotations, mode = load_annotations_from_json(str(json_file), config)

        # Assert
        assert annotations == {}
        assert mode == "manual"


class TestSaveOutputsNoFfmpeg:
    """Test save_outputs senza ffmpeg (shutil.copy)."""

    def test_copies_temp_video_to_output_when_no_ffmpeg(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=False, enable_debug=False, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        # Crea il file temporaneo per poterlo copiare
        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.postprocessing.encode_with_audio"), \
             patch("person_anonymizer.postprocessing.encode_without_audio"), \
             patch("person_anonymizer.output.shutil.copy") as mock_copy:

            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        mock_copy.assert_called_once_with(paths.temp_video, paths.output)

    def test_debug_copied_when_enabled_and_no_ffmpeg(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=False, enable_debug=True, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        # Crea entrambi i file temporanei
        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)
        (tmp_path / "temp_debug.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy") as mock_copy:
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        # Due copie: temp_video → output, temp_debug → debug
        assert mock_copy.call_count == 2

    def test_debug_not_copied_when_temp_debug_missing(self, tmp_path):
        # Arrange — temp_debug non esiste
        result = _make_result(ffmpeg_available=False, enable_debug=True, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)
        # temp_debug.avi NON creato

        with patch("person_anonymizer.output.shutil.copy") as mock_copy:
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        # Solo 1 copia: il debug non c'è
        assert mock_copy.call_count == 1


class TestSaveOutputsWithFfmpeg:
    """Test save_outputs con ffmpeg disponibile."""

    def test_encode_with_audio_called_when_ffmpeg_available(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=True, enable_debug=False, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        with patch("person_anonymizer.postprocessing.encode_with_audio") as mock_encode, \
             patch("person_anonymizer.postprocessing.encode_without_audio"):

            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        mock_encode.assert_called_once_with(paths.temp_video, "input.mp4", paths.output)

    def test_debug_encoded_when_ffmpeg_and_debug_enabled(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=True, enable_debug=True, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        # Crea il temp_debug
        (tmp_path / "temp_debug.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.postprocessing.encode_with_audio"), \
             patch("person_anonymizer.postprocessing.encode_without_audio") as mock_enc_noaudio:

            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        mock_enc_noaudio.assert_called_once_with(paths.temp_debug, paths.debug)


class TestSaveOutputsReport:
    """Test scrittura CSV report."""

    def test_csv_report_written_when_enabled(self, tmp_path):
        # Arrange
        report_data = {
            0: {
                "frame_number": 0,
                "persons_detected": 2,
                "avg_confidence": 0.85,
                "min_confidence": 0.75,
                "max_confidence": 0.95,
                "motion_zones": 0,
                "sliding_window_hits": 0,
                "multiscale_hits": 0,
                "post_check_alerts": 0,
            }
        }
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=True,
            report_data=report_data
        )
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert os.path.isfile(paths.report)
        content = open(paths.report).read()
        assert "frame_number" in content
        assert "persons_detected" in content

    def test_csv_not_written_when_report_disabled(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=False, enable_debug=False, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert not os.path.isfile(paths.report)


class TestSaveOutputsJson:
    """Test scrittura JSON annotazioni."""

    def _annotation_result(self, mode="manual", normalize=False):
        annotations = {
            0: {
                "auto": [[(0, 0), (10, 0), (10, 10), (0, 10)]],
                "manual": [],
                "intensities": [5],
            }
        }
        return annotations

    def test_json_written_in_manual_mode(self, tmp_path):
        # Arrange
        annotations = self._annotation_result()
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=False,
            mode="manual", annotations=annotations
        )
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert os.path.isfile(paths.json)
        data = json.loads(open(paths.json).read())
        assert "frames" in data
        assert "schema_version" in data
        assert data["schema_version"] == "2.0"

    def test_json_not_written_in_auto_mode_without_normalize(self, tmp_path):
        # Arrange
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=False, mode="auto"
        )
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx(normalize=False)
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert not os.path.isfile(paths.json)

    def test_json_written_when_normalize_true(self, tmp_path):
        # Arrange
        annotations = self._annotation_result()
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=False,
            mode="auto", annotations=annotations
        )
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx(normalize=True)
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert os.path.isfile(paths.json)

    def test_json_contains_video_metadata(self, tmp_path):
        # Arrange
        annotations = self._annotation_result()
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=False,
            mode="manual", annotations=annotations
        )
        paths = _make_paths(tmp_path)
        meta = VideoMeta(fps=30.0, frame_w=1920, frame_h=1080, total_frames=100)
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        data = json.loads(open(paths.json).read())
        assert data["video"]["fps"] == 30.0
        assert data["video"]["total_frames"] == 100
        assert data["video"]["resolution"] == [1920, 1080]

    def test_json_frame_polygons_serialized_as_lists(self, tmp_path):
        # Arrange — tuple → devono essere serializzate come list
        annotations = {
            0: {
                "auto": [[(0, 0), (10, 0), (10, 10), (0, 10)]],
                "manual": [],
                "intensities": [5],
            }
        }
        result = _make_result(
            ffmpeg_available=False, enable_debug=False, enable_report=False,
            mode="manual", annotations=annotations
        )
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        data = json.loads(open(paths.json).read())
        frame_data = data["frames"]["0"]
        assert isinstance(frame_data["auto"][0][0], list)


class TestSaveOutputsCleanup:
    """Test rimozione file temporanei dopo salvataggio."""

    def test_temp_files_removed_after_save(self, tmp_path):
        # Arrange
        result = _make_result(ffmpeg_available=False, enable_debug=False, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        temp_file = tmp_path / "temp.avi"
        temp_file.write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"):
            save_outputs(ctx, result, config, "input.mp4", paths, meta)

        assert not os.path.isfile(paths.temp_video)

    def test_oserror_on_cleanup_does_not_raise(self, tmp_path):
        # Arrange — os.remove solleva OSError
        result = _make_result(ffmpeg_available=False, enable_debug=False, enable_report=False)
        paths = _make_paths(tmp_path)
        meta = _make_meta()
        ctx = _make_ctx()
        config = PipelineConfig()

        (tmp_path / "temp.avi").write_bytes(b"\x00" * 8)

        with patch("person_anonymizer.output.shutil.copy"), \
             patch("person_anonymizer.output.os.remove", side_effect=OSError("Permission denied")), \
             patch("person_anonymizer.output.os.path.exists", return_value=True):

            # Non deve sollevare
            save_outputs(ctx, result, config, "input.mp4", paths, meta)
