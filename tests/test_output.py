"""
Test per output.py — load_annotations_from_json e salvataggio output.

Testa le funzioni di I/O senza dipendenze da cv2/YOLO.
"""

import json
import os
import pytest
from unittest.mock import patch

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import PipelineInputError
from person_anonymizer.output import load_annotations_from_json


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
