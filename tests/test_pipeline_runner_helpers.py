"""Tests for the pure helpers and the pre-start failure path of web/pipeline_runner.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.web.pipeline_runner import (
    PipelineRunner,
    list_job_outputs,
    resolve_yolo_model,
)


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "pkg"
    directory.mkdir()
    return directory


class TestResolveYoloModel:
    def test_parent_traversal_outside_package_returns_none(
        self, tmp_path: Path, package_dir: Path
    ) -> None:
        (tmp_path / "x.pt").write_bytes(b"w")

        assert resolve_yolo_model(model_name="../x.pt", package_dir=package_dir) is None

    def test_sibling_dir_with_common_prefix_returns_none(
        self, tmp_path: Path, package_dir: Path
    ) -> None:
        evil = tmp_path / "pkg_evil"
        evil.mkdir()
        (evil / "m.pt").write_bytes(b"w")

        assert resolve_yolo_model(model_name="../pkg_evil/m.pt", package_dir=package_dir) is None

    def test_absolute_path_outside_package_returns_none(
        self, tmp_path: Path, package_dir: Path
    ) -> None:
        outside = tmp_path / "m.pt"
        outside.write_bytes(b"w")

        assert resolve_yolo_model(model_name=str(outside), package_dir=package_dir) is None

    def test_dotdot_that_stays_inside_returns_resolved_path(self, package_dir: Path) -> None:
        (package_dir / "sub").mkdir()
        (package_dir / "m.pt").write_bytes(b"w")

        result = resolve_yolo_model(model_name="sub/../m.pt", package_dir=package_dir)

        assert result == str((package_dir / "m.pt").resolve())

    def test_existing_file_returns_absolute_path(self, package_dir: Path) -> None:
        (package_dir / "yolo.pt").write_bytes(b"w")

        result = resolve_yolo_model(model_name="yolo.pt", package_dir=package_dir)

        assert result == str((package_dir / "yolo.pt").resolve())
        assert Path(result).is_absolute()

    def test_missing_file_returns_name_unchanged(self, package_dir: Path) -> None:
        assert resolve_yolo_model(model_name="yolov8x.pt", package_dir=package_dir) == "yolov8x.pt"


class TestListJobOutputs:
    def test_files_sorted_by_name_with_size_rounded_to_two_decimals(self, tmp_path: Path) -> None:
        (tmp_path / "z_video.mp4").write_bytes(b"x" * 1_048_576)
        (tmp_path / "a_report.csv").write_bytes(b"x" * 16_384)

        assert list_job_outputs(tmp_path) == [
            {"name": "a_report.csv", "size_mb": 0.02},
            {"name": "z_video.mp4", "size_mb": 1.0},
        ]

    def test_subdirectories_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "nested").mkdir()
        (tmp_path / "file.txt").write_bytes(b"x")

        assert [entry["name"] for entry in list_job_outputs(tmp_path)] == ["file.txt"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_job_outputs(tmp_path) == []


class TestFailBeforeStart:
    def test_unauthorized_yolo_model_emits_error_closes_and_resets_job_id(
        self, tmp_path: Path
    ) -> None:
        sse = MagicMock()
        runner = PipelineRunner(sse=sse, output_dir=tmp_path)
        runner._current_job_id = "job1"

        # The web validator whitelists model names, so the guard is only reachable
        # through a config that bypasses it
        config = PipelineConfig(yolo_model="../../escape.pt")

        with patch("person_anonymizer.web.pipeline_runner._build_config", return_value=config):
            runner._run(
                job_id="job1", video_path="/input/video.mp4", config_dict={}, review_json=None
            )

        sse.emit.assert_called_once_with(
            "job1", "error", {"job_id": "job1", "message": "Percorso modello YOLO non autorizzato"}
        )
        sse.close.assert_called_once_with("job1")
        assert runner._current_job_id is None
        assert not (tmp_path / "job1").exists()
