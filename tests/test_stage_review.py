"""
Test per stage_review.py — run_manual_review_stage.

Mock di manual_reviewer, web review state, SSE manager.
"""

from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext, PipelineContext


def _make_ctx(**kwargs):
    defaults = dict(
        input="test.mp4",
        mode="manual",
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


def _make_annotations(n_frames=3):
    return {
        i: {"auto": [[(0, 0), (10, 0), (10, 10), (0, 10)]], "manual": [], "intensities": [5]}
        for i in range(n_frames)
    }


class TestRunManualReviewStageCLI:
    """Test revisione manuale tramite CLI (review_state=None)."""

    def test_cli_review_calls_run_manual_review(self):
        # Arrange
        ctx = _make_ctx(review_state=None)
        annotations = _make_annotations()
        config = PipelineConfig()
        fe = FisheyeContext()
        review_stats = {"added": 1, "removed": 0, "frames_modified": 1, "frames_reviewed": 3}

        with patch("person_anonymizer.manual_reviewer.run_manual_review",
                   return_value=(annotations, review_stats)) as mock_review:

            from person_anonymizer.stage_review import run_manual_review_stage
            result_ann, result_stats = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fe,
            )

        mock_review.assert_called_once_with("test.mp4", annotations, config, fe)

    def test_cli_review_returns_annotations_from_reviewer(self):
        # Arrange
        ctx = _make_ctx(review_state=None)
        input_annotations = _make_annotations()
        updated_annotations = {
            **input_annotations, 99: {"auto": [], "manual": [], "intensities": []}
        }
        config = PipelineConfig()
        fe = FisheyeContext()
        review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 3}

        with patch("person_anonymizer.manual_reviewer.run_manual_review",
                   return_value=(updated_annotations, review_stats)):

            from person_anonymizer.stage_review import run_manual_review_stage
            result_ann, _ = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=input_annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fe,
            )

        assert result_ann is updated_annotations

    def test_cli_review_returns_correct_review_stats(self):
        # Arrange
        ctx = _make_ctx(review_state=None)
        annotations = _make_annotations()
        config = PipelineConfig()
        fisheye = FisheyeContext()
        expected_stats = {"added": 5, "removed": 2, "frames_modified": 3, "frames_reviewed": 3}

        with patch("person_anonymizer.manual_reviewer.run_manual_review",
                   return_value=(annotations, expected_stats)):

            from person_anonymizer.stage_review import run_manual_review_stage
            _, result_stats = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        assert result_stats == expected_stats

    def test_zero_total_frames_cli_review_does_not_crash(self):
        # Arrange
        ctx = _make_ctx(review_state=None)
        annotations = {}
        config = PipelineConfig()
        fisheye = FisheyeContext()
        review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}

        with patch("person_anonymizer.manual_reviewer.run_manual_review",
                   return_value=(annotations, review_stats)):

            from person_anonymizer.stage_review import run_manual_review_stage
            result_ann, result_stats = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=0,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        assert result_stats["frames_modified"] == 0


class TestRunManualReviewStageWeb:
    """Test revisione manuale tramite web (review_state non None)."""

    def _make_web_ctx(self, annotations_result=None, n_frames=3):
        review_state = MagicMock()
        if annotations_result is None:
            annotations_result = _make_annotations(n_frames)
        review_state.wait_for_completion.return_value = annotations_result
        sse_mgr = MagicMock()
        ctx = _make_ctx(
            review_state=review_state,
            sse_manager=sse_mgr,
            job_id="job-123",
        )
        return ctx, review_state, sse_mgr

    def test_web_review_calls_setup_on_review_state(self):
        # Arrange
        annotations = _make_annotations()
        ctx, review_state, sse_mgr = self._make_web_ctx(annotations)
        config = PipelineConfig()
        fisheye = FisheyeContext()

        with patch(
            "person_anonymizer.stage_review.compute_review_stats",
            return_value={"frames_modified": 0, "added": 0, "removed": 0},
        ):
            from person_anonymizer.stage_review import run_manual_review_stage
            run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        review_state.setup.assert_called_once_with(
            "test.mp4", annotations, 3, 640, 480, 25.0, fisheye
        )

    def test_web_review_emits_review_ready_event(self):
        # Arrange
        annotations = _make_annotations()
        ctx, review_state, sse_mgr = self._make_web_ctx(annotations)
        config = PipelineConfig()
        fisheye = FisheyeContext()

        with patch(
            "person_anonymizer.stage_review.compute_review_stats",
            return_value={"frames_modified": 0, "added": 0, "removed": 0},
        ):
            from person_anonymizer.stage_review import run_manual_review_stage
            run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        sse_mgr.emit.assert_called_once_with(
            "job-123",
            "review_ready",
            {"total_frames": 3, "frame_w": 640, "frame_h": 480, "fps": 25.0},
        )

    def test_web_review_waits_for_completion(self):
        # Arrange
        updated = _make_annotations()
        ctx, review_state, sse_mgr = self._make_web_ctx(updated)
        config = PipelineConfig()
        fisheye = FisheyeContext()

        with patch(
            "person_anonymizer.stage_review.compute_review_stats",
            return_value={"frames_modified": 0, "added": 0, "removed": 0},
        ):
            from person_anonymizer.stage_review import run_manual_review_stage
            result_ann, _ = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=_make_annotations(),
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        review_state.wait_for_completion.assert_called_once()
        assert result_ann is updated

    def test_web_review_computes_review_stats(self):
        # Arrange
        annotations = _make_annotations()
        ctx, review_state, sse_mgr = self._make_web_ctx(annotations)
        config = PipelineConfig()
        fisheye = FisheyeContext()
        expected_stats = {"added": 2, "removed": 1, "frames_modified": 2, "frames_reviewed": 3}

        with patch("person_anonymizer.stage_review.compute_review_stats",
                   return_value=expected_stats) as mock_stats:
            from person_anonymizer.stage_review import run_manual_review_stage
            _, result_stats = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        mock_stats.assert_called_once()
        assert result_stats == expected_stats

    def test_web_review_initial_stats_zero(self):
        # Arrange — stats iniziali sono zero prima del web review
        annotations = _make_annotations()
        ctx, review_state, sse_mgr = self._make_web_ctx(annotations)
        config = PipelineConfig()
        fisheye = FisheyeContext()

        with patch("person_anonymizer.stage_review.compute_review_stats",
                   return_value={"added": 0, "removed": 0, "frames_modified": 0,
                                 "frames_reviewed": 0}):
            from person_anonymizer.stage_review import run_manual_review_stage
            _, result_stats = run_manual_review_stage(
                ctx=ctx,
                input_path="test.mp4",
                annotations=annotations,
                config=config,
                total_frames=3,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                fisheye=fisheye,
            )

        # Nessun errore — valori coerenti
        assert result_stats["added"] == 0


# Fixture per evitare errori di scope
@pytest.fixture
def fisheye():
    return FisheyeContext()
