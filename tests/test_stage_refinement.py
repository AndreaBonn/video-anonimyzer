"""
Test per stage_refinement.py — run_refinement_loop.

Mock di render_video, run_post_render_check, filter_artifact_detections.
"""

import threading
from unittest.mock import MagicMock, patch

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext


def _make_config(**kwargs):
    defaults = dict(
        enable_post_render_check=True,
        max_refinement_passes=2,
        post_render_check_confidence=0.3,
        refinement_overlap_threshold=0.5,
        person_padding=15,
        enable_adaptive_intensity=False,
        anonymization_intensity=10,
        adaptive_reference_height=180,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_annotations(n_frames=5):
    return {
        i: {"auto": [], "manual": [], "intensities": []}
        for i in range(n_frames)
    }


class TestRunRefinementLoopDisabled:
    """Test quando il post-render check è disabilitato."""

    def test_returns_original_annotations_when_disabled(self):
        # Arrange
        config = _make_config(enable_post_render_check=False)
        annotations = _make_annotations()

        # Act
        from person_anonymizer.stage_refinement import run_refinement_loop
        result_ann, passes, added = run_refinement_loop(
            input_path="input.mp4",
            annotations=annotations,
            model=MagicMock(),
            config=config,
            fps=25.0,
            frame_w=640,
            frame_h=480,
            method="pixelation",
            fisheye=FisheyeContext(),
            report_data={},
            temp_video_path="temp.avi",
        )

        # Assert
        assert result_ann is annotations
        assert passes == 0
        assert added == 0

    def test_render_not_called_when_disabled(self):
        # Arrange
        config = _make_config(enable_post_render_check=False)
        annotations = _make_annotations()

        with patch("person_anonymizer.stage_refinement.render_video") as mock_render:
            from person_anonymizer.stage_refinement import run_refinement_loop
            run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        mock_render.assert_not_called()


class TestRunRefinementLoopNoAlerts:
    """Test quando il post-render check non trova residui."""

    def test_stops_after_first_pass_when_no_alerts(self):
        # Arrange
        config = _make_config(max_refinement_passes=3)
        annotations = _make_annotations()

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   return_value=[]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            _, passes, added = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert passes == 1
        assert added == 0

    def test_render_called_once_when_no_alerts(self):
        # Arrange
        config = _make_config(max_refinement_passes=3)
        annotations = _make_annotations()

        with patch("person_anonymizer.stage_refinement.render_video") as mock_render, \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   return_value=[]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert mock_render.call_count == 1


class TestRunRefinementLoopWithAlerts:
    """Test quando vengono trovati residui genuini."""

    def test_genuine_alerts_added_to_annotations(self):
        # Arrange
        config = _make_config(max_refinement_passes=2)
        annotations = _make_annotations(n_frames=3)
        # Frame 1 ha un residuo genuino
        genuine_alerts = [(1, [[50, 50, 150, 200]])]

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   side_effect=[genuine_alerts, []]), \
             patch("person_anonymizer.stage_refinement.filter_artifact_detections",
                   return_value=(genuine_alerts, 0, 1)), \
             patch("person_anonymizer.stage_refinement.box_to_polygon",
                   return_value=[(50, 50), (150, 50), (150, 200), (50, 200)]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            result_ann, passes, added = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        # 1 annotazione aggiunta
        assert added == 1
        assert len(result_ann[1]["auto"]) == 1

    def test_annotations_added_count_is_cumulative(self):
        # Arrange — 2 pass, ognuno aggiunge annotazioni
        config = _make_config(max_refinement_passes=3)
        annotations = _make_annotations(n_frames=3)
        genuine_pass1 = [(1, [[50, 50, 150, 200]])]
        genuine_pass2 = [(2, [[60, 60, 160, 210]])]

        call_count = [0]
        def fake_post_check(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return genuine_pass1
            elif call_count[0] == 2:
                return genuine_pass2
            return []

        def fake_filter(*args, **kwargs):
            frame_alerts = args[0]
            return frame_alerts, 0, len(frame_alerts)

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   side_effect=fake_post_check), \
             patch("person_anonymizer.stage_refinement.filter_artifact_detections",
                   side_effect=fake_filter), \
             patch("person_anonymizer.stage_refinement.box_to_polygon",
                   return_value=[(50, 50), (150, 50), (150, 200), (50, 200)]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            _, passes, added = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert added == 2

    def test_only_artifacts_stops_loop(self):
        # Arrange — tutti i rilevamenti sono artefatti
        config = _make_config(max_refinement_passes=3)
        annotations = _make_annotations()
        fake_alerts = [(1, [[50, 50, 150, 200]])]

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   return_value=fake_alerts), \
             patch("person_anonymizer.stage_refinement.filter_artifact_detections",
                   return_value=([], 1, 0)):  # tutti artefatti, 0 genuini

            from person_anonymizer.stage_refinement import run_refinement_loop
            _, passes, added = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert passes == 1
        assert added == 0

    def test_max_passes_reached_with_residuals(self):
        # Arrange — sempre residui genuini → raggiunge max_passes
        config = _make_config(max_refinement_passes=2)
        annotations = _make_annotations(n_frames=3)
        genuine_alerts = [(1, [[50, 50, 150, 200]])]

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   return_value=genuine_alerts), \
             patch("person_anonymizer.stage_refinement.filter_artifact_detections",
                   return_value=(genuine_alerts, 0, 1)), \
             patch("person_anonymizer.stage_refinement.box_to_polygon",
                   return_value=[(50, 50), (150, 50), (150, 200), (50, 200)]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            _, passes, _ = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert passes == config.max_refinement_passes

    def test_new_frame_in_alerts_creates_annotation_entry(self):
        # Arrange — frame 99 non era nelle annotations, servono 2 pass:
        # pass 1 trova alert → aggiunge annotation, pass 2 niente alert → break
        config = _make_config(max_refinement_passes=2)
        annotations = _make_annotations(n_frames=3)
        genuine_alerts = [(99, [[10, 10, 100, 200]])]

        with patch("person_anonymizer.stage_refinement.render_video"), \
             patch("person_anonymizer.stage_refinement.run_post_render_check",
                   side_effect=[genuine_alerts, []]), \
             patch("person_anonymizer.stage_refinement.filter_artifact_detections",
                   return_value=(genuine_alerts, 0, 1)), \
             patch("person_anonymizer.stage_refinement.box_to_polygon",
                   return_value=[(10, 10), (100, 10), (100, 200), (10, 200)]):

            from person_anonymizer.stage_refinement import run_refinement_loop
            result_ann, _, _ = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
            )

        assert 99 in result_ann
        assert result_ann[99]["auto"] != []


class TestRunRefinementLoopStopEvent:
    """Test interruzione tramite stop_event."""

    def test_stop_event_prevents_execution(self):
        # Arrange
        stop = threading.Event()
        stop.set()
        config = _make_config(max_refinement_passes=3)
        annotations = _make_annotations()

        with patch("person_anonymizer.stage_refinement.render_video") as mock_render:
            from person_anonymizer.stage_refinement import run_refinement_loop
            _, passes, added = run_refinement_loop(
                input_path="input.mp4",
                annotations=annotations,
                model=MagicMock(),
                config=config,
                fps=25.0,
                frame_w=640,
                frame_h=480,
                method="pixelation",
                fisheye=FisheyeContext(),
                report_data={},
                temp_video_path="temp.avi",
                stop_event=stop,
            )

        mock_render.assert_not_called()
        assert added == 0
