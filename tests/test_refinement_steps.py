"""Test puri per i passi estratti da run_refinement_loop."""

from person_anonymizer.config import PipelineConfig
from person_anonymizer.stage_refinement import (
    ResidualAnnotation,
    apply_residual_additions,
    build_residual_additions,
)

BOX = (100, 100, 200, 300, 0.9)


def _config(**kw):
    return PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10, **kw)


class TestBuildResidualAdditions:
    def test_build_residual_additions_converts_box_to_polygon_with_intensity(self):
        config = _config(person_padding=0)

        additions = build_residual_additions([(3, [BOX])], config, 640, 480)

        assert list(additions) == [3]
        (residual,) = additions[3]
        assert isinstance(residual, ResidualAnnotation)
        assert residual.intensity == 10
        xs = [p[0] for p in residual.polygon]
        ys = [p[1] for p in residual.polygon]
        assert (min(xs), min(ys), max(xs), max(ys)) == (100, 100, 200, 300)

    def test_build_residual_additions_keeps_order_and_groups_by_frame(self):
        config = _config(person_padding=0)
        other = (10, 10, 20, 40, 0.8)

        additions = build_residual_additions([(5, [BOX, other]), (2, [other])], config, 640, 480)

        assert list(additions) == [5, 2]
        assert [len(v) for v in additions.values()] == [2, 1]
        assert additions[5][1].intensity == 10

    def test_build_residual_additions_empty_boxes_still_registers_frame(self):
        additions = build_residual_additions([(7, [])], _config(), 640, 480)

        assert additions == {7: []}

    def test_build_residual_additions_does_not_mutate_alerts(self):
        alerts = [(1, [BOX])]

        build_residual_additions(alerts, _config(), 640, 480)

        assert alerts == [(1, [BOX])]

    def test_build_residual_additions_no_alerts_returns_empty(self):
        assert build_residual_additions([], _config(), 640, 480) == {}


class TestApplyResidualAdditions:
    def test_apply_residual_additions_appends_to_existing_frame_in_place(self):
        annotations = {4: {"auto": ["old"], "manual": ["m"], "intensities": [9]}}
        additions = {4: [ResidualAnnotation(polygon="new", intensity=5)]}

        added = apply_residual_additions(annotations, additions)

        assert added == 1
        assert annotations[4] == {"auto": ["old", "new"], "manual": ["m"], "intensities": [9, 5]}

    def test_apply_residual_additions_creates_missing_frame_record(self):
        annotations = {}

        added = apply_residual_additions(
            annotations, {2: [ResidualAnnotation(polygon="p", intensity=7)], 6: []}
        )

        assert added == 1
        assert annotations == {
            2: {"auto": ["p"], "manual": [], "intensities": [7]},
            6: {"auto": [], "manual": [], "intensities": []},
        }

    def test_apply_residual_additions_empty_returns_zero_and_keeps_annotations(self):
        annotations = {1: {"auto": [], "manual": [], "intensities": []}}

        assert apply_residual_additions(annotations, {}) == 0
        assert list(annotations) == [1]
