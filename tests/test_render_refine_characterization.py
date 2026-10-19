"""Characterization tests for ``render_video`` and ``run_refinement_loop``.

Safety net for the upcoming decomposition of both functions: they exercise only
the public entry points and assert on observable results (re-read output video,
return values, stdout), never on private helpers.

``render_video`` runs on a real synthetic FFV1 video (lossless, so untouched
pixels must come back bit-identical). ``run_refinement_loop`` patches its two
expensive boundaries (``render_video`` and ``run_post_render_check``) where
``stage_refinement`` looks them up.
"""

import logging
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext, PostCheckOutcome
from person_anonymizer.rendering import render_video
from person_anonymizer.stage_refinement import run_refinement_loop

# ============================================================
# render_video
# ============================================================

FRAME_W, FRAME_H = 160, 80
N_FRAMES = 4
FPS = 10.0

# Two disjoint rectangles, same size (41x41 bounding box once drawn)
POLY_A = [(5, 10), (45, 10), (45, 50), (5, 50)]
POLY_B = [(90, 10), (130, 10), (130, 50), (90, 50)]


def _random_frames(n=N_FRAMES, seed=1234):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(n)]


def _write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (FRAME_W, FRAME_H))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _render(tmp_path, annotations, frames=None, **overrides):
    """Render a synthetic video and return (input_frames, output_frames, debug_frames)."""
    frames = frames if frames is not None else _random_frames()
    input_path = tmp_path / "in.avi"
    output_path = tmp_path / "out.avi"
    debug_path = tmp_path / "debug.avi" if overrides.pop("with_debug", False) else None
    _write_video(input_path, frames)

    kwargs = dict(
        config=overrides.pop("config", PipelineConfig(enable_adaptive_intensity=True)),
        method=overrides.pop("method", "pixelation"),
        fisheye=overrides.pop("fisheye", FisheyeContext()),
        debug_path=str(debug_path) if debug_path else None,
    )
    kwargs.update(overrides)
    render_video(
        str(input_path),
        str(output_path),
        annotations,
        FPS,
        FRAME_W,
        FRAME_H,
        **kwargs,
    )
    source = _read_video(input_path)
    debug = _read_video(debug_path) if debug_path else None
    return source, _read_video(output_path), debug


def _crop(frame, poly, margin=0):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return frame[min(ys) + margin : max(ys) + 1 - margin, min(xs) + margin : max(xs) + 1 - margin]


def _outside_mask(poly, shape=(FRAME_H, FRAME_W)):
    """True everywhere except the polygon bounding box grown by 1 pixel."""
    mask = np.ones(shape, dtype=bool)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    mask[max(0, min(ys) - 1) : max(ys) + 2, max(0, min(xs) - 1) : max(xs) + 2] = False
    return mask


def _distinct_colors(region):
    return len(np.unique(region.reshape(-1, 3), axis=0))


def _changed_fraction(before, after):
    return float(np.mean(np.any(before != after, axis=2)))


def _annotation(auto=(), manual=(), intensities=()):
    return {"auto": list(auto), "manual": list(manual), "intensities": list(intensities)}


class TestRenderVideoPixels:
    def test_render_video_obscures_only_polygon_pixels_on_annotated_frames(self, tmp_path):
        # Arrange: auto polygon on frame 1, manual polygon on frame 2
        annotations = {1: _annotation(auto=[POLY_A]), 2: _annotation(manual=[POLY_B])}

        # Act
        source, output, _ = _render(tmp_path, annotations)

        # Assert: polygon interior changed, everything else bit-identical (lossless codec)
        assert _changed_fraction(_crop(source[1], POLY_A, 2), _crop(output[1], POLY_A, 2)) > 0.9
        assert _changed_fraction(_crop(source[2], POLY_B, 2), _crop(output[2], POLY_B, 2)) > 0.9
        mask_a, mask_b = _outside_mask(POLY_A), _outside_mask(POLY_B)
        assert np.array_equal(source[1][mask_a], output[1][mask_a])
        assert np.array_equal(source[2][mask_b], output[2][mask_b])

    def test_render_video_leaves_unannotated_frames_identical_and_keeps_all_frames(self, tmp_path):
        # Arrange
        annotations = {1: _annotation(auto=[POLY_A]), 2: _annotation(manual=[POLY_B])}

        # Act
        source, output, _ = _render(tmp_path, annotations)

        # Assert
        assert len(source) == N_FRAMES
        assert len(output) == N_FRAMES
        assert np.array_equal(source[0], output[0])
        assert np.array_equal(source[3], output[3])
        # The other polygon's region on the annotated frames is untouched too
        assert np.array_equal(_crop(source[1], POLY_B), _crop(output[1], POLY_B))
        assert np.array_equal(_crop(source[2], POLY_A), _crop(output[2], POLY_A))

    def test_render_video_without_annotations_copies_every_frame(self, tmp_path):
        # Act
        source, output, _ = _render(tmp_path, {})

        # Assert
        assert len(output) == len(source) == N_FRAMES
        assert all(np.array_equal(s, o) for s, o in zip(source, output, strict=True))

    def test_render_video_applies_fisheye_undistort_before_obscuring(self, tmp_path):
        # Arrange: fake fisheye that mirrors the frame horizontally
        fisheye = MagicMock()
        fisheye.undistort.side_effect = lambda frame: frame[:, ::-1].copy()

        # Act
        source, output, _ = _render(tmp_path, {}, fisheye=fisheye)

        # Assert
        assert len(output) == N_FRAMES
        assert np.array_equal(output[0], source[0][:, ::-1])


class TestRenderVideoIntensity:
    def test_render_video_uses_per_polygon_intensity_for_auto_polygons(self, tmp_path):
        # Arrange: same-size polygons, 4px blocks vs 20px blocks
        annotations = {0: _annotation(auto=[POLY_A, POLY_B], intensities=[4, 20])}
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=10)

        # Act
        _, output, _ = _render(tmp_path, annotations, config=config)

        # Assert: 41//4 = 10 -> up to 100 blocks; 41//20 = 2 -> at most 4 blocks
        fine = _distinct_colors(_crop(output[0], POLY_A, 2))
        coarse = _distinct_colors(_crop(output[0], POLY_B, 2))
        assert fine >= 50
        assert coarse <= 4

    def test_render_video_ignores_per_polygon_intensity_when_adaptive_disabled(self, tmp_path):
        # Arrange: config intensity 10 -> 41//10 = 4 -> 16 blocks for both polygons
        annotations = {0: _annotation(auto=[POLY_A, POLY_B], intensities=[4, 20])}
        config = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)

        # Act
        _, output, _ = _render(tmp_path, annotations, config=config)

        # Assert
        for poly in (POLY_A, POLY_B):
            assert 10 <= _distinct_colors(_crop(output[0], poly, 2)) <= 16

    def test_render_video_falls_back_to_config_when_intensities_shorter_than_auto(self, tmp_path):
        # Arrange: only the first auto polygon has an intensity
        annotations = {0: _annotation(auto=[POLY_A, POLY_B], intensities=[20])}
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=10)

        # Act
        _, output, _ = _render(tmp_path, annotations, config=config)

        # Assert: A uses 20 (<= 4 blocks), B falls back to 10 (16 blocks)
        assert _distinct_colors(_crop(output[0], POLY_A, 2)) <= 4
        assert 10 <= _distinct_colors(_crop(output[0], POLY_B, 2)) <= 16

    def test_render_video_manual_polygons_use_config_intensity_not_per_polygon_list(self, tmp_path):
        # Arrange: intensities list present (4) but manual polygons ignore it
        annotations = {0: _annotation(manual=[POLY_A], intensities=[4])}
        config = PipelineConfig(enable_adaptive_intensity=True, anonymization_intensity=20)

        # Act
        _, output, _ = _render(tmp_path, annotations, config=config)

        # Assert: 20px blocks -> at most 4 distinct colors
        assert _distinct_colors(_crop(output[0], POLY_A, 2)) <= 4

    def test_render_video_blur_even_intensity_behaves_like_next_odd(self, tmp_path):
        # Arrange
        annotations = {0: _annotation(auto=[POLY_A])}
        even = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)
        odd = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=11)

        # Act
        source, out_even, _ = _render(tmp_path, annotations, config=even, method="blur")
        _, out_odd, _ = _render(tmp_path, annotations, config=odd, method="blur")

        # Assert: blur actually changed the polygon, and 10 == 11
        assert _changed_fraction(_crop(source[0], POLY_A, 2), _crop(out_even[0], POLY_A, 2)) > 0.9
        assert np.array_equal(out_even[0], out_odd[0])


class TestRenderVideoDebug:
    def test_render_video_debug_video_has_polygon_contours_and_same_frame_count(self, tmp_path):
        # Arrange
        annotations = {1: _annotation(auto=[POLY_A]), 2: _annotation(manual=[POLY_B])}
        config = PipelineConfig(enable_adaptive_intensity=True)

        # Act
        source, _, debug = _render(tmp_path, annotations, config=config, with_debug=True)

        # Assert
        assert len(debug) == N_FRAMES
        auto_edge = debug[1][30, 5].astype(int)
        manual_edge = debug[2][30, 90].astype(int)
        assert np.abs(auto_edge - np.array(config.review_auto_color)).max() <= 1
        assert np.abs(manual_edge - np.array(config.review_manual_color)).max() <= 1
        # Interior is the original blended with the fill color at review_fill_alpha
        alpha = config.review_fill_alpha
        expected = (1 - alpha) * source[1][30, 25].astype(float) + alpha * np.array(
            config.review_auto_color, dtype=float
        )
        assert np.abs(debug[1][30, 25].astype(float) - expected).max() <= 1.5

    def test_render_video_debug_video_shows_unobscured_original_outside_polygons(self, tmp_path):
        # Arrange
        annotations = {1: _annotation(auto=[POLY_A])}

        # Act
        source, _, debug = _render(tmp_path, annotations, with_debug=True)

        # Assert: outside the (grown) polygon the debug frame equals the source;
        # frames without annotations are identical to the source
        mask = _outside_mask(POLY_A)
        assert np.array_equal(debug[1][mask], source[1][mask])
        assert np.array_equal(debug[0], source[0])


class _FakeCapture:
    """Stand-in for cv2.VideoCapture replaying a scripted read() sequence."""

    script: list = []
    total = 0

    def __init__(self, _path):
        self._reads = list(type(self).script)

    def get(self, prop):
        return type(self).total if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    def read(self):
        return self._reads.pop(0) if self._reads else (False, None)

    def release(self):
        pass


class _CountdownStop:
    """stop_event that reports 'set' only after ``allowed`` loop iterations."""

    def __init__(self, allowed):
        self._allowed = allowed
        self._calls = 0

    def is_set(self):
        self._calls += 1
        return self._calls > self._allowed


class TestRenderVideoFailureModes:
    def test_render_video_skips_corrupted_frame_and_keeps_annotation_alignment(
        self, tmp_path, caplog
    ):
        # Arrange: frame index 1 cannot be decoded; annotation targets original index 2
        frames = _random_frames()
        _FakeCapture.script = [
            (True, frames[0].copy()),
            (False, None),
            (True, frames[2].copy()),
            (True, frames[3].copy()),
            (False, None),
        ]
        _FakeCapture.total = N_FRAMES
        output_path = tmp_path / "out.avi"
        annotations = {2: _annotation(auto=[POLY_A])}

        # Act
        with (
            patch("person_anonymizer.rendering.cv2.VideoCapture", _FakeCapture),
            caplog.at_level(logging.WARNING, logger="person_anonymizer.rendering"),
        ):
            render_video(
                "ignored.avi",
                str(output_path),
                annotations,
                FPS,
                FRAME_W,
                FRAME_H,
                "pixelation",
                FisheyeContext(),
                PipelineConfig(enable_adaptive_intensity=True),
            )
        output = _read_video(output_path)

        # Assert: the corrupted frame is dropped and counted, indices stay aligned
        assert len(output) == N_FRAMES - 1
        assert np.array_equal(output[0], frames[0])
        assert _changed_fraction(_crop(frames[2], POLY_A, 2), _crop(output[1], POLY_A, 2)) > 0.9
        mask = _outside_mask(POLY_A)
        assert np.array_equal(output[1][mask], frames[2][mask])
        assert np.array_equal(output[2], frames[3])
        assert "Rendering: 1 frame corrotti saltati su 4 totali" in caplog.text

    def test_render_video_clean_run_does_not_warn_about_corrupted_frames(self, tmp_path, caplog):
        # Act
        with caplog.at_level(logging.WARNING, logger="person_anonymizer.rendering"):
            _render(tmp_path, {})

        # Assert
        assert "corrotti" not in caplog.text

    def test_render_video_stop_event_interrupts_after_frames_already_written(self, tmp_path):
        # Arrange: stop requested after two loop iterations
        stop = _CountdownStop(allowed=2)

        # Act
        source, output, _ = _render(tmp_path, {}, stop_event=stop)

        # Assert
        assert len(source) == N_FRAMES
        assert len(output) == 2
        assert np.array_equal(output[0], source[0])
        assert np.array_equal(output[1], source[1])

    def test_render_video_unwritable_output_raises_runtime_error(self, tmp_path):
        # Arrange
        input_path = tmp_path / "in.avi"
        _write_video(input_path, _random_frames())

        # Act / Assert
        with pytest.raises(RuntimeError, match="VideoWriter"):
            render_video(
                str(input_path),
                str(tmp_path / "missing_dir" / "out.avi"),
                {},
                FPS,
                FRAME_W,
                FRAME_H,
                "pixelation",
                FisheyeContext(),
                PipelineConfig(),
            )


# ============================================================
# run_refinement_loop
# ============================================================

REF_W, REF_H = 640, 480
# 160px-tall box away from the frame edges, padding 5 -> polygon below
BOX_TALL = (100, 100, 180, 260, 0.9)
POLY_TALL = [(95, 95), (185, 95), (185, 265), (95, 265)]
# 100px-tall box -> polygon below
BOX_MID = (300, 300, 380, 400, 0.8)
POLY_MID = [(295, 295), (385, 295), (385, 405), (295, 405)]


def _config(**overrides):
    # intensity 40, reference 80: height 160 -> adaptive 80 (min 40); height 100 -> 50 (min 25)
    defaults = dict(
        enable_post_render_check=True,
        max_refinement_passes=3,
        post_render_check_confidence=0.3,
        refinement_overlap_threshold=0.5,
        person_padding=5,
        enable_adaptive_intensity=True,
        anonymization_intensity=40,
        adaptive_reference_height=80,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _empty_annotations(n=5):
    return {i: _annotation() for i in range(n)}


def _run(annotations, alerts_per_pass, config=None, stop_event=None):
    """Run the loop with patched boundaries. Returns (result, render_mock, check_mock)."""
    config = config or _config()
    model = MagicMock()
    report_data = {}
    with (
        patch("person_anonymizer.stage_refinement.render_video") as render,
        patch(
            "person_anonymizer.stage_refinement.run_post_render_check",
            side_effect=alerts_per_pass,
        ) as check,
    ):
        result = run_refinement_loop(
            "in.mp4",
            annotations,
            model,
            config,
            25.0,
            REF_W,
            REF_H,
            "pixelation",
            FisheyeContext(),
            report_data,
            "tmp.avi",
            stop_event=stop_event,
        )
    return result, render, check, model, report_data


def _lines(capsys):
    return [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]


class TestRefinementOutcomes:
    def test_run_refinement_loop_clean_first_pass_returns_clean_without_changes(self, capsys):
        # Arrange
        annotations = _empty_annotations()

        # Act
        (ann, passes, added, outcome), render, check, model, report = _run(
            annotations, alerts_per_pass=[[]], config=_config(max_refinement_passes=2)
        )

        # Assert
        assert ann is annotations
        assert annotations == _empty_annotations()
        assert (passes, added) == (1, 0)
        assert outcome == PostCheckOutcome(status="clean")
        assert render.call_count == 1
        assert check.call_args.args == (
            "tmp.avi",
            model,
            0.3,
            report,
            _config(max_refinement_passes=2),
        )
        assert _lines(capsys) == [
            "[FASE 2/5] Auto refinement — rendering (pass 1/2)...",
            "Verifica post-rendering (pass 1/2)...",
            "Nessun rilevamento residuo — rendering OK.",
        ]

    def test_run_refinement_loop_renders_with_current_context_each_pass(self):
        # Arrange
        config = _config(max_refinement_passes=2)
        stop = MagicMock()
        stop.is_set.return_value = False

        # Act
        (_, passes, _, _), render, _, _, _ = _run(
            _empty_annotations(), [[(1, 1, [BOX_TALL])], []], config=config, stop_event=stop
        )

        # Assert
        assert passes == 2
        assert render.call_count == 2
        first, second = render.call_args_list
        assert first.args[0] == "in.mp4" and first.args[1] == "tmp.avi"
        assert first.args[3:7] == (25.0, REF_W, REF_H, "pixelation")
        assert first.args[8] is config
        assert first.kwargs == {
            "debug_path": None,
            "desc": "Rendering (pass 1/2)",
            "stop_event": stop,
        }
        assert second.kwargs["desc"] == "Rendering (pass 2/2)"

    def test_run_refinement_loop_adds_genuine_residuals_with_computed_intensity(self, capsys):
        # Arrange: frame 2 exists, frame 7 does not
        annotations = _empty_annotations()
        alerts = [[(2, 1, [BOX_TALL]), (7, 1, [BOX_MID])], []]

        # Act
        (ann, passes, added, outcome), render, check, _, _ = _run(annotations, alerts)

        # Assert
        assert ann is annotations
        assert (passes, added) == (2, 2)
        assert outcome == PostCheckOutcome(status="clean")
        assert render.call_count == 2 and check.call_count == 2
        assert annotations[2]["auto"] == [POLY_TALL]
        assert annotations[2]["intensities"] == [80]
        assert annotations[2]["manual"] == []
        assert annotations[7] == _annotation(auto=[POLY_MID], intensities=[50])
        assert _lines(capsys) == [
            "[FASE 2/5] Auto refinement — rendering (pass 1/3)...",
            "Verifica post-rendering (pass 1/3)...",
            "Rilevamenti post-render: 2",
            "Artefatti filtrati (IoU >= 0.5): 0",
            "Residui genuini: 2",
            "Aggiunte 2 annotazioni — ri-rendering in corso...",
            "[FASE 2/5] Auto refinement — rendering (pass 2/3)...",
            "Verifica post-rendering (pass 2/3)...",
            "Nessun rilevamento residuo — rendering OK.",
        ]

    def test_run_refinement_loop_uses_config_intensity_when_adaptive_disabled(self):
        # Arrange
        config = _config(enable_adaptive_intensity=False, anonymization_intensity=33)
        annotations = _empty_annotations()

        # Act
        _run(annotations, [[(1, 1, [BOX_TALL])], []], config=config)

        # Assert
        assert annotations[1]["intensities"] == [33]

    def test_run_refinement_loop_appends_to_existing_annotations_keeping_them(self):
        # Arrange: frame 1 already has an unrelated auto and a manual polygon
        annotations = {1: _annotation(auto=[POLY_MID], manual=[POLY_A], intensities=[50])}

        # Act
        _run(annotations, [[(1, 1, [BOX_TALL])], []])

        # Assert
        assert annotations[1]["auto"] == [POLY_MID, POLY_TALL]
        assert annotations[1]["manual"] == [POLY_A]
        assert annotations[1]["intensities"] == [50, 80]

    def test_run_refinement_loop_limit_reached_returns_residual_and_drops_last_pass(self, capsys):
        # Arrange: pass 1 finds BOX_TALL (added), pass 2 finds BOX_MID elsewhere (not added)
        config = _config(max_refinement_passes=2)
        annotations = _empty_annotations()
        alerts = [
            [(1, 1, [BOX_TALL]), (3, 1, [BOX_TALL])],
            [(1, 1, [BOX_MID]), (3, 1, [BOX_MID])],
        ]

        # Act
        (ann, passes, added, outcome), render, _, _, _ = _run(annotations, alerts, config=config)

        # Assert
        assert (passes, added) == (2, 2)
        assert outcome == PostCheckOutcome(status="residual", residual_frames=2)
        assert render.call_count == 2
        for fidx in (1, 3):
            assert ann[fidx]["auto"] == [POLY_TALL]
            assert ann[fidx]["intensities"] == [80]
        assert _lines(capsys)[-4:] == [
            "Raggiunto limite di 2 pass.",
            "Residui genuini rimasti in 2 frame:",
            "Frame 1: 1 persona/e",
            "Frame 3: 1 persona/e",
        ]

    def test_run_refinement_loop_single_pass_budget_reports_residual_without_adding(self):
        # Arrange
        config = _config(max_refinement_passes=1)
        annotations = _empty_annotations()

        # Act
        (ann, passes, added, outcome), _, _, _, _ = _run(
            annotations, [[(0, 2, [BOX_TALL, BOX_MID])]], config=config
        )

        # Assert: residual_frames counts frames, not boxes
        assert (passes, added) == (1, 0)
        assert outcome == PostCheckOutcome(status="residual", residual_frames=1)
        assert ann == _empty_annotations()


class TestRefinementArtifactFilter:
    def test_run_refinement_loop_artifact_overlapping_existing_polygon_is_not_added(self, capsys):
        # Arrange: detection (102,102,178,258) vs polygon box (95,95,185,265): IoU ~ 0.93
        annotations = _empty_annotations()
        annotations[0] = _annotation(auto=[POLY_TALL], intensities=[80])
        alerts = [[(0, 1, [(102, 102, 178, 258, 0.7)])]]

        # Act
        (ann, passes, added, outcome), render, _, _, _ = _run(annotations, alerts)

        # Assert
        assert (passes, added) == (1, 0)
        assert outcome == PostCheckOutcome(status="clean")
        assert ann[0] == _annotation(auto=[POLY_TALL], intensities=[80])
        assert render.call_count == 1
        assert _lines(capsys)[-4:] == [
            "Rilevamenti post-render: 1",
            "Artefatti filtrati (IoU >= 0.5): 1",
            "Residui genuini: 0",
            "Tutti i rilevamenti sono artefatti della pixelazione — rendering OK.",
        ]

    def test_run_refinement_loop_artifact_against_manual_polygon_is_not_added(self):
        # Arrange
        annotations = _empty_annotations()
        annotations[0] = _annotation(manual=[POLY_TALL])
        alerts = [[(0, 1, [(102, 102, 178, 258, 0.7)])]]

        # Act
        (ann, _, added, outcome), _, _, _, _ = _run(annotations, alerts)

        # Assert
        assert added == 0
        assert outcome.status == "clean"
        assert ann[0]["auto"] == []

    def test_run_refinement_loop_adds_only_genuine_boxes_when_mixed_with_artifacts(self, capsys):
        # Arrange: frame 0 has [artifact, low-IoU box (~0.23, genuine)]
        annotations = _empty_annotations()
        annotations[0] = _annotation(auto=[POLY_TALL], intensities=[80])
        artifact = (102, 102, 178, 258, 0.7)
        shifted = (150, 100, 230, 260, 0.6)
        alerts = [[(0, 2, [artifact, shifted])], []]

        # Act
        (ann, passes, added, outcome), _, _, _, _ = _run(annotations, alerts)

        # Assert
        assert (passes, added) == (2, 1)
        assert outcome.status == "clean"
        assert len(ann[0]["auto"]) == 2
        assert ann[0]["auto"][1] == [(145, 95), (235, 95), (235, 265), (145, 265)]
        assert ann[0]["intensities"] == [80, 80]
        out = _lines(capsys)
        assert "Rilevamenti post-render: 2" in out
        assert "Artefatti filtrati (IoU >= 0.5): 1" in out
        assert "Residui genuini: 1" in out

    def test_run_refinement_loop_overlap_threshold_decides_artifact_vs_genuine(self):
        # Arrange: shifted box has IoU ~0.23 with the existing polygon box
        def run_with(threshold):
            annotations = _empty_annotations()
            annotations[0] = _annotation(auto=[POLY_TALL], intensities=[80])
            config = _config(refinement_overlap_threshold=threshold, max_refinement_passes=1)
            alerts = [[(0, 1, [(150, 100, 230, 260, 0.6)])]]
            (_, _, _, outcome), _, _, _, _ = _run(annotations, alerts, config=config)
            return outcome

        # Act / Assert
        assert run_with(0.2) == PostCheckOutcome(status="clean")
        assert run_with(0.5) == PostCheckOutcome(status="residual", residual_frames=1)


class TestRefinementSkipAndStop:
    def test_run_refinement_loop_disabled_check_returns_skipped_without_rendering(self, capsys):
        # Arrange
        annotations = _empty_annotations()
        config = _config(enable_post_render_check=False)

        # Act
        (ann, passes, added, outcome), render, check, _, _ = _run(annotations, [], config=config)

        # Assert
        assert ann is annotations
        assert (passes, added) == (0, 0)
        assert outcome == PostCheckOutcome(status="skipped")
        render.assert_not_called()
        check.assert_not_called()
        assert _lines(capsys) == ["[FASE 2/5] Auto refinement — saltato (verifica disabilitata)"]

    def test_run_refinement_loop_stop_event_set_before_start_does_nothing(self, capsys):
        # Arrange
        stop = MagicMock()
        stop.is_set.return_value = True

        # Act
        (_, passes, added, outcome), render, check, _, _ = _run(
            _empty_annotations(), [], stop_event=stop
        )

        # Assert
        assert (passes, added) == (0, 0)
        assert outcome == PostCheckOutcome(status="skipped")
        render.assert_not_called()
        check.assert_not_called()
        assert _lines(capsys) == ["Pipeline interrotta dall'utente."]

    def test_run_refinement_loop_stop_during_first_pass_keeps_added_annotations_and_skips(
        self, capsys
    ):
        # Arrange: stop gets set while the first pass renders; the pass still completes
        stop = MagicMock()
        stop.is_set.return_value = False
        annotations = _empty_annotations()

        def render_then_stop(*_args, **_kwargs):
            stop.is_set.return_value = True

        config = _config()
        with (
            patch("person_anonymizer.stage_refinement.render_video", side_effect=render_then_stop),
            patch(
                "person_anonymizer.stage_refinement.run_post_render_check",
                return_value=[(1, 1, [BOX_TALL])],
            ),
        ):
            ann, passes, added, outcome = run_refinement_loop(
                "in.mp4",
                annotations,
                MagicMock(),
                config,
                25.0,
                REF_W,
                REF_H,
                "pixelation",
                FisheyeContext(),
                {},
                "tmp.avi",
                stop_event=stop,
            )

        # Assert: pass 1 counted and its annotations kept, pass 2 interrupted -> skipped
        assert (passes, added) == (1, 1)
        assert ann[1]["auto"] == [POLY_TALL]
        assert outcome == PostCheckOutcome(status="skipped")
        assert _lines(capsys)[-1] == "Pipeline interrotta dall'utente."
