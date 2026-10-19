"""
Characterization tests for run_manual_review_stage.

They pin the current observable behavior of the web and CLI branches through the
public entry point only, so the function can be split without changing behavior.
The web tests use a real ReviewState and a thread acting as the "browser".
"""

import copy
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext, PipelineContext
from person_anonymizer.stage_review import run_manual_review_stage
from person_anonymizer.web.review_state import ReviewState

POLY_A = [(0, 0), (10, 0), (10, 10), (0, 10)]
POLY_B = [(20, 20), (30, 20), (30, 30), (20, 30)]
POLY_C = [(50, 50), (60, 50), (60, 60), (50, 60)]
WAIT_S = 3.0


def _make_ctx(**overrides):
    values = dict(
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
    values.update(overrides)
    return PipelineContext(**values)


def _annotations(n_frames=3):
    return {i: {"auto": [list(POLY_A)], "manual": [], "intensities": [5]} for i in range(n_frames)}


def _stage_kwargs(ctx, annotations, total_frames=3, fisheye=None):
    return dict(
        ctx=ctx,
        input_path="missing.mp4",
        annotations=annotations,
        config=PipelineConfig(),
        total_frames=total_frames,
        fps=25.0,
        frame_w=640,
        frame_h=480,
        fisheye=fisheye if fisheye is not None else FisheyeContext(),
    )


def _wait_until(condition, description):
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError(f"timeout waiting for: {description}")


class _Browser(threading.Thread):
    """Plays the Flask side: waits for the review to open, edits, confirms."""

    def __init__(self, state, sse, edit):
        super().__init__(daemon=True)
        self.state = state
        self.sse = sse
        self.edit = edit
        self.error = None

    def run(self):
        try:
            _wait_until(lambda: self.state.is_active, "review state active")
            _wait_until(lambda: self.sse.emit.called, "review_ready emitted")
            current = self.state.get_annotations()
            self.edit(current)
            self.state.complete(current)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test via .error
            self.error = exc
            self.state.cancel()


def _run_web(edit, annotations, total_frames=3, fisheye=None):
    """Run the stage on the web branch with a browser thread; return all observables."""
    state = ReviewState()
    sse = MagicMock()
    ctx = _make_ctx(review_state=state, sse_manager=sse, job_id="job-42")
    browser = _Browser(state, sse, edit)
    watchdog = threading.Timer(WAIT_S + 2, state.cancel)  # never hang the suite
    watchdog.daemon = True
    watchdog.start()
    browser.start()
    kwargs = _stage_kwargs(ctx, annotations, total_frames=total_frames, fisheye=fisheye)
    with patch.object(state, "setup", wraps=state.setup) as setup_spy:
        result = run_manual_review_stage(**kwargs)
    watchdog.cancel()
    browser.join(timeout=WAIT_S)
    assert browser.error is None, browser.error
    return result, setup_spy, sse


def _edit_add_remove_swap(current):
    current[0]["manual"].append(list(POLY_B))  # frame 0: +1
    current[1]["auto"].clear()  # frame 1: -1
    current[2]["auto"] = [list(POLY_C)]  # frame 2: swap, same count


class TestWebBranch:
    def test_web_review_returns_annotations_edited_in_browser(self, capsys):
        annotations = _annotations()

        (result_ann, _), _, _ = _run_web(_edit_add_remove_swap, annotations)

        assert result_ann[0]["manual"] == [POLY_B]
        assert result_ann[0]["auto"] == [POLY_A]
        assert result_ann[1]["auto"] == []
        assert result_ann[2]["auto"] == [POLY_C]
        assert result_ann[2]["intensities"] == [5]

    def test_web_review_stats_are_computed_against_original_annotations(self):
        annotations = _annotations()

        (_, stats), _, _ = _run_web(_edit_add_remove_swap, annotations)

        # swap on frame 2 keeps the polygon count: not a modified frame
        assert stats == {"added": 1, "removed": 1, "frames_modified": 2, "frames_reviewed": 3}

    def test_web_review_does_not_mutate_input_annotations(self):
        annotations = _annotations()
        before = copy.deepcopy(annotations)

        _run_web(_edit_add_remove_swap, annotations)

        assert annotations == before

    def test_web_review_no_edits_gives_zero_stats(self):
        (result_ann, stats), _, _ = _run_web(lambda current: None, _annotations())

        assert stats == {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 3}
        assert result_ann == _annotations()

    def test_web_review_setup_receives_positional_arguments_in_order(self):
        annotations = _annotations()
        fisheye = FisheyeContext()

        _, setup_spy, _ = _run_web(lambda current: None, annotations, fisheye=fisheye)

        setup_spy.assert_called_once()
        args, kwargs = setup_spy.call_args
        assert kwargs == {}
        assert args == ("missing.mp4", annotations, 3, 640, 480, 25.0, fisheye)
        assert args[1] is annotations
        assert args[6] is fisheye

    def test_web_review_emits_single_review_ready_event(self):
        _, _, sse = _run_web(lambda current: None, _annotations())

        sse.emit.assert_called_once_with(
            "job-42",
            "review_ready",
            {"total_frames": 3, "frame_w": 640, "frame_h": 480, "fps": 25.0},
        )

    def test_web_review_prints_summary_with_percentage(self, capsys):
        _run_web(_edit_add_remove_swap, _annotations())

        out = capsys.readouterr().out
        assert "[FASE 3/5] Revisione manuale — in attesa di conferma dal browser..." in out
        assert "  Revisione completata:" in out
        assert "  Poligoni aggiunti:     1\n" in out
        assert "  Poligoni rimossi:      1\n" in out
        assert "  Frame modificati:      2  (66.7%)" in out
        assert "apertura interfaccia" not in out
        assert "Frame revisionati" not in out

    def test_web_review_zero_total_frames_prints_not_available(self, capsys):
        (_, stats), _, _ = _run_web(lambda current: None, {}, total_frames=0)

        assert stats == {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
        assert "  Frame modificati:      0  (N/A)" in capsys.readouterr().out

    def test_web_review_stop_set_during_review_returns_unmodified_annotations(self, capsys):
        # stop_event already set: the stage cancels the state itself, no browser involved
        state = ReviewState()
        stop = threading.Event()
        stop.set()
        ctx = _make_ctx(
            review_state=state, sse_manager=MagicMock(), job_id="job-42", stop_event=stop
        )
        annotations = _annotations()
        result = {}

        def target():
            result["value"] = run_manual_review_stage(**_stage_kwargs(ctx, annotations))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=WAIT_S)
        if thread.is_alive():
            state.cancel()
            thread.join(timeout=WAIT_S)
            pytest.fail("stage blocked although stop_event was set")

        result_ann, stats = result["value"]
        assert result_ann == annotations
        assert stats == {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 3}
        # current behavior: a cancelled review still prints the "completed" summary
        out = capsys.readouterr().out
        assert "  Revisione completata:" in out
        assert "  Frame modificati:      0  (0.0%)" in out
        assert not state.is_active

    def test_web_review_stop_event_unset_waits_for_browser_and_keeps_edits(self):
        state = ReviewState()
        sse = MagicMock()
        ctx = _make_ctx(
            review_state=state, sse_manager=sse, job_id="job-42", stop_event=threading.Event()
        )
        browser = _Browser(state, sse, _edit_add_remove_swap)
        watchdog = threading.Timer(WAIT_S + 2, state.cancel)
        watchdog.daemon = True
        watchdog.start()
        browser.start()

        _, stats = run_manual_review_stage(**_stage_kwargs(ctx, _annotations()))

        watchdog.cancel()
        browser.join(timeout=WAIT_S)
        assert browser.error is None, browser.error
        assert stats["added"] == 1
        assert stats["removed"] == 1


@pytest.fixture
def cli_reviewer():
    """Patch run_manual_review wherever the stage may look it up.

    Today the stage imports it lazily from manual_reviewer; after hoisting the import
    it will be an attribute of stage_review. Both names point to the same mock.
    """
    reviewer = MagicMock(name="run_manual_review")
    with (
        patch("person_anonymizer.manual_reviewer.run_manual_review", reviewer),
        patch("person_anonymizer.stage_review.run_manual_review", reviewer, create=True),
    ):
        yield reviewer


class TestCliBranch:
    def test_cli_review_forwards_input_annotations_config_fisheye(self, cli_reviewer):
        annotations = _annotations()
        cli_reviewer.return_value = (
            annotations,
            {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 3},
        )
        kwargs = _stage_kwargs(_make_ctx(), annotations)

        run_manual_review_stage(**kwargs)

        cli_reviewer.assert_called_once()
        args, call_kwargs = cli_reviewer.call_args
        assert call_kwargs == {}
        assert len(args) == 4
        assert args[0] == "missing.mp4"
        assert args[1] is annotations
        assert args[2] is kwargs["config"]
        assert args[3] is kwargs["fisheye"]

    def test_cli_review_returns_reviewer_annotations_and_stats(self, cli_reviewer):
        reviewed = {7: {"auto": [], "manual": [list(POLY_B)], "intensities": []}}
        stats = {"added": 4, "removed": 2, "frames_modified": 1, "frames_reviewed": 3}
        cli_reviewer.return_value = (reviewed, stats)

        result_ann, result_stats = run_manual_review_stage(
            **_stage_kwargs(_make_ctx(), _annotations())
        )

        assert result_ann is reviewed
        assert result_stats is stats

    def test_cli_review_prints_instructions_and_summary(self, cli_reviewer, capsys):
        cli_reviewer.return_value = (
            _annotations(),
            {"added": 4, "removed": 2, "frames_modified": 1, "frames_reviewed": 3},
        )

        run_manual_review_stage(**_stage_kwargs(_make_ctx(), _annotations(), total_frames=4))

        out = capsys.readouterr().out
        assert "[FASE 3/5] Revisione manuale — apertura interfaccia..." in out
        assert (
            "Usa Spazio per navigare, Click per disegnare, D per eliminare, Q per confermare."
            in out
        )
        assert "  Revisione completata:" in out
        assert "  Frame revisionati:     3 / 4\n" in out
        assert "  Poligoni aggiunti:     4\n" in out
        assert "  Poligoni rimossi:      2\n" in out
        assert "  Frame modificati:      1  (25.0%)" in out
        assert "in attesa di conferma dal browser" not in out

    def test_cli_review_zero_total_frames_prints_not_available(self, cli_reviewer, capsys):
        cli_reviewer.return_value = (
            {},
            {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0},
        )

        run_manual_review_stage(**_stage_kwargs(_make_ctx(), {}, total_frames=0))

        out = capsys.readouterr().out
        assert "  Frame revisionati:     0 / 0\n" in out
        assert "  Frame modificati:      0  (N/A)" in out

    def test_cli_review_does_not_touch_web_collaborators(self, cli_reviewer):
        sse = MagicMock()
        cli_reviewer.return_value = (
            _annotations(),
            {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 3},
        )
        ctx = _make_ctx(review_state=None, sse_manager=sse, job_id="job-42")

        run_manual_review_stage(**_stage_kwargs(ctx, _annotations()))

        sse.emit.assert_not_called()
