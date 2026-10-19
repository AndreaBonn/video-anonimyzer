import hashlib
from collections.abc import Callable
from copy import deepcopy
from itertools import product
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.manual_reviewer import (
    KEY_BACKSPACE_CODES,
    KEY_ENTER,
    KEY_ESC,
    KEY_LEFT_CODES,
    KEY_NONE,
    KEY_RIGHT_CODES,
    KEY_SPACE,
    run_manual_review,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_COUNT = 3
TRIANGLE = [(100, 100), (180, 100), (140, 180)]
AUTO_A = [(40, 80), (80, 80), (80, 120), (40, 120)]
AUTO_B = [(250, 100), (350, 100), (350, 200), (250, 200)]
Step = tuple[int, list[tuple[int, int]]]
Annotations = dict[int, dict[str, list]]
SessionResult = tuple[Annotations, dict[str, int], list[np.ndarray]]


def close_polygon(points: list[tuple[int, int]]) -> Step:
    return KEY_ENTER, points


@pytest.fixture
def scripted_review(monkeypatch: pytest.MonkeyPatch) -> Callable[..., SessionResult]:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: FRAME_COUNT,
        cv2.CAP_PROP_FRAME_WIDTH: FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT: FRAME_HEIGHT,
    }[prop]
    capture.read.side_effect = lambda: (
        True,
        np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda video_path: capture)
    for name in ("namedWindow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, MagicMock())

    def run(
        steps: list[Step],
        annotations: Annotations | None = None,
        config: PipelineConfig | None = None,
    ) -> SessionResult:
        callback = MagicMock()
        displays = []
        actions = iter([*steps, (ord("q"), [])])

        def wait_key(delay: int) -> int:
            key, clicks = next(actions)
            for x, y in clicks:
                callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return key

        def register_callback(window: str, handler: Callable) -> None:
            nonlocal callback
            callback = handler

        monkeypatch.setattr(cv2, "setMouseCallback", register_callback)
        monkeypatch.setattr(cv2, "waitKeyEx", wait_key)
        monkeypatch.setattr(cv2, "imshow", lambda window, frame: displays.append(frame.copy()))
        result, stats = run_manual_review(
            video_path="scripted.mp4",
            auto_annotations=annotations if annotations is not None else {},
            config=config if config is not None else PipelineConfig(),
        )
        capture.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()
        return result, stats, displays

    return run


def test_session_enter_closes_scaled_triangle(scripted_review: Callable) -> None:
    annotations, stats, _ = scripted_review(
        steps=[close_polygon(points=[(50, 50), (90, 50), (70, 90)])],
        config=PipelineConfig(review_window_max_width=320),
    )
    assert annotations == {0: {"auto": [], "manual": [TRIANGLE], "intensities": []}}
    assert stats == {"added": 1, "removed": 0, "frames_modified": 1, "frames_reviewed": 1}


def test_session_enter_requires_three_points(scripted_review: Callable) -> None:
    annotations, stats, _ = scripted_review(steps=[close_polygon(points=TRIANGLE[:2])])
    assert annotations == {0: {"auto": [], "manual": [], "intensities": []}}
    assert stats == {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 1}


@pytest.mark.parametrize("backspace", sorted(KEY_BACKSPACE_CODES))
def test_session_backspace_removes_only_last_point(
    scripted_review: Callable, backspace: int
) -> None:
    annotations, stats, _ = scripted_review(
        steps=[(backspace, [*TRIANGLE, (220, 220)]), (KEY_ENTER, [])]
    )
    assert annotations[0]["manual"] == [TRIANGLE]
    assert stats == {"added": 1, "removed": 0, "frames_modified": 1, "frames_reviewed": 1}


def test_session_escape_discards_polygon_before_new_drawing(scripted_review: Callable) -> None:
    annotations, stats, _ = scripted_review(
        steps=[(KEY_ESC, AUTO_A), (KEY_ENTER, []), close_polygon(points=TRIANGLE)]
    )
    assert annotations[0]["manual"] == [TRIANGLE]
    assert stats == {"added": 1, "removed": 0, "frames_modified": 1, "frames_reviewed": 1}


@pytest.mark.parametrize("click, removed", [((300, 150), 1), ((500, 300), 0)])
def test_session_delete_keeps_matching_intensity(
    scripted_review: Callable, click: tuple[int, int], removed: int
) -> None:
    annotations, stats, _ = scripted_review(
        steps=[(ord("d"), []), (KEY_NONE, [click])],
        annotations={0: {"auto": [AUTO_A, AUTO_B], "manual": [], "intensities": [10, 20]}},
    )
    assert annotations[0] == {
        "auto": [AUTO_A] if removed else [AUTO_A, AUTO_B],
        "manual": [],
        "intensities": [10] if removed else [10, 20],
    }
    assert stats == {
        "added": 0,
        "removed": removed,
        "frames_modified": removed,
        "frames_reviewed": 1,
    }


def test_session_delete_prefers_manual_over_overlapping_auto(scripted_review: Callable) -> None:
    annotations, stats, _ = scripted_review(
        steps=[(ord("d"), []), (KEY_NONE, [(300, 150)])],
        annotations={0: {"auto": [AUTO_B], "manual": [AUTO_B], "intensities": [20]}},
    )
    assert annotations[0] == {"auto": [AUTO_B], "manual": [], "intensities": [20]}
    assert stats == {"added": 0, "removed": 1, "frames_modified": 1, "frames_reviewed": 1}


@pytest.mark.parametrize("left,right", sorted(product(KEY_LEFT_CODES, KEY_RIGHT_CODES)))
def test_session_navigation_clamps_and_counts_distinct_frames(
    scripted_review: Callable, left: int, right: int
) -> None:
    annotations, stats, _ = scripted_review(
        steps=[
            (left, []),
            close_polygon(points=TRIANGLE),
            (right, []),
            close_polygon(points=TRIANGLE),
            (KEY_SPACE, []),
            close_polygon(points=TRIANGLE),
            (right, []),
            close_polygon(points=TRIANGLE),
            (left, []),
            close_polygon(points=TRIANGLE),
        ]
    )
    assert annotations == {
        index: {"auto": [], "manual": [TRIANGLE] * count, "intensities": []}
        for index, count in enumerate([1, 2, 2])
    }
    assert stats == {"added": 5, "removed": 0, "frames_modified": 3, "frames_reviewed": 3}


@pytest.mark.parametrize("navigation", [*sorted(KEY_LEFT_CODES | KEY_RIGHT_CODES), KEY_SPACE])
def test_session_navigation_discards_points_and_resets_delete_mode(
    scripted_review: Callable, navigation: int
) -> None:
    target = 0 if navigation in KEY_LEFT_CODES else FRAME_COUNT - 1
    annotations, stats, _ = scripted_review(
        steps=[
            (KEY_SPACE, []),
            (navigation, AUTO_A),
            (KEY_ENTER, []),
            (ord("d"), []),
            (navigation, []),
            close_polygon(points=TRIANGLE),
        ]
    )
    assert annotations == {
        index: {"auto": [], "manual": [TRIANGLE] if index == target else [], "intensities": []}
        for index in (0, 1, target)
    }
    assert stats == {
        "added": 1,
        "removed": 0,
        "frames_modified": 1,
        "frames_reviewed": 3 if target == 2 else 2,
    }


def test_session_preserves_input_annotations_after_edits(scripted_review: Callable) -> None:
    original = {0: {"auto": [AUTO_A, AUTO_B], "manual": [TRIANGLE], "intensities": [10, 20]}}
    before = deepcopy(original)
    annotations, stats, _ = scripted_review(
        steps=[
            (ord("d"), []),
            (KEY_NONE, [(300, 150)]),
            (ord("d"), []),
            close_polygon(points=AUTO_A),
            (KEY_SPACE, []),
        ],
        annotations=original,
    )
    assert original == before
    assert annotations == {
        0: {"auto": [AUTO_A], "manual": [TRIANGLE, AUTO_A], "intensities": [10]},
        1: {"auto": [], "manual": [], "intensities": []},
    }
    assert stats == {"added": 1, "removed": 1, "frames_modified": 1, "frames_reviewed": 2}


def test_render_manual_polygon_blends_fill_and_preserves_background(
    scripted_review: Callable,
) -> None:
    manual = [(100, 100), (200, 100), (200, 200), (100, 200)]
    _, _, displays = scripted_review(
        steps=[],
        annotations={0: {"auto": [], "manual": [manual], "intensities": []}},
        config=PipelineConfig(review_fill_alpha=0.5, review_window_max_width=FRAME_WIDTH),
    )
    frame = displays[0]
    np.testing.assert_allclose(frame[150, 150], [0, 60, 128], atol=1, rtol=0)
    np.testing.assert_array_equal(frame[300, 500], [0, 0, 0])
    np.testing.assert_array_equal(frame[0, 500], [40, 40, 40])


def test_render_display_snapshot_auto_manual_and_drawing(scripted_review: Callable) -> None:
    _, _, displays = scripted_review(
        steps=[(KEY_NONE, TRIANGLE[:2])],
        annotations={0: {"auto": [AUTO_A], "manual": [AUTO_B], "intensities": [10]}},
        config=PipelineConfig(
            review_auto_color=(0, 255, 0),
            review_manual_color=(0, 120, 255),
            review_drawing_color=(255, 255, 0),
            review_fill_alpha=0.5,
            review_window_max_width=FRAME_WIDTH,
        ),
    )
    frame = displays[-1]
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    np.testing.assert_allclose(frame[100, 60], [0, 128, 0], atol=1, rtol=0)  # inside auto
    np.testing.assert_allclose(frame[150, 300], [0, 60, 128], atol=1, rtol=0)  # inside manual
    np.testing.assert_allclose(frame[100, 140], [255, 255, 0], atol=1, rtol=0)  # pending segment
    np.testing.assert_array_equal(frame[400, 500], [0, 0, 0])  # far background
    np.testing.assert_array_equal(frame[0, 500], [40, 40, 40])  # info bar, no text
    # The hash depends on the pinned opencv version (4.13.0.92).
    assert hashlib.sha256(displays[-1].tobytes()).hexdigest() == (
        "155896817595e298c4adfd8d25b3d0d730ddabddcfcb21c0f3d105e5a6a556a2"
    )
