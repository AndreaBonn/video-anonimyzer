"""Test del core puro ReviewEditor: nessun cv2.VideoCapture, nessuna finestra."""

import pytest

from person_anonymizer.review_editor import ReviewEditor, new_annotation, point_in_polygon

SQUARE_AUTO = [(10, 10), (50, 10), (50, 50), (10, 50)]
SQUARE_OTHER = [(100, 100), (150, 100), (150, 150), (100, 150)]
SQUARE_MANUAL = [(5, 5), (55, 5), (55, 55), (5, 55)]
TRIANGLE = [(0, 0), (10, 0), (5, 10)]
TOTAL_FRAMES = 3


@pytest.fixture()
def editor() -> ReviewEditor:
    annotations = {
        0: {"auto": [SQUARE_AUTO], "manual": [], "intensities": [1.0]},
        1: new_annotation(),
    }
    return ReviewEditor(annotations=annotations, total_frames=TOTAL_FRAMES)


# --- point_in_polygon -------------------------------------------------------


def test_point_in_polygon_inside_is_true() -> None:
    assert point_in_polygon((50, 50), [(0, 0), (100, 0), (100, 100), (0, 100)]) is True


def test_point_in_polygon_outside_is_false() -> None:
    assert point_in_polygon((200, 200), [(0, 0), (100, 0), (100, 100), (0, 100)]) is False


def test_point_in_polygon_on_edge_is_true() -> None:
    assert point_in_polygon((0, 50), [(0, 0), (100, 0), (100, 100), (0, 100)]) is True


# --- init / visit -----------------------------------------------------------


def test_init_deep_copies_annotations() -> None:
    original = {0: {"auto": [SQUARE_AUTO], "manual": [], "intensities": [1.0]}}
    copy = ReviewEditor(annotations=original, total_frames=TOTAL_FRAMES)
    copy.annotations[0]["auto"].clear()
    assert original[0]["auto"] == [SQUARE_AUTO]


def test_init_starts_clean(editor: ReviewEditor) -> None:
    assert editor.current_frame_idx == 0
    assert editor.pending_points == []
    assert editor.delete_mode is False
    assert editor.final_stats() == {
        "added": 0,
        "removed": 0,
        "frames_modified": 0,
        "frames_reviewed": 0,
    }


def test_visit_current_creates_missing_entry_and_marks_reviewed(editor: ReviewEditor) -> None:
    editor.current_frame_idx = 2
    entry = editor.visit_current()
    assert entry == {"auto": [], "manual": [], "intensities": []}
    assert editor.annotations[2] is entry
    assert editor.stats["frames_reviewed"] == {2}


def test_visit_current_keeps_existing_entry(editor: ReviewEditor) -> None:
    entry = editor.visit_current()
    assert entry["auto"] == [SQUARE_AUTO]
    assert entry["intensities"] == [1.0]


def test_current_annotation_for_frame_without_entry_is_empty_and_not_stored(
    editor: ReviewEditor,
) -> None:
    editor.current_frame_idx = 99
    assert editor.current_annotation() == {"auto": [], "manual": [], "intensities": []}
    assert 99 not in editor.annotations


# --- drawing ----------------------------------------------------------------


def test_add_point_then_undo_removes_only_last(editor: ReviewEditor) -> None:
    for point in TRIANGLE:
        editor.add_point(point)
    editor.undo_point()
    assert editor.pending_points == [(0, 0), (10, 0)]


def test_undo_point_without_points_is_noop(editor: ReviewEditor) -> None:
    editor.undo_point()
    assert editor.pending_points == []


def test_close_polygon_stores_points_and_counts(editor: ReviewEditor) -> None:
    editor.current_frame_idx = 5
    for point in TRIANGLE:
        editor.add_point(point)
    assert editor.close_polygon() is True
    assert editor.annotations[5]["manual"] == [TRIANGLE]
    assert editor.pending_points == []
    assert editor.final_stats()["added"] == 1
    assert editor.stats["frames_modified"] == {5}


def test_close_polygon_with_two_points_is_rejected_and_keeps_points(editor: ReviewEditor) -> None:
    editor.add_point((0, 0))
    editor.add_point((10, 0))
    assert editor.close_polygon() is False
    assert editor.pending_points == [(0, 0), (10, 0)]
    assert editor.annotations[0]["manual"] == []
    assert editor.final_stats()["added"] == 0


def test_cancel_polygon_discards_points(editor: ReviewEditor) -> None:
    editor.add_point((1, 1))
    editor.cancel_polygon()
    assert editor.pending_points == []


def test_toggle_delete_mode_flips_and_discards_points(editor: ReviewEditor) -> None:
    editor.add_point((1, 1))
    editor.toggle_delete_mode()
    assert editor.delete_mode is True
    assert editor.pending_points == []
    editor.toggle_delete_mode()
    assert editor.delete_mode is False


def test_click_adds_point_in_normal_mode_and_deletes_in_delete_mode(editor: ReviewEditor) -> None:
    editor.click((30, 30))
    assert editor.pending_points == [(30, 30)]
    assert editor.annotations[0]["auto"] == [SQUARE_AUTO]
    editor.toggle_delete_mode()
    editor.click((30, 30))
    assert editor.annotations[0]["auto"] == []
    assert editor.pending_points == []


# --- delete_at --------------------------------------------------------------


def test_delete_at_removes_manual_polygon(editor: ReviewEditor) -> None:
    editor.annotations[0]["manual"] = [SQUARE_MANUAL]
    assert editor.delete_at((30, 30)) is True
    assert editor.annotations[0]["manual"] == []
    assert editor.stats["removed"] == 1


def test_delete_at_removes_auto_polygon_and_its_intensity(editor: ReviewEditor) -> None:
    editor.annotations[0]["intensities"] = [0.5]
    assert editor.delete_at((30, 30)) is True
    assert editor.annotations[0]["auto"] == []
    assert editor.annotations[0]["intensities"] == []


def test_delete_at_keeps_other_intensities_aligned(editor: ReviewEditor) -> None:
    editor.annotations[0] = {
        "auto": [SQUARE_OTHER, SQUARE_AUTO],
        "manual": [],
        "intensities": [10, 20],
    }
    editor.delete_at((30, 30))
    assert editor.annotations[0]["auto"] == [SQUARE_OTHER]
    assert editor.annotations[0]["intensities"] == [10]


def test_delete_at_with_intensities_shorter_than_auto_does_not_fail(editor: ReviewEditor) -> None:
    editor.annotations[0] = {"auto": [SQUARE_OTHER, SQUARE_AUTO], "manual": [], "intensities": [10]}
    assert editor.delete_at((30, 30)) is True
    assert editor.annotations[0]["auto"] == [SQUARE_OTHER]
    assert editor.annotations[0]["intensities"] == [10]


def test_delete_at_removes_the_polygon_under_the_point_not_the_first(editor: ReviewEditor) -> None:
    editor.annotations[0]["manual"] = [SQUARE_OTHER, SQUARE_MANUAL]
    editor.delete_at((30, 30))
    assert editor.annotations[0]["manual"] == [SQUARE_OTHER]


def test_delete_at_prefers_manual_over_overlapping_auto(editor: ReviewEditor) -> None:
    editor.annotations[0]["manual"] = [SQUARE_MANUAL]
    editor.delete_at((30, 30))
    assert editor.annotations[0]["auto"] == [SQUARE_AUTO]
    assert editor.annotations[0]["manual"] == []


def test_delete_at_without_match_changes_nothing(editor: ReviewEditor) -> None:
    assert editor.delete_at((999, 999)) is False
    assert editor.stats["removed"] == 0
    assert editor.annotations[0]["auto"] == [SQUARE_AUTO]


def test_delete_at_marks_frame_modified(editor: ReviewEditor) -> None:
    editor.annotations[0]["manual"] = [SQUARE_MANUAL]
    editor.delete_at((30, 30))
    assert editor.stats["frames_modified"] == {0}


def test_delete_at_frame_without_entry_is_noop_and_creates_nothing(editor: ReviewEditor) -> None:
    editor.current_frame_idx = 99
    assert editor.delete_at((30, 30)) is False
    assert editor.stats["removed"] == 0
    assert 99 not in editor.annotations


# --- navigation -------------------------------------------------------------


def test_go_next_advances_and_clamps_at_last_frame(editor: ReviewEditor) -> None:
    for _ in range(TOTAL_FRAMES + 2):
        editor.go_next()
    assert editor.current_frame_idx == TOTAL_FRAMES - 1


def test_go_previous_goes_back_and_clamps_at_first_frame(editor: ReviewEditor) -> None:
    editor.current_frame_idx = 2
    editor.go_previous()
    assert editor.current_frame_idx == 1
    editor.go_previous()
    editor.go_previous()
    assert editor.current_frame_idx == 0


def test_navigation_clears_pending_points_and_delete_mode(editor: ReviewEditor) -> None:
    editor.add_point((1, 1))
    editor.delete_mode = True
    editor.go_next()
    assert editor.pending_points == []
    assert editor.delete_mode is False
    editor.add_point((2, 2))
    editor.delete_mode = True
    editor.go_previous()
    assert editor.pending_points == []
    assert editor.delete_mode is False


def test_single_frame_video_does_not_navigate() -> None:
    editor = ReviewEditor(annotations={}, total_frames=1)
    editor.go_next()
    editor.go_previous()
    assert editor.current_frame_idx == 0


# --- stats ------------------------------------------------------------------


def test_final_stats_reduces_sets_to_counts(editor: ReviewEditor) -> None:
    editor.stats["added"] = 3
    editor.stats["removed"] = 1
    editor.stats["frames_modified"] = {0, 1, 2}
    editor.stats["frames_reviewed"] = {0, 1}
    assert editor.final_stats() == {
        "added": 3,
        "removed": 1,
        "frames_modified": 3,
        "frames_reviewed": 2,
    }
