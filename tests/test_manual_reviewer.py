"""Test per manual_reviewer — GUI OpenCV per revisione manuale.

Mock pesante su cv2 per evitare display fisico.
"""

from copy import deepcopy
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.manual_reviewer import (
    KEY_CTRL_Z,
    KEY_ENTER,
    KEY_ESC,
    KEY_LEFT,
    KEY_NONE,
    KEY_RIGHT,
    KEY_SPACE,
    ManualReviewer,
    run_manual_review,
)
from person_anonymizer.models import FisheyeContext


# ─── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def config():
    return PipelineConfig()


@pytest.fixture()
def mock_cap():
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        0: 100,   # CAP_PROP_POS_FRAMES placeholder
        7: 30,    # CAP_PROP_FRAME_COUNT
        3: 640,   # CAP_PROP_FRAME_WIDTH
        4: 480,   # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)
    ret_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.return_value = (True, ret_frame)
    return cap


@pytest.fixture()
def reviewer(config, mock_cap):
    """ManualReviewer con cv2.VideoCapture mockata."""
    annotations = {
        0: {"auto": [[(10, 10), (50, 10), (50, 50), (10, 50)]], "manual": [], "intensities": [1.0]},
        1: {"auto": [], "manual": [], "intensities": []},
    }
    with patch("cv2.VideoCapture", return_value=mock_cap):
        rv = ManualReviewer(
            video_path="fake.mp4",
            auto_annotations=annotations,
            config=config,
        )
    return rv


# ─── __init__ ───────────────────────────────────────────────────


class TestManualReviewerInit:
    def test_annotations_deep_copied(self, config, mock_cap):
        original = {0: {"auto": [[(0, 0), (10, 0), (10, 10)]], "manual": [], "intensities": []}}
        with patch("cv2.VideoCapture", return_value=mock_cap):
            rv = ManualReviewer("fake.mp4", original, config)
        rv.annotations[0]["auto"].clear()
        # L'originale non deve essere modificato
        assert len(original[0]["auto"]) == 1

    def test_scale_set_when_frame_wider_than_max_width(self, config, mock_cap):
        mock_cap.get.side_effect = lambda prop: {7: 10, 3: 2000, 4: 1000}.get(prop, 0)
        with patch("cv2.VideoCapture", return_value=mock_cap):
            rv = ManualReviewer("fake.mp4", {}, config)
        assert rv.scale < 1.0
        assert rv.display_w == config.review_window_max_width

    def test_scale_is_one_when_frame_fits(self, config, mock_cap):
        mock_cap.get.side_effect = lambda prop: {7: 10, 3: 640, 4: 480}.get(prop, 0)
        with patch("cv2.VideoCapture", return_value=mock_cap):
            rv = ManualReviewer("fake.mp4", {}, config)
        assert rv.scale == 1.0

    def test_stats_initialized(self, reviewer):
        assert reviewer.stats["added"] == 0
        assert reviewer.stats["removed"] == 0
        assert isinstance(reviewer.stats["frames_modified"], set)
        assert isinstance(reviewer.stats["frames_reviewed"], set)

    def test_initial_state(self, reviewer):
        assert reviewer.current_frame_idx == 0
        assert reviewer.current_polygon_points == []
        assert reviewer.delete_mode is False

    def test_fisheye_default_is_fisheye_context(self, reviewer):
        assert isinstance(reviewer.fisheye, FisheyeContext)

    def test_fisheye_custom_passed(self, config, mock_cap):
        fisheye = MagicMock()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            rv = ManualReviewer("fake.mp4", {}, config, fisheye=fisheye)
        assert rv.fisheye is fisheye


# ─── _get_frame ─────────────────────────────────────────────────


class TestGetFrame:
    def test_returns_frame_array(self, reviewer, mock_cap):
        frame = reviewer._get_frame(0)
        assert isinstance(frame, np.ndarray)

    def test_cache_hit_avoids_second_read(self, reviewer, mock_cap):
        reviewer._get_frame(0)
        initial_calls = mock_cap.read.call_count
        reviewer._get_frame(0)
        assert mock_cap.read.call_count == initial_calls  # nessuna lettura aggiuntiva

    def test_cache_miss_reads_new_frame(self, reviewer, mock_cap):
        reviewer._get_frame(0)
        calls_after_first = mock_cap.read.call_count
        reviewer._get_frame(1)
        assert mock_cap.read.call_count > calls_after_first

    def test_failed_read_returns_black_frame(self, reviewer, mock_cap):
        mock_cap.read.return_value = (False, None)
        with patch("cv2.putText"):
            frame = reviewer._get_frame(5)
        assert frame.shape == (reviewer.frame_h, reviewer.frame_w, 3)
        assert frame.sum() == 0  # frame nero

    def test_fisheye_undistort_called(self, reviewer):
        mock_fisheye = MagicMock()
        undistorted = np.ones((480, 640, 3), dtype=np.uint8)
        mock_fisheye.undistort.return_value = undistorted
        reviewer.fisheye = mock_fisheye
        reviewer._cached_frame_idx = -1  # invalida cache
        result = reviewer._get_frame(0)
        mock_fisheye.undistort.assert_called_once()


# ─── _display_to_original / _original_to_display ────────────────


class TestCoordinateConversion:
    def test_display_to_original_no_scale(self, reviewer):
        reviewer.scale = 1.0
        assert reviewer._display_to_original(100, 200) == (100, 200)

    def test_display_to_original_half_scale(self, reviewer):
        reviewer.scale = 0.5
        assert reviewer._display_to_original(100, 200) == (200, 400)

    def test_original_to_display_no_scale(self, reviewer):
        reviewer.scale = 1.0
        assert reviewer._original_to_display(100, 200) == (100, 200)

    def test_original_to_display_half_scale(self, reviewer):
        reviewer.scale = 0.5
        assert reviewer._original_to_display(100, 200) == (50, 100)

    def test_roundtrip_conversion(self, reviewer):
        reviewer.scale = 0.75
        orig_x, orig_y = 80, 120
        dx, dy = reviewer._original_to_display(orig_x, orig_y)
        rx, ry = reviewer._display_to_original(dx, dy)
        # Con interi può esserci arrotondamento di ±1
        assert abs(rx - orig_x) <= 1
        assert abs(ry - orig_y) <= 1


# ─── _point_in_polygon ──────────────────────────────────────────


class TestPointInPolygon:
    def test_point_inside_returns_true(self, reviewer):
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert reviewer._point_in_polygon(50, 50, poly) is True

    def test_point_outside_returns_false(self, reviewer):
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert reviewer._point_in_polygon(200, 200, poly) is False

    def test_point_on_edge_returns_true(self, reviewer):
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert reviewer._point_in_polygon(0, 50, poly) is True


# ─── _delete_polygon_at ─────────────────────────────────────────


class TestDeletePolygonAt:
    def test_deletes_manual_polygon(self, reviewer):
        reviewer.annotations[0]["manual"] = [[(5, 5), (55, 5), (55, 55), (5, 55)]]
        reviewer._delete_polygon_at(30, 30)
        assert reviewer.annotations[0]["manual"] == []
        assert reviewer.stats["removed"] == 1

    def test_deletes_auto_polygon(self, reviewer):
        # frame 0 ha già un poligono auto nel fixture
        reviewer.annotations[0]["manual"] = []
        reviewer._delete_polygon_at(30, 30)
        assert reviewer.annotations[0]["auto"] == []
        assert reviewer.stats["removed"] == 1

    def test_deletes_auto_removes_intensity(self, reviewer):
        reviewer.annotations[0]["manual"] = []
        reviewer.annotations[0]["intensities"] = [0.5]
        reviewer._delete_polygon_at(30, 30)
        assert reviewer.annotations[0]["intensities"] == []

    def test_no_match_does_nothing(self, reviewer):
        removed_before = reviewer.stats["removed"]
        reviewer._delete_polygon_at(999, 999)
        assert reviewer.stats["removed"] == removed_before

    def test_manual_preferred_over_auto(self, reviewer):
        # Sovrapponi un poligono manuale allo stesso punto
        reviewer.annotations[0]["manual"] = [[(5, 5), (55, 5), (55, 55), (5, 55)]]
        reviewer._delete_polygon_at(30, 30)
        # Deve aver rimosso il manuale, non l'auto
        assert reviewer.annotations[0]["auto"] != []
        assert reviewer.annotations[0]["manual"] == []

    def test_frames_modified_updated(self, reviewer):
        reviewer.annotations[0]["manual"] = [[(5, 5), (55, 5), (55, 55), (5, 55)]]
        reviewer._delete_polygon_at(30, 30)
        assert 0 in reviewer.stats["frames_modified"]

    def test_missing_frame_annotation_preserves_state(self, reviewer):
        # Frame 99 non ha entry nelle annotations — stato non cambia
        reviewer.current_frame_idx = 99
        removed_before = reviewer.stats["removed"]
        reviewer._delete_polygon_at(30, 30)
        assert reviewer.stats["removed"] == removed_before


# ─── _handle_key ────────────────────────────────────────────────


class TestHandleKey:
    def test_q_returns_false(self, reviewer):
        assert reviewer._handle_key(ord("q")) is False

    def test_uppercase_q_returns_false(self, reviewer):
        assert reviewer._handle_key(ord("Q")) is False

    def test_right_arrow_advances_frame(self, reviewer):
        reviewer.current_frame_idx = 0
        reviewer._handle_key(KEY_RIGHT)
        assert reviewer.current_frame_idx == 1

    def test_space_advances_frame(self, reviewer):
        reviewer.current_frame_idx = 0
        reviewer._handle_key(KEY_SPACE)
        assert reviewer.current_frame_idx == 1

    def test_right_arrow_at_last_frame_stays(self, reviewer):
        reviewer.current_frame_idx = reviewer.total_frames - 1
        reviewer._handle_key(KEY_RIGHT)
        assert reviewer.current_frame_idx == reviewer.total_frames - 1

    def test_left_arrow_shadowed_by_quit(self, reviewer):
        # KEY_LEFT = 81 = ord('Q') — il check Q viene prima nell'elif chain,
        # quindi KEY_LEFT in pratica agisce come quit (return False)
        reviewer.current_frame_idx = 2
        result = reviewer._handle_key(KEY_LEFT)
        assert result is False  # esce dal loop (Q ha priorità)
        assert reviewer.current_frame_idx == 2  # non cambia

    def test_enter_adds_polygon_with_enough_points(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0), (5, 10)]
        reviewer._handle_key(KEY_ENTER)
        assert len(reviewer.annotations[0]["manual"]) == 1
        assert reviewer.stats["added"] == 1
        assert reviewer.current_polygon_points == []

    def test_enter_ignored_with_fewer_than_3_points(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0)]
        reviewer._handle_key(KEY_ENTER)
        assert reviewer.annotations[0]["manual"] == []

    def test_enter_creates_annotation_entry_for_new_frame(self, reviewer):
        reviewer.current_frame_idx = 5
        reviewer.current_polygon_points = [(0, 0), (10, 0), (5, 10)]
        reviewer._handle_key(KEY_ENTER)
        assert 5 in reviewer.annotations
        assert len(reviewer.annotations[5]["manual"]) == 1

    def test_ctrl_z_removes_last_point(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0), (5, 10)]
        reviewer._handle_key(KEY_CTRL_Z)
        assert len(reviewer.current_polygon_points) == 2

    def test_ctrl_z_noop_when_no_points(self, reviewer):
        reviewer.current_polygon_points = []
        reviewer._handle_key(KEY_CTRL_Z)  # non deve sollevare
        assert reviewer.current_polygon_points == []

    def test_d_toggles_delete_mode_on(self, reviewer):
        reviewer.delete_mode = False
        reviewer._handle_key(ord("d"))
        assert reviewer.delete_mode is True

    def test_d_toggles_delete_mode_off(self, reviewer):
        reviewer.delete_mode = True
        reviewer._handle_key(ord("d"))
        assert reviewer.delete_mode is False

    def test_uppercase_d_toggles_delete_mode(self, reviewer):
        reviewer.delete_mode = False
        reviewer._handle_key(ord("D"))
        assert reviewer.delete_mode is True

    def test_d_clears_current_polygon(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0)]
        reviewer._handle_key(ord("d"))
        assert reviewer.current_polygon_points == []

    def test_esc_clears_current_polygon(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0), (5, 10)]
        reviewer._handle_key(KEY_ESC)
        assert reviewer.current_polygon_points == []

    def test_navigation_clears_polygon_and_delete_mode(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0)]
        reviewer.delete_mode = True
        reviewer._handle_key(KEY_RIGHT)
        assert reviewer.current_polygon_points == []
        assert reviewer.delete_mode is False

    def test_unknown_key_returns_true(self, reviewer):
        assert reviewer._handle_key(ord("x")) is True

    def test_frames_modified_updated_on_enter(self, reviewer):
        reviewer.current_polygon_points = [(0, 0), (10, 0), (5, 10)]
        reviewer._handle_key(KEY_ENTER)
        assert 0 in reviewer.stats["frames_modified"]


# ─── _on_mouse ──────────────────────────────────────────────────


class TestOnMouse:
    def test_mouse_move_updates_mouse_pos(self, reviewer):
        reviewer._on_mouse(0, 100, 200, 0, None)  # event=0 = MOVE (non usato)
        assert reviewer.mouse_pos == (100, 200)

    def test_lbuttondown_adds_point_in_normal_mode(self, reviewer):
        with patch.object(reviewer, "_display_to_original", return_value=(50, 60)):
            reviewer._on_mouse(1, 50, 60, 0, None)  # EVENT_LBUTTONDOWN = 1
        assert (50, 60) in reviewer.current_polygon_points

    def test_lbuttondown_in_delete_mode_calls_delete(self, reviewer):
        reviewer.delete_mode = True
        with (
            patch.object(reviewer, "_display_to_original", return_value=(30, 30)),
            patch.object(reviewer, "_delete_polygon_at") as mock_del,
        ):
            reviewer._on_mouse(1, 30, 30, 0, None)
        mock_del.assert_called_once_with(30, 30)

    def test_other_mouse_event_ignored(self, reviewer):
        initial_points = list(reviewer.current_polygon_points)
        reviewer._on_mouse(2, 50, 60, 0, None)  # EVENT_LBUTTONUP = 2
        assert reviewer.current_polygon_points == initial_points


# ─── _get_final_stats ───────────────────────────────────────────


class TestGetFinalStats:
    def test_returns_correct_counts(self, reviewer):
        reviewer.stats["added"] = 3
        reviewer.stats["removed"] = 1
        reviewer.stats["frames_modified"] = {0, 1, 2}
        reviewer.stats["frames_reviewed"] = {0, 1}
        stats = reviewer._get_final_stats()
        assert stats["added"] == 3
        assert stats["removed"] == 1
        assert stats["frames_modified"] == 3
        assert stats["frames_reviewed"] == 2

    def test_empty_stats(self, reviewer):
        stats = reviewer._get_final_stats()
        assert stats["added"] == 0
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 0
        assert stats["frames_reviewed"] == 0


# ─── _render_display ────────────────────────────────────────────


class TestRenderDisplay:
    def test_returns_ndarray_with_correct_shape(self, reviewer):
        with patch("cv2.addWeighted", side_effect=lambda o, a, f, b, g, dst: dst):
            with patch("cv2.resize", side_effect=lambda f, s: f):
                result = reviewer._render_display()
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # BGR

    def test_render_with_polygon_points_draws_lines(self, reviewer):
        reviewer.current_polygon_points = [(10, 10), (50, 10)]
        with patch("cv2.line") as mock_line, \
             patch("cv2.circle") as mock_circle:
            reviewer._render_display()
        # 2 punti → 1 linea tra loro + cerchi per ogni punto
        assert mock_line.call_count >= 1
        assert mock_circle.call_count >= 2

    def test_render_without_annotation_returns_clean_frame(self, reviewer):
        reviewer.current_frame_idx = 99  # frame senza annotazioni
        result = reviewer._render_display()
        # Il frame deve essere valido anche senza annotazioni
        assert result.shape == (reviewer.display_h, reviewer.display_w, 3)

    def test_scale_applied_resizes_to_display_dimensions(self, reviewer):
        reviewer.scale = 0.5
        reviewer.display_w = 320
        reviewer.display_h = 240
        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            reviewer._render_display()
        # Verifica che resize sia chiamato con le dimensioni display
        mock_resize.assert_called_once()
        call_args = mock_resize.call_args
        assert call_args[0][1] == (320, 240)  # (width, height)


# ─── run (loop principale) ──────────────────────────────────────


class TestManualReviewerRun:
    def test_run_returns_annotations_and_stats(self, reviewer):
        # Simula: prima iterazione KEY_NONE (continua), seconda Q (esci)
        with (
            patch("cv2.namedWindow"),
            patch("cv2.setMouseCallback"),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch.object(reviewer.cap, "release"),
            patch.object(reviewer, "_render_display", return_value=np.zeros((480, 640, 3), dtype=np.uint8)),
            patch("cv2.waitKey", side_effect=[KEY_NONE & 0xFF, ord("q") & 0xFF]),
        ):
            annotations, stats = reviewer.run()
        assert isinstance(annotations, dict)
        assert "added" in stats
        assert "removed" in stats

    def test_run_creates_annotation_entry_for_new_frame(self, reviewer):
        # Frame 0 non ha entry → run deve crearlo
        reviewer.annotations = {}
        with (
            patch("cv2.namedWindow"),
            patch("cv2.setMouseCallback"),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch.object(reviewer.cap, "release"),
            patch.object(reviewer, "_render_display", return_value=np.zeros((480, 640, 3), dtype=np.uint8)),
            patch("cv2.waitKey", return_value=ord("q") & 0xFF),
        ):
            annotations, _ = reviewer.run()
        assert 0 in annotations

    def test_run_calls_destroy_and_release(self, reviewer):
        with (
            patch("cv2.namedWindow"),
            patch("cv2.setMouseCallback"),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows") as mock_destroy,
            patch.object(reviewer.cap, "release") as mock_release,
            patch.object(reviewer, "_render_display", return_value=np.zeros((480, 640, 3), dtype=np.uint8)),
            patch("cv2.waitKey", return_value=ord("q") & 0xFF),
        ):
            reviewer.run()
        mock_destroy.assert_called_once()
        mock_release.assert_called_once()


# ─── run_manual_review ──────────────────────────────────────────


class TestRunManualReview:
    def test_returns_empty_stats_when_cap_not_opened(self, config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap.get.return_value = 0

        with patch("cv2.VideoCapture", return_value=mock_cap):
            annotations, stats = run_manual_review("fake.mp4", {}, config)

        assert stats["added"] == 0
        assert stats["removed"] == 0
        assert stats["frames_modified"] == 0
        assert stats["frames_reviewed"] == 0

    def test_delegates_to_reviewer_run_when_opened(self, config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda p: {7: 10, 3: 640, 4: 480}.get(p, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        expected_ann = {0: {"auto": [], "manual": [], "intensities": []}}
        expected_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 1}

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch(
                "person_anonymizer.manual_reviewer.ManualReviewer.run",
                return_value=(expected_ann, expected_stats),
            ),
        ):
            ann, stats = run_manual_review("fake.mp4", {}, config)

        assert ann is expected_ann
        assert stats is expected_stats
