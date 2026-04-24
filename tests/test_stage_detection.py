"""
Test per stage_detection.py — _init_frame_processors, _process_single_frame, run_detection_loop.

Mock di tutte le dipendenze esterne: cv2, YOLO, tracking, detection.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext, FrameProcessors


def _make_config(**kwargs):
    defaults = dict(
        enable_subframe_interpolation=False,
        enable_motion_detection=False,
        enable_sliding_window=False,
        enable_tracking=True,
        enable_temporal_smoothing=False,
        interpolation_fps_threshold=30,
        quality_clahe_clip=2.0,
        quality_clahe_grid=(8, 8),
        motion_threshold=25,
        motion_min_area=500,
        motion_padding=60,
        sliding_window_grid=2,
        sliding_window_overlap=0.25,
        smoothing_alpha=0.7,
        ghost_frames=3,
        ghost_expansion=1.5,
        detection_confidence=0.2,
        nms_iou_threshold=0.55,
        person_padding=15,
        enable_adaptive_intensity=False,
        anonymization_intensity=10,
        adaptive_reference_height=180,
        enable_fisheye_correction=False,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_cap(frames=None, w=640, h=480, fps=25.0):
    """Crea un mock VideoCapture con lista frame."""
    import cv2
    cap = MagicMock()
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: w,
        cv2.CAP_PROP_FRAME_HEIGHT: h,
        cv2.CAP_PROP_FPS: fps,
    }.get(prop, 0)

    if frames is None:
        frames = [_make_frame(h, w)]

    read_returns = [(True, f) for f in frames] + [(False, None)]
    cap.read.side_effect = read_returns
    cap.release = MagicMock()
    return cap


class TestInitFrameProcessors:
    """Test _init_frame_processors."""

    def test_returns_frame_processors_with_clahe(self):
        # Arrange
        config = _make_config()

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker") as mock_tracker:
            mock_clahe.return_value = MagicMock()
            mock_tracker.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            # Act
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        # Assert
        assert proc.clahe_obj is not None
        mock_clahe.assert_called_once()

    def test_motion_detector_created_when_enabled(self):
        # Arrange
        config = _make_config(enable_motion_detection=True)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.MotionDetector") as mock_md, \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()
            mock_md.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        mock_md.assert_called_once()
        assert proc.motion_detector is not None

    def test_motion_detector_none_when_disabled(self):
        # Arrange
        config = _make_config(enable_motion_detection=False)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        assert proc.motion_detector is None

    def test_patches_populated_when_sliding_window_enabled(self):
        # Arrange
        config = _make_config(enable_sliding_window=True)
        fake_patches = [(0, 0, 320, 240), (320, 0, 640, 240)]

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.get_window_patches",
                   return_value=fake_patches), \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        assert proc.patches == fake_patches

    def test_patches_empty_when_sliding_window_disabled(self):
        # Arrange
        config = _make_config(enable_sliding_window=False)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        assert proc.patches == []

    def test_tracker_none_when_tracking_disabled(self):
        # Arrange
        config = _make_config(enable_tracking=False)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe:
            mock_clahe.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        assert proc.tracker is None

    def test_smoother_created_when_smoothing_enabled(self):
        # Arrange
        config = _make_config(enable_temporal_smoothing=True)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.TemporalSmoother") as mock_sm, \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()
            mock_sm.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        mock_sm.assert_called_once()
        assert proc.smoother is not None

    def test_do_interpolation_false_when_fps_above_threshold(self):
        # Arrange — fps 25 < threshold 30 → interpolation enabled se enable_subframe_interpolation
        config = _make_config(
            enable_subframe_interpolation=False,
            interpolation_fps_threshold=30,
        )

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker"):
            mock_clahe.return_value = MagicMock()

            from person_anonymizer.stage_detection import _init_frame_processors
            proc = _init_frame_processors(fps=25.0, frame_w=640, frame_h=480, config=config)

        assert proc.do_interpolation is False


class TestProcessSingleFrame:
    """Test _process_single_frame."""

    def _make_proc(self, config=None):
        if config is None:
            config = _make_config()
        proc = FrameProcessors()
        proc.clahe_obj = MagicMock()
        proc.motion_detector = None
        proc.patches = []
        proc.tracker = None
        proc.smoother = None
        proc.do_interpolation = False
        proc.sam3_refiner = None
        return proc

    def test_returns_frame_detection_result(self):
        # Arrange
        config = _make_config(enable_tracking=False)
        frame = _make_frame()
        proc = self._make_proc(config)

        with patch("person_anonymizer.stage_detection.enhance_frame",
                   return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]):

            from person_anonymizer.stage_detection import _process_single_frame
            result = _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        # Assert — risultato con 0 poligoni (nessuna detection)
        assert result.polygons == []
        assert result.intensities == []
        assert result.tracked == []

    def test_no_detections_returns_empty_polygons(self):
        # Arrange
        config = _make_config(enable_tracking=False)
        frame = _make_frame()
        proc = self._make_proc(config)

        with patch("person_anonymizer.stage_detection.enhance_frame", return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]):

            from person_anonymizer.stage_detection import _process_single_frame
            result = _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        assert result.polygons == []
        assert result.intensities == []

    def test_detections_produce_polygons_without_tracker(self):
        # Arrange
        config = _make_config(enable_tracking=False, enable_temporal_smoothing=False)
        frame = _make_frame()
        proc = self._make_proc(config)
        fake_box = [100, 100, 200, 300, 0.9]  # x1,y1,x2,y2,conf

        with patch("person_anonymizer.stage_detection.enhance_frame", return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([fake_box], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[fake_box]), \
             patch("person_anonymizer.stage_detection.box_to_polygon",
                   return_value=[(100, 100), (200, 100), (200, 300), (100, 300)]):

            from person_anonymizer.stage_detection import _process_single_frame
            result = _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        assert len(result.polygons) == 1

    def test_sam3_refiner_used_when_provided(self):
        # Arrange
        config = _make_config(enable_tracking=False, enable_temporal_smoothing=False)
        frame = _make_frame()
        proc = self._make_proc(config)
        proc.sam3_refiner = MagicMock()
        proc.sam3_refiner.refine_boxes.return_value = [[(10, 10), (20, 10), (20, 20), (10, 20)]]
        fake_box = [100, 100, 200, 300, 0.9]

        with patch("person_anonymizer.stage_detection.enhance_frame", return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([fake_box], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[fake_box]):

            from person_anonymizer.stage_detection import _process_single_frame
            result = _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        proc.sam3_refiner.refine_boxes.assert_called_once()
        assert len(result.polygons) == 1

    def test_smoother_clears_stale_ids(self):
        # Arrange
        config = _make_config(enable_tracking=False, enable_temporal_smoothing=True)
        frame = _make_frame()
        proc = self._make_proc(config)
        smoother = MagicMock()
        smoother.smooth.return_value = (100, 100, 200, 300)
        smoother.get_ghost_boxes.return_value = []
        proc.smoother = smoother

        with patch("person_anonymizer.stage_detection.enhance_frame", return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]), \
             patch("person_anonymizer.stage_detection.box_to_polygon",
                   return_value=[(0, 0), (10, 0), (10, 10), (0, 10)]):

            from person_anonymizer.stage_detection import _process_single_frame
            _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        smoother.clear_stale.assert_called_once()

    def test_ghost_boxes_added_to_polygons(self):
        # Arrange
        config = _make_config(enable_tracking=False, enable_temporal_smoothing=True)
        frame = _make_frame()
        proc = self._make_proc(config)
        smoother = MagicMock()
        smoother.smooth.return_value = (100, 100, 200, 300)
        smoother.get_ghost_boxes.return_value = [(99, 10, 10, 200, 300)]  # gtid, gx1, gy1, gx2, gy2
        proc.smoother = smoother

        with patch("person_anonymizer.stage_detection.enhance_frame", return_value=frame), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]), \
             patch("person_anonymizer.stage_detection.box_to_polygon",
                   return_value=[(0, 0), (10, 0), (10, 10), (0, 10)]):

            from person_anonymizer.stage_detection import _process_single_frame
            result = _process_single_frame(frame, MagicMock(), config, 640, 480, proc, None, 25.0)

        # 1 ghost box → 1 polygon
        assert len(result.polygons) == 1


class TestRunDetectionLoop:
    """Test run_detection_loop — loop frame-per-frame."""

    def _run_loop(self, frames, config=None, stop_event=None, sam3_refiner=None):
        if config is None:
            config = _make_config(enable_tracking=False)
        cap = _make_cap(frames=frames)
        fisheye = FisheyeContext(enabled=False)
        total_frames = len(frames)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker", return_value=MagicMock()), \
             patch("person_anonymizer.stage_detection.tqdm") as mock_tqdm, \
             patch("person_anonymizer.stage_detection.enhance_frame",
                   side_effect=lambda f, *a, **kw: f), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]):

            mock_clahe.return_value = MagicMock()
            mock_tqdm.return_value.__enter__ = MagicMock()
            mock_tqdm.return_value.__exit__ = MagicMock()
            # tqdm usato come context o come oggetto con update/close
            mock_tqdm_obj = MagicMock()
            mock_tqdm_obj.update = MagicMock()
            mock_tqdm_obj.close = MagicMock()
            mock_tqdm.return_value = mock_tqdm_obj

            from person_anonymizer.stage_detection import run_detection_loop
            return run_detection_loop(
                cap, total_frames, MagicMock(), config, fisheye,
                stop_event=stop_event, sam3_refiner=sam3_refiner
            )

    def test_returns_annotations_report_stats(self):
        # Arrange
        frames = [_make_frame() for _ in range(3)]

        # Act
        annotations, report_data, stats = self._run_loop(frames)

        # Assert — 3 frame processati con struttura corretta
        assert len(annotations) == 3
        assert set(annotations.keys()) == {0, 1, 2}
        assert len(report_data) == 3
        assert "unique_ids" in stats and "total_instances" in stats

    def test_annotations_keyed_by_frame_index(self):
        # Arrange
        frames = [_make_frame() for _ in range(3)]

        # Act
        annotations, _, _ = self._run_loop(frames)

        # Assert
        assert set(annotations.keys()) == {0, 1, 2}

    def test_each_annotation_has_required_keys(self):
        # Arrange
        frames = [_make_frame() for _ in range(2)]

        # Act
        annotations, _, _ = self._run_loop(frames)

        # Assert
        for fidx, ann in annotations.items():
            assert "auto" in ann
            assert "manual" in ann
            assert "intensities" in ann

    def test_report_data_contains_frame_stats(self):
        # Arrange
        frames = [_make_frame() for _ in range(2)]

        # Act
        _, report_data, _ = self._run_loop(frames)

        # Assert
        for fidx, data in report_data.items():
            assert "frame_number" in data
            assert "persons_detected" in data
            assert "avg_confidence" in data

    def test_stop_event_interrupts_loop(self):
        # Arrange — stop_event già settato
        stop = threading.Event()
        stop.set()
        frames = [_make_frame() for _ in range(10)]

        # Act
        annotations, _, _ = self._run_loop(frames, stop_event=stop)

        # Assert — 0 frame processati perché stop è già settato
        assert len(annotations) == 0

    def test_empty_detections_counted_in_zero_det(self):
        # Arrange — tutti i frame hanno 0 detection
        frames = [_make_frame() for _ in range(5)]

        # Act
        annotations, report_data, stats = self._run_loop(frames)

        # Assert
        for data in report_data.values():
            assert data["persons_detected"] == 0

    def test_corrupted_frame_skipped(self):
        # Arrange — secondo frame è corrotto (ret=False)
        config = _make_config(enable_tracking=False)
        cap = MagicMock()
        import cv2
        cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 25.0,
        }.get(prop, 0)

        good_frame = _make_frame()
        # 3 frame totali: buono, corrotto, buono (ultimo triggera break)
        cap.read.side_effect = [
            (True, good_frame),
            (False, None),   # corrotto ma non ultimo
            (True, good_frame),
            (False, None),   # fine stream
        ]
        cap.release = MagicMock()
        fisheye = FisheyeContext(enabled=False)

        with patch("person_anonymizer.stage_detection.cv2.createCLAHE") as mock_clahe, \
             patch("person_anonymizer.stage_detection.create_tracker", return_value=MagicMock()), \
             patch("person_anonymizer.stage_detection.tqdm") as mock_tqdm, \
             patch("person_anonymizer.stage_detection.enhance_frame",
                   side_effect=lambda f, *a, **kw: f), \
             patch("person_anonymizer.stage_detection.run_full_detection",
                   return_value=([], 0, 0)), \
             patch("person_anonymizer.stage_detection.apply_nms", return_value=[]):

            mock_clahe.return_value = MagicMock()
            mock_tqdm_obj = MagicMock()
            mock_tqdm_obj.update = MagicMock()
            mock_tqdm_obj.close = MagicMock()
            mock_tqdm.return_value = mock_tqdm_obj

            from person_anonymizer.stage_detection import run_detection_loop
            annotations, _, _ = run_detection_loop(
                cap, 3, MagicMock(), config, fisheye
            )

        # Frame 0 e 2 processati, frame 1 saltato
        assert 0 in annotations
        # Frame corrotto (idx 1) non in annotations
        assert 1 not in annotations

    def test_stats_contain_unique_ids_and_total_instances(self):
        # Arrange
        frames = [_make_frame() for _ in range(2)]

        # Act
        _, _, stats = self._run_loop(frames)

        # Assert — con 0 detection, contatori a zero
        assert stats["unique_ids"] == set()
        assert stats["total_instances"] == 0
