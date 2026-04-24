"""
Test aggiuntivi per tracking.py — coverage righe mancanti.

Copre: create_tracker (con mock ultralytics), update_tracker
(path success e fallback su eccezione), TemporalSmoother.clear_stale
(branch linea 165: tid già in ghost_countdown non reinserito).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.tracking import TemporalSmoother, create_tracker, update_tracker

# ============================================================
# create_tracker — righe 36-55
# ============================================================


class TestCreateTracker:
    """Verifica create_tracker con mock ultralytics."""

    def test_create_tracker_returns_instance(self):
        # Arrange — mock completo di ultralytics
        mock_tracker_instance = MagicMock()
        mock_byte_tracker_cls = MagicMock(return_value=mock_tracker_instance)
        mock_namespace_cls = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))

        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(),
            "ultralytics.trackers": MagicMock(),
            "ultralytics.trackers.byte_tracker": MagicMock(BYTETracker=mock_byte_tracker_cls),
            "ultralytics.utils": MagicMock(IterableSimpleNamespace=mock_namespace_cls),
        }):
            config = PipelineConfig(track_max_age=30, track_match_thresh=0.8)
            tracker = create_tracker(fps=25.0, config=config)

        # Assert — restituisce l'istanza del tracker
        assert tracker is mock_tracker_instance
        assert mock_byte_tracker_cls.called

    def test_create_tracker_buffer_at_least_track_max_age(self):
        # Arrange — fps=5, track_max_age=30, fps*2=10 → buffer = max(30, 10) = 30
        captured_args = {}

        def capture_namespace(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_byte_tracker_cls = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(),
            "ultralytics.trackers": MagicMock(),
            "ultralytics.trackers.byte_tracker": MagicMock(BYTETracker=mock_byte_tracker_cls),
            "ultralytics.utils": MagicMock(IterableSimpleNamespace=capture_namespace),
        }):
            config = PipelineConfig(track_max_age=30, track_match_thresh=0.8)
            create_tracker(fps=5.0, config=config)

        # Assert — track_buffer >= track_max_age
        assert captured_args.get("track_buffer", 0) >= 30

    def test_create_tracker_returns_byte_tracker(self):
        # Verifica che create_tracker restituisca un BYTETracker configurato
        config = PipelineConfig()
        result = create_tracker(fps=25.0, config=config)
        # Il risultato deve avere gli attributi tipici di BYTETracker
        assert hasattr(result, "update")
        assert hasattr(result, "reset")


# ============================================================
# update_tracker — righe 58-97
# ============================================================


class TestUpdateTracker:
    """Verifica update_tracker con tracker mock."""

    def test_empty_boxes_passes_empty_array(self):
        # Arrange — tracker mock che restituisce lista vuota
        tracker = MagicMock()
        tracker.update.return_value = []
        frame_shape = (100, 200, 3)

        # Act
        result = update_tracker(tracker, nms_boxes=[], frame_shape=frame_shape)

        # Assert — update chiamato con array vuoto
        call_args = tracker.update.call_args[0]
        det_array = call_args[0]
        assert det_array.shape == (0, 6)
        assert result == []

    def test_valid_boxes_converted_to_det_array(self):
        # Arrange
        tracker = MagicMock()
        tracker.update.return_value = []
        nms_boxes = [[10.0, 20.0, 50.0, 80.0, 0.9]]
        frame_shape = (100, 200, 3)

        # Act
        update_tracker(tracker, nms_boxes=nms_boxes, frame_shape=frame_shape)

        # Assert — array det con shape (1, 6)
        call_args = tracker.update.call_args[0]
        det_array = call_args[0]
        assert det_array.shape == (1, 6)
        assert det_array[0, 0] == pytest.approx(10.0)
        assert det_array[0, 4] == pytest.approx(0.9)

    def test_tracker_returns_tracks_as_results(self):
        # Arrange — tracker restituisce 1 track
        mock_track = MagicMock()
        mock_track.tlbr = np.array([10.0, 20.0, 50.0, 80.0])
        mock_track.track_id = 42
        mock_track.score = 0.85

        tracker = MagicMock()
        tracker.update.return_value = [mock_track]
        frame_shape = (100, 200, 3)

        # Act
        result = update_tracker(
            tracker, nms_boxes=[[10, 20, 50, 80, 0.85]], frame_shape=frame_shape
        )

        # Assert — result contiene (track_id, x1, y1, x2, y2, conf)
        assert len(result) == 1
        tid, x1, y1, x2, y2, score = result[0]
        assert tid == 42
        assert x1 == pytest.approx(10.0)
        assert score == pytest.approx(0.85)

    def test_tracker_exception_triggers_fallback(self):
        # Arrange — tracker.update lancia eccezione → fallback
        tracker = MagicMock()
        tracker.update.side_effect = RuntimeError("tracker crash")
        nms_boxes = [[5.0, 10.0, 40.0, 70.0, 0.75]]
        frame_shape = (100, 200, 3)

        # Act — deve usare il fallback senza propagare eccezione
        result = update_tracker(tracker, nms_boxes=nms_boxes, frame_shape=frame_shape)

        # Assert — fallback: restituisce box originali con id sequenziali
        assert len(result) == 1
        tid, x1, y1, x2, y2, conf = result[0]
        assert tid == 0
        assert x1 == pytest.approx(5.0)
        assert conf == pytest.approx(0.75)

    def test_tracker_exception_empty_boxes_fallback_empty(self):
        # Arrange — tracker fallisce con box vuoti
        tracker = MagicMock()
        tracker.update.side_effect = ValueError("empty")
        frame_shape = (100, 200, 3)

        # Act
        result = update_tracker(tracker, nms_boxes=[], frame_shape=frame_shape)

        # Assert — fallback con lista vuota → risultato vuoto
        assert result == []

    def test_track_with_exception_in_extraction_skipped(self):
        # Arrange — oggetto track che lancia eccezione su accesso a tlbr
        class BadTrack:
            @property
            def tlbr(self):
                raise AttributeError("corrupted track data")

        tracker = MagicMock()
        tracker.update.return_value = [BadTrack()]
        frame_shape = (100, 200, 3)

        # Act — il track corrotto viene saltato senza propagare l'eccezione
        result = update_tracker(tracker, nms_boxes=[[0, 0, 10, 10, 0.5]], frame_shape=frame_shape)

        # Assert — track saltato, nessun risultato estratto
        assert result == []

    def test_multiple_boxes_creates_correct_det_array(self):
        # Arrange
        tracker = MagicMock()
        tracker.update.return_value = []
        nms_boxes = [
            [0.0, 0.0, 10.0, 10.0, 0.9],
            [50.0, 50.0, 100.0, 100.0, 0.7],
        ]
        frame_shape = (200, 200, 3)

        # Act
        update_tracker(tracker, nms_boxes=nms_boxes, frame_shape=frame_shape)

        # Assert
        call_args = tracker.update.call_args[0]
        det_array = call_args[0]
        assert det_array.shape == (2, 6)


# ============================================================
# TemporalSmoother.clear_stale — riga 165 (branch già in ghost_countdown)
# ============================================================


class TestClearStaleExtended:
    """Verifica che clear_stale non reinserisca track già in ghost_countdown."""

    def test_already_in_ghost_countdown_not_reset(self):
        # Arrange — track in ghost con countdown 3
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=10)
        smoother.smooth(track_id=99, x1=0, y1=0, x2=50, y2=50)
        smoother.clear_stale(active_ids=set())
        # countdown ora = 10
        assert smoother.ghost_countdown[99] == 10

        # Act — seconda chiamata a clear_stale con stesso track inattivo
        smoother.clear_stale(active_ids=set())

        # Assert — countdown NON deve essere reimpostato a ghost_frames=10
        # (rimane inalterato o già decrementato da get_ghost_boxes)
        assert smoother.ghost_countdown[99] == 10  # non resettato

    def test_alpha_invalid_zero_raises(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="alpha deve essere tra"):
            TemporalSmoother(alpha=0.0)

    def test_alpha_invalid_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha deve essere tra"):
            TemporalSmoother(alpha=1.1)

    def test_alpha_one_valid(self):
        # alpha=1.0 è valido (sempre il frame corrente)
        smoother = TemporalSmoother(alpha=1.0)
        result = smoother.smooth(track_id=1, x1=10, y1=10, x2=50, y2=50)
        assert result == (10, 10, 50, 50)
