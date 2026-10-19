"""
Test aggiuntivi per tracking.py — coverage righe mancanti.

Copre: create_tracker (ByteTracker reale),
TemporalSmoother.clear_stale
(branch linea 165: tid già in ghost_countdown non reinserito).
"""

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.tracking import TemporalSmoother, create_tracker

# ============================================================
# create_tracker
# ============================================================


class TestCreateTracker:
    """Verifica la configurazione del ByteTracker reale."""

    def test_create_tracker_buffer_is_track_max_age_at_low_fps(self):
        # fps=5 -> 2 seconds = 10 frames, below track_max_age=30
        tracker = create_tracker(fps=5.0, config=PipelineConfig(track_max_age=30))

        assert tracker.byte_tracker.args.track_buffer == 30

    def test_create_tracker_buffer_is_two_seconds_at_high_fps(self):
        tracker = create_tracker(fps=25.0, config=PipelineConfig(track_max_age=30))

        assert tracker.byte_tracker.args.track_buffer == 50

    def test_create_tracker_passes_match_threshold(self):
        tracker = create_tracker(fps=25.0, config=PipelineConfig(track_match_thresh=0.65))

        assert tracker.byte_tracker.args.match_thresh == 0.65


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
