"""
Test supplementari per person_anonymizer/config.py e models.py.

Copre le righe di validazione non ancora testate in PipelineConfig.__post_init__:
- person_padding boundary (linea 122: < 0, linea 126: > 200)
- ghost_expansion boundary (linea 164: < 1.0, linea 168: > 2.0)
- max_refinement_passes boundary (linea 173: > 10)
- sliding_window_grid boundary (linea 179: > 10)
- nms_iou_internal e nms_iou_threshold boundary (0 e 1 esclusivi)
- adaptive_reference_height (< 1)

Copre models.py line 113, 115: FisheyeContext.undistort con enabled=True.
"""

import pytest

from person_anonymizer.config import PipelineConfig


# ---------------------------------------------------------------------------
# TestPipelineConfigPersonPadding
# ---------------------------------------------------------------------------


class TestPipelineConfigPersonPadding:
    """Verifica validazione person_padding (0-200)."""

    def test_person_padding_negative_raises(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="person_padding"):
            PipelineConfig(person_padding=-1)

    def test_person_padding_over_200_raises(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="person_padding"):
            PipelineConfig(person_padding=201)

    def test_person_padding_zero_valid(self):
        # Boundary valido inferiore
        config = PipelineConfig(person_padding=0)
        assert config.person_padding == 0

    def test_person_padding_200_valid(self):
        # Boundary valido superiore
        config = PipelineConfig(person_padding=200)
        assert config.person_padding == 200


# ---------------------------------------------------------------------------
# TestPipelineConfigNmsIou
# ---------------------------------------------------------------------------


class TestPipelineConfigNmsIou:
    """Verifica validazione nms_iou_internal e nms_iou_threshold (0 < x < 1)."""

    def test_nms_iou_internal_zero_raises(self):
        with pytest.raises(ValueError, match="nms_iou_internal"):
            PipelineConfig(nms_iou_internal=0.0)

    def test_nms_iou_internal_one_raises(self):
        with pytest.raises(ValueError, match="nms_iou_internal"):
            PipelineConfig(nms_iou_internal=1.0)

    def test_nms_iou_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="nms_iou_threshold"):
            PipelineConfig(nms_iou_threshold=0.0)

    def test_nms_iou_threshold_one_raises(self):
        with pytest.raises(ValueError, match="nms_iou_threshold"):
            PipelineConfig(nms_iou_threshold=1.0)

    def test_nms_iou_internal_mid_valid(self):
        config = PipelineConfig(nms_iou_internal=0.5)
        assert config.nms_iou_internal == pytest.approx(0.5)

    def test_nms_iou_threshold_mid_valid(self):
        config = PipelineConfig(nms_iou_threshold=0.5)
        assert config.nms_iou_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestPipelineConfigGhostExpansion
# ---------------------------------------------------------------------------


class TestPipelineConfigGhostExpansion:
    """Verifica validazione ghost_expansion (1.0-2.0)."""

    def test_ghost_expansion_below_one_raises(self):
        with pytest.raises(ValueError, match="ghost_expansion"):
            PipelineConfig(ghost_expansion=0.99)

    def test_ghost_expansion_above_two_raises(self):
        with pytest.raises(ValueError, match="ghost_expansion"):
            PipelineConfig(ghost_expansion=2.01)

    def test_ghost_expansion_one_valid(self):
        config = PipelineConfig(ghost_expansion=1.0)
        assert config.ghost_expansion == pytest.approx(1.0)

    def test_ghost_expansion_two_valid(self):
        config = PipelineConfig(ghost_expansion=2.0)
        assert config.ghost_expansion == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestPipelineConfigMaxRefinementPasses
# ---------------------------------------------------------------------------


class TestPipelineConfigMaxRefinementPasses:
    """Verifica validazione max_refinement_passes (1-10)."""

    def test_max_refinement_passes_zero_raises(self):
        with pytest.raises(ValueError, match="max_refinement_passes"):
            PipelineConfig(max_refinement_passes=0)

    def test_max_refinement_passes_eleven_raises(self):
        with pytest.raises(ValueError, match="max_refinement_passes"):
            PipelineConfig(max_refinement_passes=11)

    def test_max_refinement_passes_one_valid(self):
        config = PipelineConfig(max_refinement_passes=1)
        assert config.max_refinement_passes == 1

    def test_max_refinement_passes_ten_valid(self):
        config = PipelineConfig(max_refinement_passes=10)
        assert config.max_refinement_passes == 10


# ---------------------------------------------------------------------------
# TestPipelineConfigSlidingWindowGrid
# ---------------------------------------------------------------------------


class TestPipelineConfigSlidingWindowGrid:
    """Verifica validazione sliding_window_grid (1-10)."""

    def test_sliding_window_grid_zero_raises(self):
        with pytest.raises(ValueError, match="sliding_window_grid"):
            PipelineConfig(sliding_window_grid=0)

    def test_sliding_window_grid_eleven_raises(self):
        with pytest.raises(ValueError, match="sliding_window_grid"):
            PipelineConfig(sliding_window_grid=11)

    def test_sliding_window_grid_one_valid(self):
        config = PipelineConfig(sliding_window_grid=1)
        assert config.sliding_window_grid == 1

    def test_sliding_window_grid_ten_valid(self):
        config = PipelineConfig(sliding_window_grid=10)
        assert config.sliding_window_grid == 10


# ---------------------------------------------------------------------------
# TestPipelineConfigAdaptiveReferenceHeight
# ---------------------------------------------------------------------------


class TestPipelineConfigAdaptiveReferenceHeight:
    """Verifica validazione adaptive_reference_height (>= 1)."""

    def test_adaptive_reference_height_zero_raises(self):
        with pytest.raises(ValueError, match="adaptive_reference_height"):
            PipelineConfig(adaptive_reference_height=0)

    def test_adaptive_reference_height_negative_raises(self):
        with pytest.raises(ValueError, match="adaptive_reference_height"):
            PipelineConfig(adaptive_reference_height=-5)

    def test_adaptive_reference_height_one_valid(self):
        config = PipelineConfig(adaptive_reference_height=1)
        assert config.adaptive_reference_height == 1

    def test_adaptive_reference_height_large_valid(self):
        config = PipelineConfig(adaptive_reference_height=1000)
        assert config.adaptive_reference_height == 1000


# ---------------------------------------------------------------------------
# TestPipelineConfigGhostFrames
# ---------------------------------------------------------------------------


class TestPipelineConfigGhostFrames:
    """Verifica validazione ghost_frames (0-120)."""

    def test_ghost_frames_over_120_raises(self):
        with pytest.raises(ValueError, match="ghost_frames"):
            PipelineConfig(ghost_frames=121)

    def test_ghost_frames_120_valid(self):
        config = PipelineConfig(ghost_frames=120)
        assert config.ghost_frames == 120

    def test_ghost_frames_zero_valid(self):
        config = PipelineConfig(ghost_frames=0)
        assert config.ghost_frames == 0


# ---------------------------------------------------------------------------
# TestFisheyeContextUndistort (models.py lines 113, 115)
# ---------------------------------------------------------------------------


class TestFisheyeContextUndistort:
    """Verifica FisheyeContext.undistort — path enabled=True con mappe valide."""

    def test_undistort_returns_frame_when_disabled(self):
        # Arrange
        import numpy as np

        from person_anonymizer.models import FisheyeContext

        ctx = FisheyeContext(enabled=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        result = ctx.undistort(frame)

        # Assert — restituisce il frame originale senza modifiche
        assert result is frame

    def test_undistort_returns_frame_when_enabled_but_no_maps(self):
        # Arrange
        import numpy as np

        from person_anonymizer.models import FisheyeContext

        ctx = FisheyeContext(enabled=True, undist_map1=None, undist_map2=None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act — enabled=True ma map1 è None → deve restituire il frame originale
        result = ctx.undistort(frame)

        # Assert
        assert result is frame

    def test_undistort_applies_remap_when_enabled_with_maps(self):
        # Arrange — line 113 (if branch) e 115 (cv2.remap return)
        import numpy as np
        from unittest.mock import patch, MagicMock

        from person_anonymizer.models import FisheyeContext

        map1 = np.zeros((100, 100), dtype=np.float32)
        map2 = np.zeros((100, 100), dtype=np.float32)
        ctx = FisheyeContext(enabled=True, undist_map1=map1, undist_map2=map2)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        remapped = np.ones((100, 100, 3), dtype=np.uint8)

        # Act
        with patch("cv2.remap", return_value=remapped) as mock_remap:
            result = ctx.undistort(frame)

        # Assert — remap viene chiamato e il risultato è quello restituito da cv2.remap
        mock_remap.assert_called_once()
        assert result is remapped
