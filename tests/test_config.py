"""
Test per config.py — PipelineConfig e costanti di modulo.

Verifica valori di default, parametri custom e costanti pubbliche
senza alcuna dipendenza da cv2, ultralytics o modelli pesanti.
"""

import pytest

from person_anonymizer.config import SUPPORTED_EXTENSIONS, VERSION, PipelineConfig


class TestPipelineConfigInvariants:
    """Verifica invarianti e contratti di PipelineConfig."""

    def test_inference_scales_not_empty(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert len(config.inference_scales) > 0

    def test_inference_scales_all_positive(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert all(s > 0 for s in config.inference_scales)

    def test_anonymization_intensity_positive(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.anonymization_intensity > 0

    def test_detection_confidence_in_unit_range(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert 0.0 < config.detection_confidence < 1.0

    def test_nms_thresholds_in_unit_range(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert 0.0 < config.nms_iou_threshold < 1.0
        assert 0.0 < config.nms_iou_internal < 1.0

    def test_quality_clahe_grid_is_pair_of_positive_ints(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert isinstance(config.quality_clahe_grid, tuple)
        assert len(config.quality_clahe_grid) == 2
        assert all(isinstance(x, int) and x > 0 for x in config.quality_clahe_grid)

    def test_adaptive_reference_height_positive(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.adaptive_reference_height > 0

    def test_ghost_expansion_at_least_one(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.ghost_expansion >= 1.0

    def test_smoothing_alpha_in_valid_range(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert 0.0 < config.smoothing_alpha <= 1.0

    def test_max_refinement_passes_positive(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.max_refinement_passes >= 1

    def test_operation_mode_is_valid_value(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert — i valori ammessi sono "manual" e "auto"
        assert config.operation_mode in ("manual", "auto")

    def test_anonymization_method_is_valid_value(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.anonymization_method in ("pixelation", "blur")


class TestPipelineConfigCustomValues:
    """Verifica che i parametri custom sovrascrivano i default."""

    def test_custom_operation_mode(self):
        # Arrange / Act
        config = PipelineConfig(operation_mode="auto")

        # Assert
        assert config.operation_mode == "auto"

    def test_custom_detection_confidence(self):
        # Arrange / Act
        config = PipelineConfig(detection_confidence=0.50)

        # Assert
        assert config.detection_confidence == pytest.approx(0.50)

    def test_custom_anonymization_method(self):
        config = PipelineConfig(anonymization_method="blur")
        assert config.anonymization_method == "blur"

    def test_custom_person_padding(self):
        config = PipelineConfig(person_padding=30)
        assert config.person_padding == 30

    def test_custom_inference_scales(self):
        # Arrange
        scales = [1.0, 2.0]

        # Act
        config = PipelineConfig(inference_scales=scales)

        # Assert
        assert config.inference_scales == [1.0, 2.0]

    def test_custom_enable_motion_detection(self):
        config = PipelineConfig(enable_motion_detection=True)
        assert config.enable_motion_detection is True

    def test_custom_multiple_params(self):
        # Arrange / Act
        config = PipelineConfig(
            operation_mode="auto",
            detection_confidence=0.35,
            anonymization_intensity=20,
            enable_tracking=False,
        )

        # Assert
        assert config.operation_mode == "auto"
        assert config.detection_confidence == pytest.approx(0.35)
        assert config.anonymization_intensity == 20
        assert config.enable_tracking is False


class TestSupportedExtensions:
    """Verifica le estensioni video supportate."""

    def test_supported_extensions_is_set(self):
        assert isinstance(SUPPORTED_EXTENSIONS, set)

    def test_mp4_supported(self):
        assert ".mp4" in SUPPORTED_EXTENSIONS

    def test_mov_supported(self):
        assert ".mov" in SUPPORTED_EXTENSIONS

    def test_avi_supported(self):
        assert ".avi" in SUPPORTED_EXTENSIONS

    def test_mkv_supported(self):
        assert ".mkv" in SUPPORTED_EXTENSIONS

    def test_webm_supported(self):
        assert ".webm" in SUPPORTED_EXTENSIONS


class TestVersion:
    """Verifica la costante VERSION."""

    def test_version_is_string(self):
        assert isinstance(VERSION, str)

    def test_version_is_not_empty(self):
        assert len(VERSION) > 0


class TestPipelineConfigValidation:
    """Test per la validazione __post_init__ di PipelineConfig."""

    def test_invalid_detection_confidence_raises(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            PipelineConfig(detection_confidence=1.5)

    def test_invalid_detection_confidence_negative(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            PipelineConfig(detection_confidence=-0.1)

    def test_invalid_anonymization_intensity_zero(self):
        with pytest.raises(ValueError, match="anonymization_intensity"):
            PipelineConfig(anonymization_intensity=0)

    def test_invalid_anonymization_intensity_over_100(self):
        with pytest.raises(ValueError, match="anonymization_intensity"):
            PipelineConfig(anonymization_intensity=101)

    def test_invalid_operation_mode(self):
        with pytest.raises(ValueError, match="operation_mode"):
            PipelineConfig(operation_mode="automatic")

    def test_invalid_anonymization_method(self):
        with pytest.raises(ValueError, match="anonymization_method"):
            PipelineConfig(anonymization_method="mosaic")

    def test_invalid_smoothing_alpha_zero(self):
        with pytest.raises(ValueError, match="smoothing_alpha"):
            PipelineConfig(smoothing_alpha=0.0)

    def test_invalid_ghost_frames_negative(self):
        with pytest.raises(ValueError, match="ghost_frames"):
            PipelineConfig(ghost_frames=-1)

    def test_empty_inference_scales(self):
        with pytest.raises(ValueError, match="inference_scales"):
            PipelineConfig(inference_scales=[])

    def test_valid_boundary_values_accepted(self):
        # Arrange / Act — non deve lanciare eccezioni
        config = PipelineConfig(
            detection_confidence=0.01,
            anonymization_intensity=1,
            smoothing_alpha=1.0,
            ghost_frames=0,
            max_refinement_passes=1,
        )
        # Assert
        assert config.detection_confidence == 0.01


class TestPipelineConfigSam3Fields:
    """Test per i campi SAM3 aggiunti a PipelineConfig."""

    def test_default_detection_backend_is_yolo(self):
        config = PipelineConfig()
        assert config.detection_backend == "yolo"

    def test_default_sam3_model(self):
        config = PipelineConfig()
        assert config.sam3_model == "sam3_hiera_large.pt"

    def test_default_sam3_text_prompt(self):
        config = PipelineConfig()
        assert config.sam3_text_prompt == "person"

    def test_default_sam3_mask_simplify_epsilon(self):
        config = PipelineConfig()
        assert 0.0 < config.sam3_mask_simplify_epsilon < 1.0

    def test_default_sam3_min_mask_area(self):
        config = PipelineConfig()
        assert config.sam3_min_mask_area >= 1

    def test_custom_detection_backend_yolo_sam3(self):
        config = PipelineConfig(detection_backend="yolo+sam3")
        assert config.detection_backend == "yolo+sam3"

    def test_custom_detection_backend_sam3(self):
        config = PipelineConfig(detection_backend="sam3")
        assert config.detection_backend == "sam3"

    def test_invalid_detection_backend_raises(self):
        with pytest.raises(ValueError, match="detection_backend"):
            PipelineConfig(detection_backend="invalid")

    def test_invalid_sam3_min_mask_area_zero(self):
        with pytest.raises(ValueError, match="sam3_min_mask_area"):
            PipelineConfig(sam3_min_mask_area=0)

    def test_invalid_sam3_mask_simplify_epsilon_zero(self):
        with pytest.raises(ValueError, match="sam3_mask_simplify_epsilon"):
            PipelineConfig(sam3_mask_simplify_epsilon=0.0)

    def test_invalid_sam3_mask_simplify_epsilon_one(self):
        with pytest.raises(ValueError, match="sam3_mask_simplify_epsilon"):
            PipelineConfig(sam3_mask_simplify_epsilon=1.0)
