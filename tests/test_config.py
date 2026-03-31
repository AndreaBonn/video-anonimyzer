"""
Test per config.py — PipelineConfig e costanti di modulo.

Verifica valori di default, parametri custom e costanti pubbliche
senza alcuna dipendenza da cv2, ultralytics o modelli pesanti.
"""

import pytest

from config import PipelineConfig, SUPPORTED_EXTENSIONS, VERSION


class TestPipelineConfigDefaults:
    """Verifica i valori di default di PipelineConfig."""

    def test_default_operation_mode(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.operation_mode == "manual"

    def test_default_detection_confidence(self):
        # Arrange / Act
        config = PipelineConfig()

        # Assert
        assert config.detection_confidence == pytest.approx(0.20)

    def test_default_anonymization_method(self):
        config = PipelineConfig()
        assert config.anonymization_method == "pixelation"

    def test_default_anonymization_intensity(self):
        config = PipelineConfig()
        assert config.anonymization_intensity == 10

    def test_default_person_padding(self):
        config = PipelineConfig()
        assert config.person_padding == 15

    def test_default_enable_sliding_window(self):
        config = PipelineConfig()
        assert config.enable_sliding_window is True

    def test_default_enable_tracking(self):
        config = PipelineConfig()
        assert config.enable_tracking is True

    def test_default_enable_adaptive_intensity(self):
        config = PipelineConfig()
        assert config.enable_adaptive_intensity is True

    def test_default_adaptive_reference_height(self):
        config = PipelineConfig()
        assert config.adaptive_reference_height == 80

    def test_default_nms_iou_internal(self):
        config = PipelineConfig()
        assert config.nms_iou_internal == pytest.approx(0.45)

    def test_default_nms_iou_threshold(self):
        config = PipelineConfig()
        assert config.nms_iou_threshold == pytest.approx(0.55)

    def test_default_inference_scales(self):
        config = PipelineConfig()
        assert config.inference_scales == [1.0, 1.5, 2.0, 2.5]

    def test_default_tta_augmentations(self):
        config = PipelineConfig()
        assert config.tta_augmentations == ["flip_h"]

    def test_default_yolo_model(self):
        config = PipelineConfig()
        assert config.yolo_model == "yolov8x.pt"

    def test_default_sliding_window_grid(self):
        config = PipelineConfig()
        assert config.sliding_window_grid == 3

    def test_default_enable_post_render_check(self):
        config = PipelineConfig()
        assert config.enable_post_render_check is True


class TestPipelineConfigCustomValues:
    """Verifica che i parametri custom sovrascrivano i default."""

    def test_custom_operation_mode(self):
        # Arrange / Act
        config = PipelineConfig(operation_mode="automatic")

        # Assert
        assert config.operation_mode == "automatic"

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
            operation_mode="automatic",
            detection_confidence=0.35,
            anonymization_intensity=20,
            enable_tracking=False,
        )

        # Assert
        assert config.operation_mode == "automatic"
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
