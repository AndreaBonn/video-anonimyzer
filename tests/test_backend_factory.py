"""Test per backend_factory — usa mock per YOLO e SAM3."""

from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.backend_factory import DetectionBackend, load_detection_backend
from person_anonymizer.config import PipelineConfig


class TestDetectionBackend:
    def test_default_values(self):
        backend = DetectionBackend(yolo_model=MagicMock())
        assert backend.sam3_refiner is None
        assert backend.sam3_video_detector is None
        assert backend.backend_name == "yolo"


class TestLoadDetectionBackend:
    @patch("ultralytics.YOLO")
    def test_yolo_backend_loads_only_yolo(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        config = PipelineConfig(detection_backend="yolo")
        backend = load_detection_backend(config)
        assert backend.backend_name == "yolo"
        assert backend.yolo_model is not None
        assert backend.sam3_refiner is None
        assert backend.sam3_video_detector is None
        mock_yolo_cls.assert_called_once_with(config.yolo_model)

    @patch("ultralytics.YOLO")
    def test_yolo_sam3_raises_without_sam3(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        config = PipelineConfig(detection_backend="yolo+sam3")
        with pytest.raises(ImportError, match="SAM3"):
            load_detection_backend(config)

    @patch("ultralytics.YOLO")
    def test_sam3_raises_without_sam3(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        config = PipelineConfig(detection_backend="sam3")
        with pytest.raises(ImportError, match="SAM3"):
            load_detection_backend(config)

    @patch("ultralytics.YOLO")
    def test_yolo_sam3_creates_refiner(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        config = PipelineConfig(detection_backend="yolo+sam3")

        mock_check = MagicMock(return_value=True)
        mock_refiner = MagicMock()
        mock_refiner_cls = MagicMock(return_value=mock_refiner)

        with (
            patch(
                "person_anonymizer.sam3_backend.check_sam3_available", mock_check
            ),
            patch(
                "person_anonymizer.sam3_backend.Sam3ImageRefiner", mock_refiner_cls
            ),
            patch(
                "person_anonymizer.sam3_backend.Sam3VideoDetector", MagicMock()
            ),
        ):
            backend = load_detection_backend(config)
            assert backend.backend_name == "yolo+sam3"
            assert backend.sam3_refiner is mock_refiner
            assert backend.sam3_video_detector is None

    @patch("ultralytics.YOLO")
    def test_sam3_creates_video_detector(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        config = PipelineConfig(detection_backend="sam3")

        mock_check = MagicMock(return_value=True)
        mock_detector = MagicMock()
        mock_detector_cls = MagicMock(return_value=mock_detector)

        with (
            patch(
                "person_anonymizer.sam3_backend.check_sam3_available", mock_check
            ),
            patch(
                "person_anonymizer.sam3_backend.Sam3ImageRefiner", MagicMock()
            ),
            patch(
                "person_anonymizer.sam3_backend.Sam3VideoDetector", mock_detector_cls
            ),
        ):
            backend = load_detection_backend(config)
            assert backend.backend_name == "sam3"
            assert backend.sam3_video_detector is mock_detector
            assert backend.sam3_refiner is None
