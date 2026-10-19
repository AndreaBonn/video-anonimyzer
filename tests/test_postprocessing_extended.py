"""
Test aggiuntivi per postprocessing.py — coverage righe mancanti.

Copre: encode_with_audio (successo, fallback video-only, fallback copia),
encode_without_audio (successo, fallback copia), run_post_render_check.
Mock: ffmpeg, cv2.VideoCapture, shutil.copy.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.postprocessing import (
    _detect_frame_at_scales,
    encode_with_audio,
    encode_without_audio,
    run_post_render_check,
)

# ============================================================
# encode_with_audio — righe 43-80
# ============================================================


class TestEncodeWithAudio:
    """Verifica encode_with_audio con vari scenari ffmpeg."""

    def test_success_calls_ffmpeg_run(self):
        # Arrange — ffmpeg non lancia eccezioni
        mock_output = MagicMock()
        mock_output.overwrite_output.return_value.run = MagicMock()

        with patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.input.return_value.audio = MagicMock()
            mock_ffmpeg.output.return_value = mock_output

            # Act
            encode_with_audio("input.avi", "original.mp4", "output.mp4")

        # Assert — ffmpeg.output e run chiamati
        assert mock_ffmpeg.output.called
        assert mock_output.overwrite_output.return_value.run.called

    def test_audio_failure_falls_back_to_video_only(self):
        # Arrange — prima chiamata lancia ffmpeg.Error, seconda riesce
        import ffmpeg as ffmpeg_module

        mock_output_success = MagicMock()
        mock_output_success.overwrite_output.return_value.run = MagicMock()

        call_count = [0]

        def fake_output(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Prima chiamata (con audio): fallisce
                mock_fail = MagicMock()
                mock_fail.overwrite_output.return_value.run.side_effect = ffmpeg_module.Error(
                    "ffmpeg", b"", b"error"
                )
                return mock_fail
            else:
                # Seconda chiamata (senza audio): riesce
                return mock_output_success

        with patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.input.return_value.audio = MagicMock()
            mock_ffmpeg.output.side_effect = fake_output
            mock_ffmpeg.Error = ffmpeg_module.Error

            # Act — non deve lanciare eccezione
            encode_with_audio("input.avi", "original.mp4", "output.mp4")

        # Assert — ffmpeg.output chiamato due volte (audio + video-only)
        assert mock_ffmpeg.output.call_count == 2

    def test_both_ffmpeg_failures_fallback_to_copy(self):
        # Arrange — entrambe le chiamate ffmpeg falliscono → shutil.copy
        import ffmpeg as ffmpeg_module

        with (
            patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg,
            patch("person_anonymizer.postprocessing.shutil.copy") as mock_copy,
        ):
            mock_fail = MagicMock()
            mock_fail.overwrite_output.return_value.run.side_effect = ffmpeg_module.Error(
                "ffmpeg", b"", b"error"
            )
            mock_ffmpeg.output.return_value = mock_fail
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.input.return_value.audio = MagicMock()
            mock_ffmpeg.Error = ffmpeg_module.Error

            # Act
            encode_with_audio("input.avi", "original.mp4", "output.mp4")

        # Assert — shutil.copy chiamato con il file sorgente e destinazione
        mock_copy.assert_called_once_with("input.avi", "output.mp4")


# ============================================================
# encode_without_audio — righe 83-96
# ============================================================


class TestEncodeWithoutAudio:
    """Verifica encode_without_audio."""

    def test_success_calls_ffmpeg_run(self):
        # Arrange
        mock_output = MagicMock()
        mock_output.overwrite_output.return_value.run = MagicMock()

        with patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.output.return_value = mock_output

            # Act
            encode_without_audio("input.avi", "output.mp4")

        # Assert
        assert mock_ffmpeg.output.called
        assert mock_output.overwrite_output.return_value.run.called

    def test_ffmpeg_failure_fallback_to_copy(self):
        # Arrange
        import ffmpeg as ffmpeg_module

        with (
            patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg,
            patch("person_anonymizer.postprocessing.shutil.copy") as mock_copy,
        ):
            mock_fail = MagicMock()
            mock_fail.overwrite_output.return_value.run.side_effect = ffmpeg_module.Error(
                "ffmpeg", b"", b"error"
            )
            mock_ffmpeg.output.return_value = mock_fail
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.Error = ffmpeg_module.Error

            # Act
            encode_without_audio("input.avi", "output.mp4")

        # Assert
        mock_copy.assert_called_once_with("input.avi", "output.mp4")

    def test_correct_codec_parameters(self):
        # Arrange
        mock_output = MagicMock()
        mock_output.overwrite_output.return_value.run = MagicMock()

        with patch("person_anonymizer.postprocessing.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value = MagicMock()
            mock_ffmpeg.output.return_value = mock_output

            # Act
            encode_without_audio("video.avi", "out.mp4")

        # Assert — parametri H.264 corretti
        call_kwargs = mock_ffmpeg.output.call_args[1]
        assert call_kwargs.get("vcodec") == "libx264"
        assert call_kwargs.get("crf") == 18
        assert call_kwargs.get("pix_fmt") == "yuv420p"


# ============================================================
# run_post_render_check — righe 129-172
# ============================================================


class TestRunPostRenderCheck:
    """Verifica run_post_render_check con cv2.VideoCapture mock."""

    def _make_cap_mock(self, frames, frame_w=100, frame_h=80):
        """Restituisce un mock di cv2.VideoCapture che emette frame sintetici."""
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {
            3: frame_w,  # CAP_PROP_FRAME_WIDTH
            4: frame_h,  # CAP_PROP_FRAME_HEIGHT
            7: len(frames),  # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)

        read_returns = [(True, f) for f in frames] + [(False, None)]
        cap.read.side_effect = read_returns
        return cap

    def test_no_detections_returns_empty(self):
        # Arrange — modello non rileva nulla
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap_mock = self._make_cap_mock([frame, frame])
        config = PipelineConfig()

        mock_result = MagicMock()
        mock_result.boxes = []

        model = MagicMock()
        model.return_value = [mock_result]
        report_data = {}

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap_mock),
            patch(
                "person_anonymizer.postprocessing.tqdm",
                side_effect=lambda **kw: MagicMock(
                    __enter__=lambda s: s,
                    __exit__=lambda s, *a: None,
                    update=MagicMock(),
                    close=MagicMock(),
                ),
            ),
        ):
            result = run_post_render_check(
                "video.avi", model, 0.5, report_data, config, check_scales=[1.0]
            )

        # Assert — nessun frame con detection genuina
        assert result == []

    def test_detection_at_scale_1_appended_to_alert(self):
        # Arrange — modello rileva 1 box al frame 0
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap_mock = self._make_cap_mock([frame])

        fake_box = MagicMock()
        fake_box.xyxy = [np.array([10.0, 10.0, 50.0, 50.0])]
        fake_box.conf = [0.9]

        mock_result = MagicMock()
        mock_result.boxes = [fake_box]

        model = MagicMock()
        model.return_value = [mock_result]
        report_data = {}
        config = PipelineConfig()

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap_mock),
            patch("person_anonymizer.postprocessing.cv2.dnn") as mock_dnn,
            patch(
                "person_anonymizer.postprocessing.tqdm",
                side_effect=lambda **kw: MagicMock(update=MagicMock(), close=MagicMock()),
            ),
        ):
            mock_dnn.NMSBoxes.return_value = np.array([[0]])
            result = run_post_render_check(
                "video.avi", model, 0.5, report_data, config, check_scales=[1.0]
            )

        # Assert — 1 frame con alert
        assert len(result) == 1
        frame_idx, count, nms_boxes = result[0]
        assert frame_idx == 0
        assert count >= 1

    def test_default_check_scales_used_when_none(self):
        # Arrange — check_scales=None → usa [1.0, 2.0]
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap_mock = self._make_cap_mock([frame])

        mock_result = MagicMock()
        mock_result.boxes = []

        model = MagicMock()
        model.return_value = [mock_result]
        config = PipelineConfig()

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap_mock),
            patch("person_anonymizer.postprocessing.cv2.resize", return_value=frame),
            patch("person_anonymizer.postprocessing.cv2.INTER_CUBIC", 2),
            patch(
                "person_anonymizer.postprocessing.tqdm",
                side_effect=lambda **kw: MagicMock(update=MagicMock(), close=MagicMock()),
            ),
        ):
            result = run_post_render_check("video.avi", model, 0.5, {}, config, check_scales=None)

        # Assert — nessuna detection con mock vuoto, lista vuota
        assert result == []

    def test_detection_updates_report_data(self):
        # Arrange — frame_idx=0 presente in report_data → aggiorna post_check_alerts
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap_mock = self._make_cap_mock([frame])

        fake_box = MagicMock()
        fake_box.xyxy = [np.array([10.0, 10.0, 50.0, 50.0])]
        fake_box.conf = [0.9]

        mock_result = MagicMock()
        mock_result.boxes = [fake_box]

        model = MagicMock()
        model.return_value = [mock_result]
        report_data = {0: {"detections": 1}}
        config = PipelineConfig()

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap_mock),
            patch("person_anonymizer.postprocessing.cv2.dnn") as mock_dnn,
            patch(
                "person_anonymizer.postprocessing.tqdm",
                side_effect=lambda **kw: MagicMock(update=MagicMock(), close=MagicMock()),
            ),
        ):
            mock_dnn.NMSBoxes.return_value = np.array([[0]])
            run_post_render_check("video.avi", model, 0.5, report_data, config, check_scales=[1.0])

        # Assert — report_data aggiornato
        assert "post_check_alerts" in report_data[0]

    def test_cap_released_on_exception(self):
        # Arrange — cap.read lancia eccezione dopo il primo frame
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap_mock = MagicMock()
        cap_mock.get.side_effect = lambda prop: {3: 100, 4: 80, 7: 2}.get(prop, 0)
        cap_mock.read.side_effect = [
            (True, frame),
            RuntimeError("read error"),
        ]

        model = MagicMock()
        model.return_value = [MagicMock(boxes=[])]
        config = PipelineConfig()

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap_mock),
            patch(
                "person_anonymizer.postprocessing.tqdm",
                side_effect=lambda **kw: MagicMock(update=MagicMock(), close=MagicMock()),
            ),
        ):
            with pytest.raises(RuntimeError):
                run_post_render_check("video.avi", model, 0.5, {}, config, check_scales=[1.0])

        # Assert — cap.release() chiamato nel finally
        cap_mock.release.assert_called()


# ============================================================
# _detect_frame_at_scales / run_post_render_check scale handling
# ============================================================


def _model_with_box(coords=(10.0, 10.0, 50.0, 50.0), conf=0.9):
    fake_box = MagicMock()
    fake_box.xyxy = [np.array(coords)]
    fake_box.conf = [conf]
    model = MagicMock()
    model.return_value = [MagicMock(boxes=[fake_box])]
    return model


class TestDetectFrameAtScales:
    """Verifica la detection multi-scala di un singolo frame."""

    def test_scale_one_calls_model_directly_without_resize(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        model = _model_with_box()

        with patch("person_anonymizer.postprocessing.cv2.resize") as mock_resize:
            boxes = _detect_frame_at_scales(model, frame, [1.0], 0.4)

        mock_resize.assert_not_called()
        model.assert_called_once_with(frame, conf=0.4, classes=[0], verbose=False)
        assert boxes == [[10.0, 10.0, 50.0, 50.0, 0.9]]

    def test_scale_two_resizes_to_double_size_and_uses_detect_and_rescale(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        scaled = np.zeros((160, 200, 3), dtype=np.uint8)
        model = MagicMock()

        with (
            patch(
                "person_anonymizer.postprocessing.cv2.resize", return_value=scaled
            ) as mock_resize,
            patch(
                "person_anonymizer.postprocessing.detect_and_rescale",
                return_value=[[1.0, 2.0, 3.0, 4.0, 0.7]],
            ) as mock_rescale,
        ):
            boxes = _detect_frame_at_scales(model, frame, [2.0], 0.4)

        mock_resize.assert_called_once()
        assert mock_resize.call_args.args == (frame, (200, 160))
        assert mock_resize.call_args.kwargs == {"interpolation": cv2.INTER_CUBIC}
        mock_rescale.assert_called_once_with(model, scaled, 0.4, 2.0)
        model.assert_not_called()
        assert boxes == [[1.0, 2.0, 3.0, 4.0, 0.7]]

    def test_boxes_from_all_scales_are_concatenated_in_scale_order(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        model = _model_with_box()

        with (
            patch("person_anonymizer.postprocessing.cv2.resize", return_value=frame),
            patch(
                "person_anonymizer.postprocessing.detect_and_rescale",
                return_value=[[5.0, 5.0, 6.0, 6.0, 0.5]],
            ),
        ):
            boxes = _detect_frame_at_scales(model, frame, [1.0, 2.0], 0.4)

        assert boxes == [[10.0, 10.0, 50.0, 50.0, 0.9], [5.0, 5.0, 6.0, 6.0, 0.5]]

    def test_no_scales_returns_empty_list(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        assert _detect_frame_at_scales(MagicMock(), frame, [], 0.4) == []


class TestRunPostRenderCheckReportData:
    """report_data viene aggiornato solo per i frame con detection presenti in report."""

    def test_report_updated_only_for_frames_present_in_report_data(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {7: 2}.get(prop, 0)
        cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
        report_data = {1: {"detections": 0}}

        with (
            patch("person_anonymizer.postprocessing.cv2.VideoCapture", return_value=cap),
            patch("person_anonymizer.postprocessing.tqdm", return_value=MagicMock()),
            patch(
                "person_anonymizer.postprocessing.apply_nms",
                return_value=[[10.0, 10.0, 50.0, 50.0, 0.9]],
            ),
        ):
            alerts = run_post_render_check(
                "video.avi",
                _model_with_box(),
                0.5,
                report_data,
                PipelineConfig(),
                check_scales=[1.0],
            )

        assert [a[0] for a in alerts] == [0, 1]
        assert report_data == {1: {"detections": 0, "post_check_alerts": 1}}
        cap.release.assert_called_once()
