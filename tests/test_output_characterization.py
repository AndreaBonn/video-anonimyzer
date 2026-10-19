"""
Test di caratterizzazione di save_outputs / load_annotations_from_json.

Rete di sicurezza per il refactor di save_outputs: passano solo dal contratto
pubblico, con file reali in tmp_path e mock ai soli boundary di encoding.
"""

import csv
import json
import os
from unittest.mock import patch

import pytest

from person_anonymizer.config import VERSION, PipelineConfig
from person_anonymizer.models import (
    OutputPaths,
    PipelineContext,
    PipelineResult,
    VideoMeta,
)
from person_anonymizer.output import load_annotations_from_json, save_outputs

ENCODE_AUDIO = "person_anonymizer.postprocessing.encode_with_audio"
ENCODE_NOAUDIO = "person_anonymizer.postprocessing.encode_without_audio"

CSV_FIELDS = [
    "frame_number",
    "persons_detected",
    "avg_confidence",
    "min_confidence",
    "max_confidence",
    "motion_zones",
    "sliding_window_hits",
    "multiscale_hits",
    "post_check_alerts",
]


def _result(**kwargs):
    defaults = dict(
        annotations={},
        report_data={},
        review_stats={},
        method="pixelation",
        mode="manual",
        enable_debug=False,
        enable_report=False,
        ffmpeg_available=False,
        actual_refinement_passes=0,
        refinement_annotations_added=0,
    )
    defaults.update(kwargs)
    return PipelineResult(**defaults)


def _paths(tmp_path):
    return OutputPaths(
        output=str(tmp_path / "out.mp4"),
        temp_video=str(tmp_path / "temp.avi"),
        temp_debug=str(tmp_path / "temp_debug.avi"),
        debug=str(tmp_path / "debug.mp4"),
        report=str(tmp_path / "report.csv"),
        json=str(tmp_path / "annotations.json"),
    )


def _ctx(**kwargs):
    defaults = dict(
        input="test.mp4",
        mode="manual",
        method="pixelation",
        output=None,
        no_debug=True,
        no_report=True,
        review=None,
        normalize=False,
    )
    defaults.update(kwargs)
    return PipelineContext(**defaults)


def _row(fidx, persons):
    return {
        "frame_number": fidx,
        "persons_detected": persons,
        "avg_confidence": 0.5,
        "min_confidence": 0.25,
        "max_confidence": 0.75,
        "motion_zones": 1,
        "sliding_window_hits": 2,
        "multiscale_hits": 3,
        "post_check_alerts": 4,
    }


def _sub(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def _save(tmp_path, result, ctx=None, config=None, meta=None, temp=b"VIDEO"):
    """Esegue save_outputs con un temp video reale e senza encoding."""
    paths = _paths(tmp_path)
    (tmp_path / "temp.avi").write_bytes(temp)
    save_outputs(
        ctx or _ctx(),
        result,
        config or PipelineConfig(),
        "/some/dir/input.mp4",
        paths,
        meta or VideoMeta(fps=25.0, frame_w=640, frame_h=480, total_frames=10),
    )
    return paths


class TestJsonRoundTrip:
    """Il JSON scritto da save_outputs è il contratto con --review."""

    def test_round_trip_preserves_auto_and_manual_polygons(self, tmp_path):
        annotations = {
            0: {
                "auto": [[(0, 0), (10, 0), (10, 80), (0, 80)]],
                "manual": [[(5, 5), (6, 5), (6, 6)]],
                "intensities": [999],
            },
            7: {"auto": [], "manual": [[(1, 2), (3, 4), (5, 6)]], "intensities": []},
        }
        paths = _save(tmp_path, _result(annotations=annotations))

        loaded, mode = load_annotations_from_json(paths.json, PipelineConfig())

        assert mode == "manual"
        assert sorted(loaded) == [0, 7]
        assert loaded[0]["auto"] == [[(0, 0), (10, 0), (10, 80), (0, 80)]]
        assert loaded[0]["manual"] == [[(5, 5), (6, 5), (6, 6)]]
        assert loaded[7]["auto"] == []
        assert loaded[7]["manual"] == [[(1, 2), (3, 4), (5, 6)]]

    def test_round_trip_recomputes_intensity_for_known_box(self, tmp_path):
        # Box alto 80px, base 10, riferimento 80: adattiva = int(10 * 1.0) = 10,
        # minimo = max(3, 80 // 4) = 20 -> 20. Il valore salvato (999) è ignorato.
        annotations = {
            0: {"auto": [[(0, 0), (10, 0), (10, 80), (0, 80)]], "manual": [], "intensities": [999]}
        }
        paths = _save(tmp_path, _result(annotations=annotations))

        loaded, _ = load_annotations_from_json(paths.json, PipelineConfig())

        assert loaded[0]["intensities"] == [20]

    def test_round_trip_intensity_is_base_when_adaptive_disabled(self, tmp_path):
        annotations = {
            0: {"auto": [[(0, 0), (10, 0), (10, 80), (0, 80)]], "manual": [], "intensities": [999]}
        }
        paths = _save(tmp_path, _result(annotations=annotations))

        loaded, _ = load_annotations_from_json(
            paths.json, PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=7)
        )

        assert loaded[0]["intensities"] == [7]


class TestJsonContent:
    """Struttura del JSON scritto."""

    def test_header_fields_match_inputs(self, tmp_path):
        config = PipelineConfig(
            yolo_model="custom.pt",
            detection_confidence=0.3,
            nms_iou_threshold=0.6,
            nms_iou_internal=0.4,
            inference_scales=[1.0, 2.0],
            sliding_window_grid=4,
            person_padding=20,
            anonymization_intensity=12,
            adaptive_reference_height=90,
            smoothing_alpha=0.5,
            ghost_frames=7,
            max_refinement_passes=5,
            refinement_overlap_threshold=0.25,
        )
        result = _result(
            method="blur",
            mode="manual",
            review_stats={"reviewed": 3},
            actual_refinement_passes=2,
            refinement_annotations_added=9,
        )
        meta = VideoMeta(fps=30.0, frame_w=1920, frame_h=1080, total_frames=100)

        paths = _save(tmp_path, result, config=config, meta=meta)
        data = json.loads(open(paths.json).read())

        assert data["schema_version"] == "2.0"
        assert data["tool_version"] == VERSION
        assert data["mode"] == "manual"
        assert data["review_stats"] == {"reviewed": 3}
        assert data["video"] == {
            "filename": "input.mp4",
            "total_frames": 100,
            "fps": 30.0,
            "resolution": [1920, 1080],
        }
        assert data["pipeline_config"] == {
            "yolo_model": "custom.pt",
            "detection_confidence": 0.3,
            "nms_iou_threshold": 0.6,
            "nms_iou_internal": 0.4,
            "inference_scales": [1.0, 2.0],
            "sliding_window_grid": 4,
            "padding": 20,
            "anonymization_method": "blur",
            "base_intensity": 12,
            "adaptive_reference_height": 90,
            "smoothing_alpha": 0.5,
            "ghost_frames": 7,
        }
        assert data["refinement"] == {
            "max_passes": 5,
            "actual_passes": 2,
            "annotations_added": 9,
            "overlap_threshold": 0.25,
        }
        assert isinstance(data["generated"], str) and "T" in data["generated"]

    def test_frame_keys_are_strings_and_points_are_lists(self, tmp_path):
        annotations = {3: {"auto": [[(1, 2), (3, 4)]], "manual": [], "intensities": [5]}}

        paths = _save(tmp_path, _result(annotations=annotations))
        data = json.loads(open(paths.json).read())

        assert data["frames"] == {
            "3": {"auto": [[[1, 2], [3, 4]]], "manual": [], "intensities": [5]}
        }

    def test_intensities_key_written_when_non_empty(self, tmp_path):
        annotations = {0: {"auto": [[(0, 0), (1, 1)]], "manual": [], "intensities": [8]}}

        paths = _save(tmp_path, _result(annotations=annotations))
        data = json.loads(open(paths.json).read())

        assert data["frames"]["0"]["intensities"] == [8]

    @pytest.mark.parametrize("ann", [{"auto": [], "manual": [], "intensities": []}, {"auto": []}])
    def test_empty_or_missing_intensities_omit_key(self, tmp_path, ann):
        paths = _save(tmp_path, _result(annotations={0: ann}))
        data = json.loads(open(paths.json).read())

        assert data["frames"]["0"] == {"auto": [], "manual": []}
        assert "intensities" not in data["frames"]["0"]

    def test_json_written_in_auto_mode_only_with_normalize(self, tmp_path):
        auto = _result(mode="auto", annotations={0: {"auto": [], "manual": []}})
        without = _save(_sub(tmp_path, "a"), auto, ctx=_ctx(normalize=False))
        assert not os.path.isfile(without.json)

        with_norm = _save(_sub(tmp_path, "b"), auto, ctx=_ctx(normalize=True))
        assert os.path.isfile(with_norm.json)
        assert json.loads(open(with_norm.json).read())["mode"] == "auto"


class TestCsvReport:
    """Report CSV."""

    def test_header_and_rows_follow_frame_order(self, tmp_path):
        # Chiavi inserite fuori ordine: l'output è ordinato per frame.
        report = {5: _row(5, 2), 1: _row(1, 7), 3: _row(3, 0)}

        paths = _save(tmp_path, _result(enable_report=True, report_data=report))

        with open(paths.report, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_FIELDS
            rows = list(reader)
        assert [r["frame_number"] for r in rows] == ["1", "3", "5"]
        assert [r["persons_detected"] for r in rows] == ["7", "0", "2"]
        assert rows[0] == {k: str(v) for k, v in _row(1, 7).items()}

    def test_empty_report_data_writes_header_only(self, tmp_path):
        paths = _save(tmp_path, _result(enable_report=True, report_data={}))

        assert open(paths.report).read().strip() == ",".join(CSV_FIELDS)

    def test_csv_absent_when_disabled_but_present_when_enabled(self, tmp_path):
        data = {0: _row(0, 1)}
        disabled = _save(_sub(tmp_path, "off"), _result(report_data=data))
        enabled = _save(_sub(tmp_path, "on"), _result(enable_report=True, report_data=data))

        assert not os.path.exists(disabled.report)
        assert os.path.isfile(enabled.report)


class TestDebugAndCleanup:
    """Debug e file temporanei."""

    def test_no_ffmpeg_debug_enabled_copies_both_files(self, tmp_path):
        paths = _paths(tmp_path)
        (tmp_path / "temp_debug.avi").write_bytes(b"DEBUG")

        _save(tmp_path, _result(enable_debug=True), temp=b"VIDEO")

        assert open(paths.output, "rb").read() == b"VIDEO"
        assert open(paths.debug, "rb").read() == b"DEBUG"
        assert not os.path.exists(paths.temp_video)
        assert not os.path.exists(paths.temp_debug)

    def test_no_ffmpeg_debug_disabled_leaves_no_debug_but_removes_temp_debug(self, tmp_path):
        paths = _paths(tmp_path)
        (tmp_path / "temp_debug.avi").write_bytes(b"DEBUG")

        _save(tmp_path, _result(enable_debug=False), temp=b"VIDEO")

        assert os.path.isfile(paths.output)
        assert not os.path.exists(paths.debug)
        assert not os.path.exists(paths.temp_video)
        assert not os.path.exists(paths.temp_debug)

    def test_ffmpeg_debug_enabled_encodes_debug_without_audio(self, tmp_path):
        paths = _paths(tmp_path)
        (tmp_path / "temp_debug.avi").write_bytes(b"DEBUG")

        with patch(ENCODE_AUDIO) as enc_audio, patch(ENCODE_NOAUDIO) as enc_noaudio:
            _save(tmp_path, _result(ffmpeg_available=True, enable_debug=True))

        enc_audio.assert_called_once_with(paths.temp_video, "/some/dir/input.mp4", paths.output)
        enc_noaudio.assert_called_once_with(paths.temp_debug, paths.debug)
        assert not os.path.exists(paths.temp_video)
        assert not os.path.exists(paths.temp_debug)

    def test_ffmpeg_debug_disabled_skips_debug_encoding_and_removes_temp_debug(self, tmp_path):
        paths = _paths(tmp_path)
        (tmp_path / "temp_debug.avi").write_bytes(b"DEBUG")

        with patch(ENCODE_AUDIO) as enc_audio, patch(ENCODE_NOAUDIO) as enc_noaudio:
            _save(tmp_path, _result(ffmpeg_available=True, enable_debug=False))

        enc_audio.assert_called_once()
        enc_noaudio.assert_not_called()
        assert not os.path.exists(paths.temp_video)
        assert not os.path.exists(paths.temp_debug)

    def test_ffmpeg_debug_enabled_but_temp_debug_missing_skips_debug_encoding(self, tmp_path):
        with patch(ENCODE_AUDIO), patch(ENCODE_NOAUDIO) as enc_noaudio:
            _save(tmp_path, _result(ffmpeg_available=True, enable_debug=True))

        enc_noaudio.assert_not_called()


class TestFfmpegUnavailable:
    """Senza ffmpeg l'output è una copia byte per byte del temp."""

    def test_output_is_byte_copy_and_encoders_not_called(self, tmp_path):
        with patch(ENCODE_AUDIO) as enc_audio, patch(ENCODE_NOAUDIO) as enc_noaudio:
            paths = _save(tmp_path, _result(ffmpeg_available=False), temp=b"\x01\x02\x03")

        assert open(paths.output, "rb").read() == b"\x01\x02\x03"
        enc_audio.assert_not_called()
        enc_noaudio.assert_not_called()

    def test_no_message_printed_by_save_outputs_on_fallback(self, tmp_path, capsys):
        # save_outputs non stampa nulla sul fallback: l'avviso "ffmpeg assente"
        # (se esiste) vive altrove, fuori dal suo contratto.
        _save(tmp_path, _result(ffmpeg_available=False))

        assert capsys.readouterr().out == ""


class TestEncodingFailureCurrentBehavior:
    """Comportamento ATTUALE se l'encoding fallisce. NON è dichiarato corretto."""

    def test_current_behavior_encode_error_propagates_and_temp_is_deleted(self, tmp_path):
        paths = _paths(tmp_path)
        (tmp_path / "temp_debug.avi").write_bytes(b"DEBUG")

        with patch(ENCODE_AUDIO, side_effect=RuntimeError("ffmpeg boom")):
            with pytest.raises(RuntimeError, match="ffmpeg boom"):
                _save(tmp_path, _result(ffmpeg_available=True, enable_debug=True))

        # Il finally cancella i temp anche se l'output non è mai stato prodotto:
        # il video elaborato è perso. Fotografia dello stato attuale.
        assert not os.path.exists(paths.output)
        assert not os.path.exists(paths.temp_video)
        assert not os.path.exists(paths.temp_debug)

    def test_current_behavior_encode_error_skips_report_and_json(self, tmp_path):
        paths = _paths(tmp_path)
        result = _result(
            ffmpeg_available=True,
            enable_report=True,
            report_data={0: _row(0, 1)},
            annotations={0: {"auto": [], "manual": []}},
        )

        with patch(ENCODE_AUDIO, side_effect=RuntimeError("ffmpeg boom")):
            with pytest.raises(RuntimeError):
                _save(tmp_path, result)

        assert not os.path.exists(paths.report)
        assert not os.path.exists(paths.json)
