"""
Test for the backend pieces of the web UX improvements.

Covers: video info returned on upload (I11), SAM3 capability endpoint (I14),
inline video preview (D2), post-render check outcome (D6).
"""

import io
import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import person_anonymizer.web.app as web_app
from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import FisheyeContext, PostCheckOutcome
from person_anonymizer.web.pipeline_runner import PipelineRunner

FPS = 10
N_FRAMES = 25
WIDTH, HEIGHT = 64, 48


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(web_app.app, "output_dir", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    web_app.app.config["TESTING"] = True
    web_app.app.config["RATELIMIT_ENABLED"] = False
    with web_app.app.test_client() as c:
        yield c


def _make_mp4_bytes(tmp_path) -> bytes:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


# --- I11: upload returns duration and resolution ---------------------------


def test_upload_returns_duration_and_resolution(client, tmp_path):
    data = {"video": (io.BytesIO(_make_mp4_bytes(tmp_path)), "clip.mp4")}

    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")

    body = resp.get_json()
    assert resp.status_code == 200
    assert body["width"] == WIDTH
    assert body["height"] == HEIGHT
    assert body["duration_s"] == pytest.approx(N_FRAMES / FPS, abs=0.2)


def test_upload_unreadable_video_returns_null_info(client):
    # Valid ISO BMFF magic bytes, but not a decodable video
    data = {"video": (io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64), "x.mp4")}

    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")

    body = resp.get_json()
    assert resp.status_code == 200
    assert body["duration_s"] is None
    assert body["width"] is None


# --- I14: capabilities endpoint --------------------------------------------


@pytest.mark.parametrize("available", [True, False])
def test_capabilities_reports_sam3_availability(client, available):
    with patch("person_anonymizer.web.routes_output.check_sam3_available", return_value=available):
        resp = client.get("/api/capabilities")

    assert resp.status_code == 200
    assert resp.get_json() == {"sam3": available}


# --- D2: inline preview ----------------------------------------------------


def _make_output(client_app_dir, job_id="aabbccddeeff"):
    job = client_app_dir / job_id
    job.mkdir()
    (job / "clip_anonymized.mp4").write_bytes(b"video-bytes")
    (job / "clip_report.csv").write_text("a,b\n")
    return job_id


def test_download_video_inline_has_no_attachment_header(client):
    job_id = _make_output(web_app.app.output_dir)

    resp = client.get(f"/api/download/{job_id}/video?inline=1")

    assert resp.status_code == 200
    assert "attachment" not in resp.headers.get("Content-Disposition", "")
    assert resp.data == b"video-bytes"


def test_download_video_default_is_attachment(client):
    job_id = _make_output(web_app.app.output_dir)

    resp = client.get(f"/api/download/{job_id}/video")

    assert "attachment" in resp.headers["Content-Disposition"]


def test_download_inline_ignored_for_non_video(client):
    job_id = _make_output(web_app.app.output_dir)

    resp = client.get(f"/api/download/{job_id}/report?inline=1")

    assert "attachment" in resp.headers["Content-Disposition"]


# --- D6: post-render check outcome -----------------------------------------


def _run_loop(config, alerts=None, filtered=None):
    from person_anonymizer.stage_refinement import run_refinement_loop

    with (
        patch("person_anonymizer.stage_refinement.render_video"),
        patch(
            "person_anonymizer.stage_refinement.run_post_render_check", return_value=alerts or []
        ),
        patch(
            "person_anonymizer.stage_refinement.filter_artifact_detections", return_value=filtered
        ),
        patch(
            "person_anonymizer.stage_refinement.box_to_polygon",
            return_value=[(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
    ):
        return run_refinement_loop(
            input_path="in.mp4",
            annotations={1: {"auto": [], "manual": [], "intensities": []}},
            model=MagicMock(),
            config=config,
            fps=25.0,
            frame_w=640,
            frame_h=480,
            method="pixelation",
            fisheye=FisheyeContext(),
            report_data={},
            temp_video_path="t.avi",
        )


def _config(**kw):
    return PipelineConfig(max_refinement_passes=2, enable_adaptive_intensity=False, **kw)


def test_outcome_skipped_when_check_disabled():
    *_, outcome = _run_loop(_config(enable_post_render_check=False))

    assert outcome == PostCheckOutcome(status="skipped")


def test_outcome_clean_when_no_alerts():
    *_, outcome = _run_loop(_config())

    assert outcome == PostCheckOutcome(status="clean")


def test_outcome_clean_when_all_alerts_are_artifacts():
    alerts = [(1, 1, [[0, 0, 10, 10, 0.9]])]

    *_, outcome = _run_loop(_config(), alerts=alerts, filtered=([], 1, 0))

    assert outcome == PostCheckOutcome(status="clean")


def test_outcome_residual_counts_frames_after_last_pass():
    alerts = [(1, 1, [[0, 0, 10, 10, 0.9]]), (2, 1, [[0, 0, 10, 10, 0.9]])]
    genuine = [(1, [[0, 0, 10, 10, 0.9]]), (2, [[0, 0, 10, 10, 0.9]])]

    *_, outcome = _run_loop(_config(), alerts=alerts, filtered=(genuine, 0, 2))

    assert outcome == PostCheckOutcome(status="residual", residual_frames=2)


def test_completed_event_includes_post_check(tmp_path):
    sse = MagicMock()
    runner = PipelineRunner(sse, tmp_path)
    outcome = PostCheckOutcome(status="residual", residual_frames=3)

    with (
        patch("person_anonymizer.pipeline.run_pipeline", return_value=outcome),
        patch("person_anonymizer.web.pipeline_runner.TqdmCapture"),
        patch("person_anonymizer.web.pipeline_runner.StdoutCapture"),
    ):
        t = threading.Thread(
            target=runner._run,
            args=("aabbccddeeff", str(tmp_path / "v.mp4"), {"operation_mode": "auto"}, None),
        )
        t.start()
        t.join(timeout=5.0)

    completed = [c[0][2] for c in sse.emit.call_args_list if c[0][1] == "completed"]
    assert completed[0]["post_check"] == {"status": "residual", "residual_frames": 3}


# --- I11: local preview of the uploaded file needs blob: media ---------------


def test_csp_allows_blob_media_for_local_preview(client):
    resp = client.get("/")

    csp = resp.headers["Content-Security-Policy"]
    assert "media-src 'self' blob:" in csp
    assert "unsafe-inline" not in csp


# --- P5: a confirmed stop ends with "stopped", never "completed" -------------


def test_stopped_pipeline_emits_only_stopped(tmp_path):
    from person_anonymizer.models import PipelineStoppedError

    sse = MagicMock()
    runner = PipelineRunner(sse, tmp_path)

    def stop_and_raise(*_a, **_kw):
        runner._stop_event.set()
        raise PipelineStoppedError("Pipeline interrotta dall'utente")

    with (
        patch("person_anonymizer.pipeline.run_pipeline", side_effect=stop_and_raise),
        patch("person_anonymizer.web.pipeline_runner.TqdmCapture"),
        patch("person_anonymizer.web.pipeline_runner.StdoutCapture"),
    ):
        t = threading.Thread(
            target=runner._run,
            args=("aabbccddeeff", str(tmp_path / "v.mp4"), {"operation_mode": "auto"}, None),
        )
        t.start()
        t.join(timeout=5.0)

    events = [c[0][1] for c in sse.emit.call_args_list]
    assert events.count("stopped") == 1
    assert "error" not in events
    assert "completed" not in events


# --- Stop during manual review must unblock the pipeline thread -------------


def test_review_cancel_unblocks_wait_for_completion():
    from person_anonymizer.web.review_state import ReviewState

    state = ReviewState()
    result = {}
    t = threading.Thread(
        target=lambda: result.update(done=state.wait_for_completion()), daemon=True
    )
    t.start()

    state.cancel()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert "done" in result


def test_runner_stop_cancels_pending_review(tmp_path):
    runner = PipelineRunner(MagicMock(), tmp_path)
    alive = threading.Event()
    runner._thread = threading.Thread(target=alive.wait, daemon=True)
    runner._thread.start()
    runner._current_job_id = "aabbccddeeff"
    try:
        with patch.object(runner.review_state, "cancel") as mock_cancel:
            stopped = runner.stop("aabbccddeeff")
    finally:
        alive.set()
        runner._thread.join()

    assert stopped is True
    mock_cancel.assert_called_once()


def test_stop_after_last_check_emits_completed_only(tmp_path):
    sse = MagicMock()
    runner = PipelineRunner(sse, tmp_path)

    def finish_then_stop(*_a, **_kw):
        runner._stop_event.set()
        return PostCheckOutcome(status="clean")

    with (
        patch("person_anonymizer.pipeline.run_pipeline", side_effect=finish_then_stop),
        patch("person_anonymizer.web.pipeline_runner.TqdmCapture"),
        patch("person_anonymizer.web.pipeline_runner.StdoutCapture"),
    ):
        t = threading.Thread(
            target=runner._run,
            args=("aabbccddeeff", str(tmp_path / "v.mp4"), {"operation_mode": "auto"}, None),
        )
        t.start()
        t.join(timeout=5.0)

    events = [c[0][1] for c in sse.emit.call_args_list]
    assert events.count("completed") == 1
    assert "stopped" not in events


def test_video_info_nan_metadata_returns_nulls(tmp_path):
    from person_anonymizer.web.video_info import read_video_info

    cap = MagicMock()
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: float("nan"),
        cv2.CAP_PROP_FRAME_COUNT: 100.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
    }[prop]
    with patch("person_anonymizer.web.video_info.cv2.VideoCapture", return_value=cap):
        info = read_video_info(tmp_path / "x.mp4")

    assert info == {"duration_s": None, "width": None, "height": None}


def test_index_declares_favicon_that_is_served(client):
    html = client.get("/").get_data(as_text=True)

    assert '<link rel="icon" href="/static/favicon.svg"' in html
    icon = client.get("/static/favicon.svg")
    assert icon.status_code == 200
    assert icon.mimetype == "image/svg+xml"
