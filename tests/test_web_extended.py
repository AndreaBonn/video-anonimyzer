"""
Test aggiuntivi per massimizzare coverage di:
  - person_anonymizer/web/app.py
  - person_anonymizer/web/routes_review.py
  - person_anonymizer/web/routes_output.py
  - person_anonymizer/web/middleware.py
  - person_anonymizer/web/sse_manager.py
  - person_anonymizer/web/config_validator.py

Pattern: AAA. Naming: test_<funzione>_<scenario>_<risultato>.
"""

import io
import json
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.web.app import app
from person_anonymizer.web.extensions import limiter


# ---------------------------------------------------------------------------
# Fixture condivisa
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Flask test client con TESTING abilitato e rate limiting disabilitato."""
    app.config["TESTING"] = True
    app.config["RATELIMIT_ENABLED"] = False
    app.config["RATELIMIT_STORAGE_URI"] = "memory://"
    limiter.enabled = False
    with app.test_client() as c:
        yield c
    limiter.enabled = True


# ---------------------------------------------------------------------------
# TestMiddleware — middleware.py righe 25-28, 49
# ---------------------------------------------------------------------------


class TestMiddlewareCsrf:
    """Verifica comportamento CSRF fuori da TESTING mode."""

    def test_csrf_check_blocks_post_without_header_in_prod_mode(self):
        """csrf_check rifiuta POST senza X-Requested-With quando TESTING=False."""
        # Arrange — disabilita temporaneamente TESTING
        app.config["TESTING"] = False
        app.config["RATELIMIT_ENABLED"] = False
        try:
            with app.test_client() as c:
                # Act — POST senza header CSRF
                resp = c.post(
                    "/api/stop",
                    data=json.dumps({"job_id": "aabbccddeeff"}),
                    content_type="application/json",
                )
                # Assert
                assert resp.status_code == 403
                data = resp.get_json()
                assert data is not None
                assert "CSRF" in data["error"]
        finally:
            app.config["TESTING"] = True

    def test_csrf_check_allows_post_with_x_requested_with_header(self):
        """csrf_check lascia passare POST con X-Requested-With."""
        # Arrange — disabilita temporaneamente TESTING
        app.config["TESTING"] = False
        app.config["RATELIMIT_ENABLED"] = False
        try:
            with app.test_client() as c:
                # Act — POST con header CSRF
                resp = c.post(
                    "/api/stop",
                    data=json.dumps({"job_id": "aabbccddeeff"}),
                    content_type="application/json",
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )
                # Assert — deve passare la guardia CSRF (può dare 400 per job non attivo)
                assert resp.status_code != 403
        finally:
            app.config["TESTING"] = True

    def test_csrf_check_allows_get_without_header(self, client):
        """csrf_check non blocca GET (lettura sicura)."""
        # Arrange / Act
        resp = client.get("/api/status")

        # Assert
        assert resp.status_code == 200

    def test_add_request_id_sets_header(self, client):
        """add_request_id aggiunge X-Request-ID nella response."""
        # Arrange / Act
        resp = client.get("/api/status")

        # Assert
        assert "X-Request-ID" in resp.headers
        rid = resp.headers["X-Request-ID"]
        assert len(rid) == 16
        assert rid.isalnum()

    def test_security_headers_hsts_not_present_on_http(self, client):
        """HSTS non deve essere presente su connessioni non sicure."""
        # Arrange / Act
        resp = client.get("/")

        # Assert — is_secure è False nel test client → niente HSTS
        assert "Strict-Transport-Security" not in resp.headers

    def test_security_headers_x_xss_protection_disabled(self, client):
        """X-XSS-Protection deve essere '0' (disabilitato, browser moderni)."""
        # Arrange / Act
        resp = client.get("/")

        # Assert
        assert resp.headers.get("X-XSS-Protection") == "0"

    def test_security_headers_referrer_policy(self, client):
        """Referrer-Policy deve essere strict-origin-when-cross-origin."""
        # Arrange / Act
        resp = client.get("/")

        # Assert
        assert (
            resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        )

    def test_security_headers_permissions_policy(self, client):
        """Permissions-Policy deve disabilitare camera, mic, geoloc."""
        # Arrange / Act
        resp = client.get("/")

        # Assert
        pp = resp.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp
        assert "geolocation=()" in pp


# ---------------------------------------------------------------------------
# TestSSEManager — sse_manager.py righe 39-62
# ---------------------------------------------------------------------------


class TestSSEManagerEmit:
    """Verifica SSEManager.emit e close."""

    def test_emit_delivers_event_to_subscriber(self):
        """emit mette l'evento nella coda del subscriber."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001122"
        q = mgr.subscribe(job_id)

        # Act
        mgr.emit(job_id, "progress", {"pct": 50})

        # Assert
        event = q.get_nowait()
        assert event["type"] == "progress"
        assert event["data"]["pct"] == 50

    def test_emit_delivers_to_multiple_subscribers(self):
        """emit distribuisce l'evento a tutti i subscriber del job."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001133"
        q1 = mgr.subscribe(job_id)
        q2 = mgr.subscribe(job_id)

        # Act
        mgr.emit(job_id, "started", {"job_id": job_id})

        # Assert
        e1 = q1.get_nowait()
        e2 = q2.get_nowait()
        assert e1["type"] == "started"
        assert e2["type"] == "started"

    def test_emit_no_subscribers_does_not_raise(self):
        """emit su job senza subscriber non deve sollevare eccezioni."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()

        # Act / Assert — nessuna eccezione
        mgr.emit("nonexistent_job", "test", {})

    def test_emit_full_queue_drops_event_silently(self):
        """emit su coda piena scarta l'evento senza eccezione."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001144"
        q = mgr.subscribe(job_id)

        # Riempie la coda (maxsize=200)
        for i in range(200):
            q.put_nowait({"type": "fill", "data": {"i": i}})

        # Act — coda piena, l'evento deve essere scartato
        mgr.emit(job_id, "overflow", {"x": 1})  # non deve sollevare

        # Assert — la coda è ancora piena (200 eventi, quello in overflow scartato)
        assert q.full()

    def test_close_sends_none_sentinel_to_subscriber(self):
        """close invia None nelle code dei subscriber (segnale di fine stream)."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001155"
        q = mgr.subscribe(job_id)

        # Act
        mgr.close(job_id)

        # Assert
        sentinel = q.get(timeout=2)
        assert sentinel is None

    def test_close_removes_job_from_subscribers(self):
        """Dopo close, il job non deve più avere subscriber registrati."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001166"
        mgr.subscribe(job_id)

        # Act
        mgr.close(job_id)

        # Assert — nessun subscriber rimasto
        assert job_id not in mgr._subscribers

    def test_unsubscribe_unknown_queue_does_not_raise(self):
        """unsubscribe con una coda non registrata non deve sollevare eccezioni."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001177"
        q = queue.Queue()

        # Act / Assert — nessuna eccezione
        mgr.unsubscribe(job_id, q)

    def test_unsubscribe_last_subscriber_removes_job_key(self):
        """Dopo aver rimosso l'unico subscriber, la chiave job viene eliminata."""
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager

        mgr = SSEManager()
        job_id = "aabbcc001188"
        q = mgr.subscribe(job_id)

        # Act
        mgr.unsubscribe(job_id, q)

        # Assert
        assert job_id not in mgr._subscribers


# ---------------------------------------------------------------------------
# TestAppUploadJson — app.py upload-json endpoint
# ---------------------------------------------------------------------------


class TestUploadJson:
    """Verifica endpoint /api/upload-json."""

    def test_upload_json_missing_file_returns_400(self, client):
        """POST senza json_file restituisce 400."""
        # Arrange / Act
        resp = client.post(
            "/api/upload-json",
            data={"job_id": "aabbccddeeff"},
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_upload_json_empty_filename_returns_400(self, client):
        """POST con filename vuoto restituisce 400."""
        # Arrange
        data = {
            "json_file": (io.BytesIO(b"{}"), ""),
            "job_id": "aabbccddeeff",
        }
        # Act
        resp = client.post(
            "/api/upload-json",
            data=data,
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_upload_json_non_json_extension_returns_400(self, client):
        """POST con file .txt restituisce 400."""
        # Arrange
        data = {
            "json_file": (io.BytesIO(b"{}"), "data.txt"),
            "job_id": "aabbccddeeff",
        }
        # Act
        resp = client.post(
            "/api/upload-json",
            data=data,
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 400
        assert "json" in resp.get_json()["error"].lower()

    def test_upload_json_missing_job_id_returns_400(self, client):
        """POST senza job_id restituisce 400."""
        # Arrange
        data = {
            "json_file": (io.BytesIO(b'{"frames": {}}'), "annotations.json"),
        }
        # Act
        resp = client.post(
            "/api/upload-json",
            data=data,
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 400
        assert "job_id" in resp.get_json()["error"].lower()

    def test_upload_json_invalid_job_id_returns_400(self, client):
        """POST con job_id non valido restituisce 400."""
        # Arrange
        data = {
            "json_file": (io.BytesIO(b'{"frames": {}}'), "annotations.json"),
            "job_id": "INVALID_ID!",
        }
        # Act
        resp = client.post(
            "/api/upload-json",
            data=data,
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_upload_json_nonexistent_job_dir_returns_404(self, client):
        """POST con job_id valido ma cartella inesistente restituisce 404."""
        # Arrange
        data = {
            "json_file": (io.BytesIO(b'{"frames": {}}'), "annotations.json"),
            "job_id": "aabbccddeeff",
        }
        # Act — la cartella upload/aabbccddeeff non esiste
        resp = client.post(
            "/api/upload-json",
            data=data,
            content_type="multipart/form-data",
        )
        # Assert
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_upload_json_invalid_json_content_returns_400(self, client, tmp_path):
        """POST con contenuto non JSON restituisce 400."""
        # Arrange — crea la cartella job per superare il check 404
        from person_anonymizer.web.app import UPLOAD_DIR

        job_id = "aabbccddeeff"
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            data = {
                "json_file": (io.BytesIO(b"not valid json!!!"), "annotations.json"),
                "job_id": job_id,
            }
            # Act
            resp = client.post(
                "/api/upload-json",
                data=data,
                content_type="multipart/form-data",
            )
            # Assert
            assert resp.status_code == 400
            assert "JSON" in resp.get_json()["error"]
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_upload_json_json_array_returns_400(self, client):
        """POST con JSON array (non oggetto) restituisce 400."""
        # Arrange — crea cartella job
        from person_anonymizer.web.app import UPLOAD_DIR

        job_id = "aabbccddeeff"
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            data = {
                "json_file": (io.BytesIO(b"[1, 2, 3]"), "annotations.json"),
                "job_id": job_id,
            }
            # Act
            resp = client.post(
                "/api/upload-json",
                data=data,
                content_type="multipart/form-data",
            )
            # Assert
            assert resp.status_code == 400
            assert "oggetto" in resp.get_json()["error"]
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_upload_json_valid_payload_returns_filename(self, client):
        """POST con payload valido restituisce il filename."""
        # Arrange
        from person_anonymizer.web.app import UPLOAD_DIR

        job_id = "aabbccddeeff"
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            content = json.dumps({"frames": {}}).encode()
            data = {
                "json_file": (io.BytesIO(content), "annotations.json"),
                "job_id": job_id,
            }
            # Act
            resp = client.post(
                "/api/upload-json",
                data=data,
                content_type="multipart/form-data",
            )
            # Assert
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["filename"] == "annotations.json"
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestAppStartPipelineExtended — app.py start_pipeline ulteriori scenari
# ---------------------------------------------------------------------------


class TestStartPipelineExtended:
    """Scenari aggiuntivi per /api/start."""

    def test_start_no_json_payload_returns_400(self, client):
        """POST senza body JSON restituisce 400."""
        # Arrange / Act
        resp = client.post("/api/start", content_type="application/json", data="")
        # Assert
        assert resp.status_code == 400

    def test_start_valid_job_id_video_not_found_returns_404(self, client):
        """POST con job_id e video_filename validi ma file assente → 404."""
        # Arrange
        payload = {
            "job_id": "aabbccddeeff",
            "video_filename": "nonexistent.mp4",
        }
        # Act
        resp = client.post("/api/start", json=payload)
        # Assert
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_start_pipeline_already_running_returns_409(self, client):
        """Se una pipeline è già in esecuzione, start restituisce 409."""
        # Arrange — mock pipeline_runner.start per simulare "già in esecuzione"
        from person_anonymizer.web.app import UPLOAD_DIR

        job_id = "aabbccddeeff"
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        # Crea un file video finto per superare il check di esistenza
        fake_video = job_dir / "video.mp4"
        fake_video.write_bytes(b"\x00\x00\x00 ftyp" + b"\x00" * 100)
        try:
            with patch.object(
                app.pipeline_runner, "start", return_value=(False, "Una pipeline è già in esecuzione")
            ):
                payload = {
                    "job_id": job_id,
                    "video_filename": "video.mp4",
                }
                # Act
                resp = client.post("/api/start", json=payload)
                # Assert
                assert resp.status_code == 409
                assert "error" in resp.get_json()
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestAppStopPipeline — app.py stop_pipeline
# ---------------------------------------------------------------------------


class TestStopPipeline:
    """Verifica endpoint /api/stop."""

    def test_stop_missing_job_id_returns_400(self, client):
        """POST con job_id invalido restituisce 400."""
        # Arrange / Act
        resp = client.post("/api/stop", json={})
        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_stop_valid_job_id_no_pipeline_returns_404(self, client):
        """POST con job_id valido ma nessuna pipeline attiva → 404."""
        # Arrange / Act
        resp = client.post("/api/stop", json={"job_id": "aabbccddeeff"})
        # Assert
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_stop_returns_stopping_when_pipeline_active(self, client):
        """POST con pipeline attiva restituisce status stopping."""
        # Arrange
        with patch.object(app.pipeline_runner, "stop", return_value=True):
            # Act
            resp = client.post("/api/stop", json={"job_id": "aabbccddeeff"})
            # Assert
            assert resp.status_code == 200
            assert resp.get_json()["status"] == "stopping"


# ---------------------------------------------------------------------------
# TestAppStatus — app.py status endpoint
# ---------------------------------------------------------------------------


class TestStatus:
    """Verifica endpoint /api/status."""

    def test_status_returns_200_with_running_field(self, client):
        """GET /api/status restituisce running e job_id."""
        # Arrange / Act
        resp = client.get("/api/status")
        # Assert
        assert resp.status_code == 200
        data = resp.get_json()
        assert "running" in data

    def test_status_not_running_returns_false(self, client):
        """Quando nessuna pipeline è attiva, running è False."""
        # Arrange / Act
        resp = client.get("/api/status")
        # Assert
        data = resp.get_json()
        assert data["running"] is False

    def test_status_running_returns_job_id(self, client):
        """Quando pipeline attiva, job_id è presente."""
        # Arrange
        with patch.object(
            app.pipeline_runner,
            "get_status",
            return_value={"running": True, "job_id": "aabbccddeeff"},
        ):
            # Act
            resp = client.get("/api/status")
            # Assert
            data = resp.get_json()
            assert data["running"] is True
            assert data["job_id"] == "aabbccddeeff"


# ---------------------------------------------------------------------------
# TestProgressStream — app.py progress_stream (SSE)
# ---------------------------------------------------------------------------


class TestProgressStream:
    """Verifica endpoint /api/progress (SSE)."""

    def test_progress_invalid_job_id_returns_400(self, client):
        """GET con job_id invalido restituisce 400."""
        # Arrange / Act
        resp = client.get("/api/progress?job_id=bad")
        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_progress_too_many_subscribers_emits_error_event(self, client):
        """Quando SSEManager.subscribe solleva RuntimeError, lo stream emette errore."""
        # Arrange
        with patch.object(
            app.sse_manager,
            "subscribe",
            side_effect=RuntimeError("Troppi client connessi"),
        ):
            # Act
            resp = client.get("/api/progress?job_id=aabbccddeeff", buffered=True)
            # Assert — il content-type è SSE
            assert resp.status_code == 200
            assert "text/event-stream" in resp.content_type
            data = resp.data.decode()
            assert "error" in data
            assert "Troppi client" in data


# ---------------------------------------------------------------------------
# TestRoutesReview — routes_review.py blueprint
# ---------------------------------------------------------------------------


class TestRoutesReview:
    """Verifica tutti gli endpoint del blueprint review."""

    def _make_review_state_mock(self, is_active=True, total_frames=10):
        """Helper per creare un mock ReviewState."""
        rs = MagicMock()
        rs.is_active = is_active
        rs.get_metadata.return_value = {"total_frames": total_frames, "fps": 25}
        rs.get_annotations.return_value = {0: {"auto": [], "manual": []}}
        rs.get_frame_jpeg.return_value = (b"\xff\xd8\xff" + b"\x00" * 100, 1.0)
        return rs

    # --- /api/review/status ---

    def test_review_status_inactive_returns_active_false(self, client):
        """Quando review non attiva, status ritorna active: False."""
        # Arrange
        rs = self._make_review_state_mock(is_active=False)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/status")
            # Assert
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["active"] is False

    def test_review_status_active_returns_metadata(self, client):
        """Quando review attiva, status include i metadati."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/status")
            # Assert
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["active"] is True
            assert data["total_frames"] == 10

    # --- /api/review/frame/<int:frame_idx> ---

    def test_review_frame_not_active_returns_404(self, client):
        """Frame non disponibile quando review non attiva."""
        # Arrange
        rs = self._make_review_state_mock(is_active=False)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/frame/0")
            # Assert
            assert resp.status_code == 404
            assert "error" in resp.get_json()

    def test_review_frame_out_of_range_returns_400(self, client):
        """Frame index fuori range restituisce 400."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=5)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/frame/10")
            # Assert
            assert resp.status_code == 400
            assert "error" in resp.get_json()

    def test_review_frame_valid_returns_jpeg(self, client):
        """Frame valido restituisce JPEG con X-Scale-Factor header."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/frame/0")
            # Assert
            assert resp.status_code == 200
            assert resp.content_type == "image/jpeg"
            assert "X-Scale-Factor" in resp.headers

    def test_review_frame_none_jpeg_returns_404(self, client):
        """Se get_frame_jpeg ritorna None, restituisce 404."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        rs.get_frame_jpeg.return_value = (None, 1.0)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/frame/0")
            # Assert
            assert resp.status_code == 404

    def test_review_frame_max_width_capped_at_1920(self, client):
        """max_width è cappato a 1920 anche se si richiede un valore maggiore."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/frame/0?max_width=9999")
            # Assert
            assert resp.status_code == 200
            # Verifica che get_frame_jpeg sia stato chiamato con max_width <= 1920
            call_args = rs.get_frame_jpeg.call_args
            assert call_args.kwargs.get("max_width", call_args.args[1] if len(call_args.args) > 1 else 1920) <= 1920

    # --- /api/review/annotations ---

    def test_review_annotations_not_active_returns_404(self, client):
        """Annotations non disponibili quando review non attiva."""
        # Arrange
        rs = self._make_review_state_mock(is_active=False)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/annotations")
            # Assert
            assert resp.status_code == 404
            assert "error" in resp.get_json()

    def test_review_annotations_active_returns_dict(self, client):
        """Annotations attiva restituisce dizionario con chiavi stringa."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.get("/api/review/annotations")
            # Assert
            assert resp.status_code == 200
            data = resp.get_json()
            assert isinstance(data, dict)
            # Le chiavi devono essere stringhe (JSON serializza int come str)
            for key in data.keys():
                assert isinstance(key, str)

    # --- /api/review/annotations/<int:frame_idx> PUT ---

    def test_review_update_annotations_not_active_returns_404(self, client):
        """PUT annotations non disponibile quando review non attiva."""
        # Arrange
        rs = self._make_review_state_mock(is_active=False)
        with patch.object(app.pipeline_runner, "review_state", rs):
            payload = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": []}
            # Act
            resp = client.put("/api/review/annotations/0", json=payload)
            # Assert
            assert resp.status_code == 404

    def test_review_update_annotations_out_of_range_returns_400(self, client):
        """PUT annotations con frame fuori range restituisce 400."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=5)
        with patch.object(app.pipeline_runner, "review_state", rs):
            payload = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": []}
            # Act
            resp = client.put("/api/review/annotations/99", json=payload)
            # Assert
            assert resp.status_code == 400

    def test_review_update_annotations_missing_payload_returns_400(self, client):
        """PUT annotations senza body restituisce 400."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.put(
                "/api/review/annotations/0",
                data="",
                content_type="application/json",
            )
            # Assert
            assert resp.status_code == 400

    def test_review_update_annotations_invalid_payload_returns_422(self, client):
        """PUT annotations con payload malformato restituisce 422."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Poligono con solo 2 punti (invalido)
            payload = {"auto": [[[0, 0], [10, 10]]], "manual": []}
            # Act
            resp = client.put("/api/review/annotations/0", json=payload)
            # Assert
            assert resp.status_code == 422
            assert "error" in resp.get_json()

    def test_review_update_annotations_valid_returns_ok(self, client):
        """PUT annotations valido restituisce ok: True."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True, total_frames=10)
        with patch.object(app.pipeline_runner, "review_state", rs):
            payload = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": []}
            # Act
            resp = client.put("/api/review/annotations/0", json=payload)
            # Assert
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            rs.update_annotations.assert_called_once_with(0, payload)

    # --- /api/review/confirm ---

    def test_review_confirm_not_active_returns_404(self, client):
        """POST confirm quando review non attiva restituisce 404."""
        # Arrange
        rs = self._make_review_state_mock(is_active=False)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.post("/api/review/confirm")
            # Assert
            assert resp.status_code == 404

    def test_review_confirm_active_calls_complete(self, client):
        """POST confirm quando review attiva chiama rs.complete e restituisce ok."""
        # Arrange
        rs = self._make_review_state_mock(is_active=True)
        with patch.object(app.pipeline_runner, "review_state", rs):
            # Act
            resp = client.post("/api/review/confirm")
            # Assert
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            rs.complete.assert_called_once()


# ---------------------------------------------------------------------------
# TestRoutesOutput — routes_output.py blueprint
# ---------------------------------------------------------------------------


class TestRoutesOutput:
    """Verifica endpoint del blueprint output."""

    # --- /api/download/<job_id>/<file_type> ---

    def test_download_valid_job_invalid_type_returns_400(self, client, tmp_path):
        """Tipo file non valido restituisce 400."""
        # Arrange — crea cartella output per far passare il check 404
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/invalid_type")
            # Assert
            assert resp.status_code == 400
            assert "error" in resp.get_json()
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_download_valid_type_file_not_found_returns_404(self, client):
        """Tipo valido ma file assente restituisce 404."""
        # Arrange — crea cartella output vuota
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/video")
            # Assert
            assert resp.status_code == 404
            assert "error" in resp.get_json()
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_download_video_file_present_returns_file(self, client):
        """File presente viene scaricato con as_attachment."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        video_file = job_dir / "output_anonymized.mp4"
        video_file.write_bytes(b"fake video content")
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/video")
            # Assert
            assert resp.status_code == 200
            assert "no-store" in resp.headers.get("Cache-Control", "")
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_download_annotations_file_present_returns_file(self, client):
        """File annotations JSON viene scaricato."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        ann_file = job_dir / "output_annotations.json"
        ann_file.write_text('{"frames": {}}')
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/annotations")
            # Assert
            assert resp.status_code == 200
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_download_report_file_present_returns_file(self, client):
        """File report CSV viene scaricato."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        rep_file = job_dir / "output_report.csv"
        rep_file.write_text("frame,count\n0,1\n")
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/report")
            # Assert
            assert resp.status_code == 200
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_download_debug_file_present_returns_file(self, client):
        """File debug MP4 viene scaricato."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        dbg_file = job_dir / "output_debug.mp4"
        dbg_file.write_bytes(b"fake debug video")
        try:
            # Act
            resp = client.get(f"/api/download/{job_id}/debug")
            # Assert
            assert resp.status_code == 200
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    # --- /api/outputs/<job_id> ---

    def test_list_outputs_empty_dir_returns_empty_list(self, client):
        """Cartella output esistente ma vuota restituisce lista vuota."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        try:
            # Act
            resp = client.get(f"/api/outputs/{job_id}")
            # Assert
            assert resp.status_code == 200
            assert resp.get_json()["files"] == []
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_list_outputs_classifies_file_types_correctly(self, client):
        """list_outputs classifica correttamente i tipi di file."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        # Crea file di vari tipi
        (job_dir / "out_anonymized.mp4").write_bytes(b"x")
        (job_dir / "out_debug.mp4").write_bytes(b"x")
        (job_dir / "out_report.csv").write_text("a,b\n")
        (job_dir / "out_annotations.json").write_text("{}")
        try:
            # Act
            resp = client.get(f"/api/outputs/{job_id}")
            # Assert
            assert resp.status_code == 200
            files = resp.get_json()["files"]
            types_found = {f["type"] for f in files}
            assert "video" in types_found
            assert "debug" in types_found
            assert "report" in types_found
            assert "annotations" in types_found
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_list_outputs_unknown_file_type_is_unknown(self, client):
        """File con suffisso non riconosciuto viene classificato come 'unknown'."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        (job_dir / "random_file.bin").write_bytes(b"x")
        try:
            # Act
            resp = client.get(f"/api/outputs/{job_id}")
            # Assert
            files = resp.get_json()["files"]
            assert any(f["type"] == "unknown" for f in files)
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)

    def test_list_outputs_includes_size_mb(self, client):
        """list_outputs include size_mb per ogni file."""
        # Arrange
        from person_anonymizer.web.app import OUTPUT_DIR

        job_id = "aabbccddeeff"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        (job_dir / "out_annotations.json").write_text('{"frames": {}}')
        try:
            # Act
            resp = client.get(f"/api/outputs/{job_id}")
            # Assert
            files = resp.get_json()["files"]
            assert len(files) > 0
            assert "size_mb" in files[0]
            assert isinstance(files[0]["size_mb"], float)
        finally:
            import shutil

            shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestConfigValidator — config_validator.py
# ---------------------------------------------------------------------------


class TestConfigValidatorExtended:
    """Verifica validate_config_params per tutti i validator definiti."""

    def test_validate_valid_operation_mode_auto(self):
        """operation_mode 'auto' è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        # Arrange / Act
        valid, msg = validate_config_params({"operation_mode": "auto"})
        # Assert
        assert valid is True

    def test_validate_invalid_operation_mode_returns_false(self):
        """operation_mode sconosciuto è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        # Arrange / Act
        valid, msg = validate_config_params({"operation_mode": "batch"})
        # Assert
        assert valid is False
        assert "operation_mode" in msg

    def test_validate_bool_field_non_bool_returns_false(self):
        """Campo booleano con valore stringa è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        # Arrange / Act
        valid, msg = validate_config_params({"enable_tracking": "yes"})
        # Assert
        assert valid is False
        assert "booleano" in msg

    def test_validate_bool_field_true_is_valid(self):
        """Campo booleano con True è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        # Arrange / Act
        valid, msg = validate_config_params({"enable_tracking": True})
        # Assert
        assert valid is True

    def test_validate_anonymization_intensity_boundary_min(self):
        """anonymization_intensity = 1 è al limite minimo valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"anonymization_intensity": 1})
        assert valid is True

    def test_validate_anonymization_intensity_boundary_max(self):
        """anonymization_intensity = 100 è al limite massimo valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"anonymization_intensity": 100})
        assert valid is True

    def test_validate_anonymization_intensity_out_of_range(self):
        """anonymization_intensity = 101 è fuori range."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"anonymization_intensity": 101})
        assert valid is False

    def test_validate_detection_confidence_too_low(self):
        """detection_confidence = 0 è fuori range."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"detection_confidence": 0.0})
        assert valid is False

    def test_validate_detection_confidence_too_high(self):
        """detection_confidence = 1.0 è fuori range."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"detection_confidence": 1.0})
        assert valid is False

    def test_validate_detection_confidence_valid(self):
        """detection_confidence = 0.5 è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"detection_confidence": 0.5})
        assert valid is True

    def test_validate_detection_backend_invalid(self):
        """detection_backend con valore non permesso è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"detection_backend": "custom_backend"})
        assert valid is False

    def test_validate_detection_backend_valid_options(self):
        """Tutti i backend validi sono accettati."""
        from person_anonymizer.web.config_validator import validate_config_params

        for backend in ("yolo", "yolo+sam3", "sam3"):
            valid, _ = validate_config_params({"detection_backend": backend})
            assert valid is True, f"Backend '{backend}' dovrebbe essere valido"

    def test_validate_yolo_model_valid(self):
        """yolo_model con valore consentito è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"yolo_model": "yolov8x.pt"})
        assert valid is True

    def test_validate_yolo_model_invalid(self):
        """yolo_model arbitrario è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"yolo_model": "custom.pt"})
        assert valid is False

    def test_validate_sam3_text_prompt_with_special_chars(self):
        """sam3_text_prompt con caratteri non permessi è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"sam3_text_prompt": "person; DROP TABLE"})
        assert valid is False

    def test_validate_sam3_text_prompt_valid(self):
        """sam3_text_prompt valido viene accettato."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"sam3_text_prompt": "person"})
        assert valid is True

    def test_validate_inference_scales_valid_list(self):
        """inference_scales lista valida viene accettata."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"inference_scales": [1.0, 1.5, 2.0]})
        assert valid is True

    def test_validate_inference_scales_out_of_range(self):
        """inference_scales con valore < 0.5 è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"inference_scales": [0.1]})
        assert valid is False

    def test_validate_quality_clahe_grid_valid_tuple(self):
        """quality_clahe_grid con lista [8, 8] è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"quality_clahe_grid": [8, 8]})
        assert valid is True

    def test_validate_quality_clahe_grid_wrong_length(self):
        """quality_clahe_grid con lunghezza != 2 è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"quality_clahe_grid": [8, 8, 8]})
        assert valid is False

    def test_validate_tta_augmentations_valid(self):
        """tta_augmentations con flip_h è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"tta_augmentations": ["flip_h"]})
        assert valid is True

    def test_validate_tta_augmentations_invalid_value(self):
        """tta_augmentations con valore sconosciuto è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"tta_augmentations": ["flip_v"]})
        assert valid is False

    def test_validate_empty_config_is_valid(self):
        """Config vuota è valida (tutti i default)."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({})
        assert valid is True
        assert msg == ""

    def test_validate_nms_iou_threshold_boundary(self):
        """nms_iou_threshold deve essere strettamente > 0 e < 1."""
        from person_anonymizer.web.config_validator import validate_config_params

        assert validate_config_params({"nms_iou_threshold": 0.0})[0] is False
        assert validate_config_params({"nms_iou_threshold": 1.0})[0] is False
        assert validate_config_params({"nms_iou_threshold": 0.5})[0] is True

    def test_validate_ghost_frames_boundary(self):
        """ghost_frames 0 e 120 sono validi; -1 e 121 no."""
        from person_anonymizer.web.config_validator import validate_config_params

        assert validate_config_params({"ghost_frames": 0})[0] is True
        assert validate_config_params({"ghost_frames": 120})[0] is True
        assert validate_config_params({"ghost_frames": -1})[0] is False
        assert validate_config_params({"ghost_frames": 121})[0] is False

    def test_validate_sliding_window_grid_range(self):
        """sliding_window_grid 1-10 valido, 0 e 11 no."""
        from person_anonymizer.web.config_validator import validate_config_params

        assert validate_config_params({"sliding_window_grid": 1})[0] is True
        assert validate_config_params({"sliding_window_grid": 10})[0] is True
        assert validate_config_params({"sliding_window_grid": 0})[0] is False
        assert validate_config_params({"sliding_window_grid": 11})[0] is False

    def test_validate_sam3_model_path_traversal_blocked(self):
        """sam3_model con path traversal (basename != value) è invalido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, msg = validate_config_params({"sam3_model": "../other/model.pt"})
        assert valid is False

    def test_validate_sam3_model_valid_basename(self):
        """sam3_model con solo basename .pt è valido."""
        from person_anonymizer.web.config_validator import validate_config_params

        valid, _ = validate_config_params({"sam3_model": "sam3_hq.pt"})
        assert valid is True

    def test_allowed_fields_contains_expected_keys(self):
        """_ALLOWED_FIELDS contiene i campi minimi necessari per il frontend."""
        from person_anonymizer.web.config_validator import _ALLOWED_FIELDS

        required = {
            "operation_mode",
            "anonymization_method",
            "detection_confidence",
            "enable_tracking",
            "yolo_model",
        }
        missing = required - _ALLOWED_FIELDS
        assert missing == set(), f"Campi mancanti da _ALLOWED_FIELDS: {missing}"

    def test_bool_fields_contains_expected_flags(self):
        """_BOOL_FIELDS contiene i flag booleani principali."""
        from person_anonymizer.web.config_validator import _BOOL_FIELDS

        expected = {
            "enable_tracking",
            "enable_temporal_smoothing",
            "enable_debug_video",
        }
        missing = expected - _BOOL_FIELDS
        assert missing == set(), f"Flag mancanti da _BOOL_FIELDS: {missing}"


# ---------------------------------------------------------------------------
# TestAppCleanupOldJobs — app.py _cleanup_old_jobs
# ---------------------------------------------------------------------------


class TestCleanupOldJobs:
    """Verifica la logica di cleanup dei job scaduti."""

    def test_cleanup_removes_old_job_directories(self, tmp_path):
        """_cleanup_old_jobs rimuove directory più vecchie di max_age_seconds."""
        # Arrange
        import time
        from person_anonymizer.web.app import _cleanup_old_jobs

        # Crea una directory "vecchia" settando mtime nel passato
        old_dir = tmp_path / "old_job"
        old_dir.mkdir()
        old_time = time.time() - 7200  # 2 ore fa
        import os

        os.utime(old_dir, (old_time, old_time))

        # Patch UPLOAD_DIR e OUTPUT_DIR
        with (
            patch("person_anonymizer.web.app.UPLOAD_DIR", tmp_path),
            patch("person_anonymizer.web.app.OUTPUT_DIR", tmp_path / "outputs"),
        ):
            # Act
            _cleanup_old_jobs(max_age_seconds=3600)

        # Assert
        assert not old_dir.exists()

    def test_cleanup_keeps_recent_job_directories(self, tmp_path):
        """_cleanup_old_jobs non rimuove directory recenti."""
        # Arrange
        from person_anonymizer.web.app import _cleanup_old_jobs

        recent_dir = tmp_path / "recent_job"
        recent_dir.mkdir()
        # mtime è recente (creato adesso)

        with (
            patch("person_anonymizer.web.app.UPLOAD_DIR", tmp_path),
            patch("person_anonymizer.web.app.OUTPUT_DIR", tmp_path / "outputs"),
        ):
            # Act
            _cleanup_old_jobs(max_age_seconds=3600)

        # Assert
        assert recent_dir.exists()


# ---------------------------------------------------------------------------
# TestAnnotationValidationExtended — routes_review._validate_annotation_frame
# ---------------------------------------------------------------------------


class TestAnnotationValidationExtended:
    """Casi aggiuntivi per _validate_annotation_frame."""

    def test_out_of_range_coordinates_returns_false(self):
        """Coordinate oltre ±10000 sono invalide."""
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Arrange
        payload = {"auto": [[[0, 0], [10, 0], [99999, 99999]]], "manual": []}
        # Act
        valid, msg = _validate_annotation_frame(payload)
        # Assert
        assert valid is False
        assert "range" in msg

    def test_empty_auto_and_manual_is_valid(self):
        """Payload con auto=[] e manual=[] è valido."""
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        valid, msg = _validate_annotation_frame({"auto": [], "manual": []})
        assert valid is True
        assert msg == ""

    def test_polygon_not_list_returns_false(self):
        """Poligono che non è una lista restituisce errore."""
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        payload = {"auto": ["not_a_polygon"], "manual": []}
        valid, msg = _validate_annotation_frame(payload)
        assert valid is False

    def test_manual_key_validated_independently(self):
        """La validazione copre anche i poligoni in 'manual'."""
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        payload = {"auto": [], "manual": [[[0, 0], [1, 1]]]}  # solo 2 punti
        valid, msg = _validate_annotation_frame(payload)
        assert valid is False
        assert "almeno 3" in msg
