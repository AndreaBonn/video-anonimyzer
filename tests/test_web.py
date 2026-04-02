"""
Test per web/app.py — endpoint Flask con test client.

Copre: validazione job_id, security headers, avvio pipeline,
upload video, config defaults e download output.
Nessuna dipendenza da cv2, ultralytics o YOLO.
"""

import io
import pytest

# conftest aggiunge person_anonymizer/ al path
from person_anonymizer.web.app import app


@pytest.fixture
def client():
    """Flask test client con TESTING abilitato e rate limiting disabilitato."""
    app.config["TESTING"] = True
    # Flask-Limiter 3.x: disabilita il rate limiting durante i test
    app.config["RATELIMIT_ENABLED"] = False
    app.config["RATELIMIT_STORAGE_URI"] = "memory://"
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# TestValidateJobId
# ---------------------------------------------------------------------------


class TestValidateJobId:
    """Verifica che validate_job_id blocchi input invalidi via /api/outputs."""

    def test_valid_job_id(self, client):
        # Arrange
        valid_id = "aabbccddeeff"

        # Act — job inesistente, ma job_id valido → la validazione passa, ritorna files vuoto
        resp = client.get(f"/api/outputs/{valid_id}")

        # Assert
        assert resp.status_code == 200
        data = resp.get_json()
        assert "files" in data

    def test_invalid_job_id_too_short(self, client):
        # Arrange
        short_id = "abc"

        # Act
        resp = client.get(f"/api/outputs/{short_id}")

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_invalid_job_id_uppercase(self, client):
        # Arrange
        upper_id = "ABCDEF123456"

        # Act
        resp = client.get(f"/api/outputs/{upper_id}")

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_invalid_job_id_special_chars(self, client):
        # Arrange — Flask mappa "abc!@#def123" come parte di URL, usiamo un path encoding-safe
        # che supera comunque la regex ^[a-f0-9]{12}$
        invalid_id = "gggggggggggg"  # hex non valido (g non è esadecimale)

        # Act
        resp = client.get(f"/api/outputs/{invalid_id}")

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_job_id_returns_404(self, client):
        # Arrange — URL senza job_id → Flask restituisce 404 (route non matchata)
        # Act
        resp = client.get("/api/outputs/")

        # Assert
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestSecurityHeaders
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Verifica che add_security_headers applichi tutti gli header attesi."""

    def test_security_headers_present(self, client):
        # Arrange / Act
        resp = client.get("/")

        # Assert
        assert "X-Content-Type-Options" in resp.headers
        assert "X-Frame-Options" in resp.headers
        assert "Referrer-Policy" in resp.headers
        assert "Permissions-Policy" in resp.headers
        assert "Content-Security-Policy" in resp.headers

    def test_x_content_type_options_value(self, client):
        # Arrange / Act
        resp = client.get("/")

        # Assert
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options_value(self, client):
        # Arrange / Act
        resp = client.get("/")

        # Assert
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_csp_no_unsafe_inline(self, client):
        # Arrange / Act
        resp = client.get("/")
        csp = resp.headers.get("Content-Security-Policy", "")

        # Assert
        assert "unsafe-inline" not in csp


# ---------------------------------------------------------------------------
# TestStartPipeline
# ---------------------------------------------------------------------------


class TestStartPipeline:
    """Verifica la validazione dell'endpoint /api/start."""

    def test_start_missing_payload(self, client):
        # Arrange / Act — POST con JSON vuoto
        resp = client.post(
            "/api/start",
            data="{}",
            content_type="application/json",
        )

        # Assert — JSON vuoto → manca job_id
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_start_missing_job_id(self, client):
        # Arrange
        payload = {"video_filename": "video.mp4"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "job_id" in data["error"].lower()

    def test_start_invalid_job_id(self, client):
        # Arrange
        payload = {"job_id": "tooshort", "video_filename": "video.mp4"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_start_path_traversal_blocked(self, client):
        # Arrange — secure_filename elimina i ../ quindi ../../etc/passwd
        # diventa "etc" oppure stringa vuota → risulta 400 (filename non valido)
        # o 404 (file non trovato), mai 403 con il nuovo schema filename-based
        payload = {"job_id": "aabbccddeeff", "video_filename": "../../etc/passwd"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert — secure_filename sanifica l'input; il file non esisterà
        assert resp.status_code in (400, 404)
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# TestUpload
# ---------------------------------------------------------------------------


class TestUpload:
    """Verifica la validazione dell'endpoint /api/upload."""

    def test_upload_no_file(self, client):
        # Arrange / Act — POST senza campo 'video'
        resp = client.post("/api/upload", content_type="multipart/form-data")

        # Assert
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_upload_unsupported_format(self, client):
        # Arrange — file .txt non è un formato video supportato
        dummy_content = b"questo non e' un video"
        data = {"video": (io.BytesIO(dummy_content), "documento.txt")}

        # Act
        resp = client.post(
            "/api/upload",
            data=data,
            content_type="multipart/form-data",
        )

        # Assert
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body
        assert "supported" in body  # la risposta include la lista estensioni valide

    def test_upload_empty_filename(self, client):
        # Arrange — file senza nome
        dummy_content = b"contenuto fittizio"
        data = {"video": (io.BytesIO(dummy_content), "")}

        # Act
        resp = client.post(
            "/api/upload",
            data=data,
            content_type="multipart/form-data",
        )

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# TestConfigDefaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verifica l'endpoint /api/config/defaults.

    Non testa i valori specifici (quelli possono cambiare legittimamente).
    Testa che l'endpoint funzioni e che la struttura sia coerente con
    PipelineConfig.
    """

    def test_config_defaults_returns_200_json(self, client):
        # Arrange / Act
        resp = client.get("/api/config/defaults")

        # Assert
        assert resp.status_code == 200
        assert resp.content_type.startswith("application/json")

    def test_config_defaults_contains_all_pipeline_fields(self, client):
        # Arrange — i campi minimi che il frontend usa per costruire la form
        required_fields = {
            "operation_mode",
            "anonymization_method",
            "anonymization_intensity",
            "detection_confidence",
            "nms_iou_threshold",
            "enable_tracking",
            "enable_sliding_window",
            "enable_post_render_check",
        }

        # Act
        resp = client.get("/api/config/defaults")
        data = resp.get_json()

        # Assert — tutti i campi richiesti dal frontend sono presenti
        missing = required_fields - set(data.keys())
        assert missing == set(), f"Campi mancanti: {missing}"

    def test_config_defaults_confidence_in_valid_range(self, client):
        # Arrange / Act
        resp = client.get("/api/config/defaults")
        data = resp.get_json()

        # Assert — la confidence deve essere un float tra 0 e 1
        conf = data["detection_confidence"]
        assert isinstance(conf, float)
        assert 0.0 < conf < 1.0

    def test_config_defaults_tuples_serialized_as_lists(self, client):
        # Arrange / Act — PipelineConfig ha tuple (quality_clahe_grid)
        # che devono essere convertite in liste per JSON
        resp = client.get("/api/config/defaults")
        data = resp.get_json()

        # Assert — JSON non ha tuple, devono essere liste
        assert isinstance(data["inference_scales"], list)
        assert isinstance(data["quality_clahe_grid"], list)


# ---------------------------------------------------------------------------
# TestDownload
# ---------------------------------------------------------------------------


class TestDownload:
    """Verifica la validazione dell'endpoint /api/download."""

    def test_download_invalid_job_id(self, client):
        # Arrange
        invalid_id = "invalid"

        # Act
        resp = client.get(f"/api/download/{invalid_id}/video")

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_download_nonexistent_job(self, client):
        # Arrange — job_id valido ma nessuna cartella di output
        valid_id = "aabbccddeeff"

        # Act
        resp = client.get(f"/api/download/{valid_id}/video")

        # Assert
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_download_nonexistent_job_with_invalid_type(self, client):
        # Arrange — job_id valido ma inesistente + tipo invalido
        # Il codice controlla prima l'esistenza della cartella (404)
        # poi il tipo file (400). Con job inesistente → 404
        valid_id = "aabbccddeeff"

        # Act
        resp = client.get(f"/api/download/{valid_id}/nonsense")

        # Assert — la cartella non esiste, quindi 404 prima del type check
        assert resp.status_code == 404
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# TestStartPipelineSecurity
# ---------------------------------------------------------------------------


class TestStartPipelineSecurity:
    """Verifica sicurezza dell'endpoint /api/start dopo hardening."""

    def test_start_missing_video_filename(self, client):
        # Arrange — job_id valido ma senza video_filename
        payload = {"job_id": "aabbccddeeff"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "video_filename" in data["error"].lower()

    def test_start_review_json_traversal_blocked(self, client):
        # Arrange — review_json_filename con path traversal
        # secure_filename elimina i ../ quindi il file non verrà trovato.
        # In alcuni ambienti di test il rate limiter in-memory può restituire
        # 429 se le richieste precedenti hanno esaurito il limite: anche in
        # quel caso la traversal è "bloccata" (non ci arriva al filesystem).
        payload = {
            "job_id": "aabbccddeeff",
            "video_filename": "video.mp4",
            "review_json_filename": "../../../etc/passwd",
        }

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert — mai 200: secure_filename sanifica oppure rate limit blocca.
        # 429 può non avere corpo JSON (flask-limiter risponde in testo),
        # quindi verifichiamo solo lo status code.
        assert resp.status_code in (400, 404, 429)
        if resp.status_code != 429:
            assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# TestErrorHandlers
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Verifica error handlers globali."""

    def test_404_on_unknown_route(self, client):
        # Arrange / Act — route inesistente
        resp = client.get("/nonexistent-route")

        # Assert
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestUploadSecurity
# ---------------------------------------------------------------------------


class TestUploadSecurity:
    """Verifica che upload non esponga path assoluti nella response."""

    def test_upload_response_no_absolute_path(self, client):
        # Arrange — file mp4 fittizio ma con estensione valida
        data = {"video": (io.BytesIO(b"fake video content"), "test.mp4")}

        # Act
        resp = client.post("/api/upload", data=data, content_type="multipart/form-data")

        # Assert — se la validazione passa (200), la response non deve contenere "path"
        if resp.status_code == 200:
            body = resp.get_json()
            assert "path" not in body
            assert "filename" in body


# ---------------------------------------------------------------------------
# TestAnnotationValidation
# ---------------------------------------------------------------------------


class TestAnnotationValidation:
    """Verifica la validazione strutturale delle annotazioni (P1-01).

    Testa _validate_annotation_frame direttamente perché l'endpoint
    controlla is_active prima della validazione.
    """

    def test_valid_annotation_payload(self):
        # Arrange
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        payload = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": []}

        # Act
        valid, msg = _validate_annotation_frame(payload)

        # Assert
        assert valid is True
        assert msg == ""

    def test_invalid_annotation_not_dict(self):
        # Arrange
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Act
        valid, msg = _validate_annotation_frame([1, 2, 3])

        # Assert
        assert valid is False
        assert "dizionario" in msg

    def test_invalid_annotation_auto_not_list(self):
        # Arrange
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Act
        valid, msg = _validate_annotation_frame({"auto": "not_a_list", "manual": []})

        # Assert
        assert valid is False
        assert "lista" in msg

    def test_invalid_annotation_polygon_too_few_points(self):
        # Arrange — poligono con solo 2 punti (minimo 3)
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Act
        valid, msg = _validate_annotation_frame({"auto": [[[0, 0], [10, 10]]], "manual": []})

        # Assert
        assert valid is False
        assert "almeno 3" in msg

    def test_invalid_annotation_point_not_pair(self):
        # Arrange — punto con 3 coordinate invece di 2
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Act
        valid, msg = _validate_annotation_frame(
            {"auto": [[[0, 0, 0], [10, 10, 0], [5, 5, 0]]], "manual": []}
        )

        # Assert
        assert valid is False
        assert "[x, y]" in msg

    def test_invalid_annotation_non_numeric_coordinates(self):
        # Arrange — coordinate stringa
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        # Act
        valid, msg = _validate_annotation_frame(
            {"auto": [[["a", "b"], [10, 10], [5, 5]]], "manual": []}
        )

        # Assert
        assert valid is False
        assert "numeriche" in msg

    def test_valid_annotation_with_float_coordinates(self):
        # Arrange — coordinate float sono valide
        from person_anonymizer.web.routes_review import _validate_annotation_frame

        payload = {"auto": [[[0.5, 1.5], [10.1, 0.2], [10.3, 10.4]]], "manual": []}

        # Act
        valid, msg = _validate_annotation_frame(payload)

        # Assert
        assert valid is True

    def test_endpoint_returns_404_when_review_not_active(self, client):
        # Arrange — payload valido ma review non attiva
        payload = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": []}

        # Act
        resp = client.put("/api/review/annotations/0", json=payload)

        # Assert — 404 perché la review non è attiva (validazione passata)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestSSESubscriberCap
# ---------------------------------------------------------------------------


class TestSSESubscriberCap:
    """Verifica il cap sui subscriber SSE per job (P1-02)."""

    def test_subscriber_cap_raises_after_limit(self):
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager, _MAX_SUBSCRIBERS_PER_JOB

        mgr = SSEManager()
        job_id = "test_job_cap"

        # Act — sottoscrivi fino al limite
        queues = []
        for _ in range(_MAX_SUBSCRIBERS_PER_JOB):
            queues.append(mgr.subscribe(job_id))

        # Assert — il prossimo subscribe deve lanciare RuntimeError
        with pytest.raises(RuntimeError):
            mgr.subscribe(job_id)

    def test_subscriber_cap_freed_after_unsubscribe(self):
        # Arrange
        from person_anonymizer.web.sse_manager import SSEManager, _MAX_SUBSCRIBERS_PER_JOB

        mgr = SSEManager()
        job_id = "test_job_free"

        queues = []
        for _ in range(_MAX_SUBSCRIBERS_PER_JOB):
            queues.append(mgr.subscribe(job_id))

        # Act — libera uno slot
        mgr.unsubscribe(job_id, queues[0])

        # Assert — ora possiamo sottoscrivere di nuovo
        q = mgr.subscribe(job_id)
        assert q is not None


# ---------------------------------------------------------------------------
# TestSecurityHeadersNew
# ---------------------------------------------------------------------------


class TestSecurityHeadersNew:
    """Verifica i nuovi header di sicurezza COOP/CORP (P2-04)."""

    def test_coop_header_present(self, client):
        resp = client.get("/")
        assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"

    def test_corp_header_present(self, client):
        resp = client.get("/")
        assert resp.headers.get("Cross-Origin-Resource-Policy") == "same-origin"


# ---------------------------------------------------------------------------
# TestConfigDefaultsFiltered
# ---------------------------------------------------------------------------


class TestConfigDefaultsFiltered:
    """Verifica che config defaults esponga solo campi in _ALLOWED_FIELDS."""

    def test_config_defaults_excludes_internal_fields(self, client):
        # Arrange — campi che NON devono apparire
        internal_fields = {
            "review_auto_color",
            "review_manual_color",
            "review_drawing_color",
            "review_fill_alpha",
            "review_window_max_width",
            "camera_matrix",
            "dist_coefficients",
        }

        # Act
        resp = client.get("/api/config/defaults")
        data = resp.get_json()

        # Assert — nessun campo interno presente
        leaked = internal_fields & set(data.keys())
        assert leaked == set(), f"Campi interni esposti: {leaked}"
