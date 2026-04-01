"""
Test per web/app.py — endpoint Flask con test client.

Copre: validazione job_id, security headers, avvio pipeline,
upload video, config defaults e download output.
Nessuna dipendenza da cv2, ultralytics o YOLO.
"""

import io
import pytest

# conftest aggiunge person_anonymizer/ al path
from web.app import app


@pytest.fixture
def client():
    """Flask test client con TESTING abilitato."""
    app.config["TESTING"] = True
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
        payload = {"video_path": "/some/path/video.mp4"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "job_id" in data["error"].lower()

    def test_start_invalid_job_id(self, client):
        # Arrange
        payload = {"job_id": "tooshort", "video_path": "/some/path/video.mp4"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_start_path_traversal_blocked(self, client):
        # Arrange — path traversal verso /etc/passwd
        payload = {"job_id": "aabbccddeeff", "video_path": "../../etc/passwd"}

        # Act
        resp = client.post("/api/start", json=payload)

        # Assert
        assert resp.status_code == 403
        data = resp.get_json()
        assert "error" in data
        assert "autorizzato" in data["error"].lower()


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
