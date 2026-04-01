"""
Test per la validazione dei parametri di configurazione dalla web UI.

Verifica che validate_config_params e _build_config rifiutino input
pericolosi, fuori range o di tipo errato — la difesa principale contro
DoS e comportamenti imprevisti causati da input client malevoli.

Nessuna dipendenza da cv2, ultralytics o YOLO.
"""

import pytest

from web.pipeline_runner import validate_config_params, _build_config


# ============================================================
# validate_config_params — input malevoli
# ============================================================


class TestValidateConfigRejectsAttacks:
    """Verifica che la validazione blocchi input pericolosi da un client malevolo."""

    def test_rejects_extreme_refinement_passes(self):
        # Un client che invia max_refinement_passes=9999999 causa DoS
        # via iterazioni infinite nella pipeline
        valid, msg = validate_config_params({"max_refinement_passes": 9999999})

        assert valid is False
        assert "max_refinement_passes" in msg

    def test_rejects_yolo_model_path_traversal(self):
        # Un client che invia yolo_model="/etc/passwd" tenta di
        # far caricare un file arbitrario a YOLO
        valid, msg = validate_config_params({"yolo_model": "/etc/passwd"})

        assert valid is False
        assert "yolo_model" in msg

    def test_rejects_yolo_model_arbitrary_name(self):
        # Solo i modelli conosciuti devono essere accettati
        valid, msg = validate_config_params({"yolo_model": "malicious_model.pt"})

        assert valid is False

    def test_rejects_string_as_confidence(self):
        # detection_confidence deve essere un numero, non una stringa
        valid, msg = validate_config_params({"detection_confidence": "../../etc/passwd"})

        assert valid is False
        assert "detection_confidence" in msg

    def test_rejects_negative_sliding_window_grid(self):
        # grid negativa causerebbe divisione per zero in get_window_patches
        valid, msg = validate_config_params({"sliding_window_grid": -1})

        assert valid is False

    def test_rejects_zero_sliding_window_grid(self):
        # grid=0 causerebbe divisione per zero
        valid, msg = validate_config_params({"sliding_window_grid": 0})

        assert valid is False

    def test_rejects_confidence_above_one(self):
        # confidence > 1.0 non ha significato per YOLO
        valid, msg = validate_config_params({"detection_confidence": 1.5})

        assert valid is False

    def test_rejects_confidence_below_zero(self):
        valid, msg = validate_config_params({"detection_confidence": -0.5})

        assert valid is False

    def test_rejects_non_bool_for_boolean_field(self):
        # Un intero dove serve un booleano potrebbe causare bug sottili
        valid, msg = validate_config_params({"enable_tracking": 1})

        assert valid is False
        assert "enable_tracking" in msg

    def test_rejects_string_for_boolean_field(self):
        valid, msg = validate_config_params({"enable_post_render_check": "true"})

        assert valid is False


# ============================================================
# validate_config_params — input validi
# ============================================================


class TestValidateConfigAcceptsValid:
    """Verifica che input legittimi passino la validazione."""

    def test_accepts_empty_config(self):
        # Nessun override → tutti i default di PipelineConfig
        valid, msg = validate_config_params({})

        assert valid is True
        assert msg == ""

    def test_accepts_valid_confidence(self):
        valid, msg = validate_config_params({"detection_confidence": 0.35})

        assert valid is True

    def test_accepts_known_yolo_model(self):
        valid, msg = validate_config_params({"yolo_model": "yolov8n.pt"})

        assert valid is True

    def test_accepts_boolean_true(self):
        valid, msg = validate_config_params({"enable_tracking": True})

        assert valid is True

    def test_accepts_valid_complete_config(self):
        # Un set completo di parametri validi come li invierebbe il frontend
        config = {
            "operation_mode": "auto",
            "anonymization_method": "pixelation",
            "anonymization_intensity": 15,
            "detection_confidence": 0.35,
            "nms_iou_threshold": 0.55,
            "yolo_model": "yolov8x.pt",
            "enable_tracking": True,
            "enable_sliding_window": True,
            "max_refinement_passes": 3,
            "sliding_window_grid": 3,
        }
        valid, msg = validate_config_params(config)

        assert valid is True

    def test_accepts_boundary_values(self):
        # Valori ai limiti del range ammesso
        config = {
            "detection_confidence": 0.01,  # minimo
            "anonymization_intensity": 1,   # minimo
            "max_refinement_passes": 10,    # massimo
            "sliding_window_grid": 10,      # massimo
        }
        valid, msg = validate_config_params(config)

        assert valid is True


# ============================================================
# _build_config — integrazione validazione
# ============================================================


class TestBuildConfigIntegration:
    """Verifica che _build_config lanci ValueError per input invalidi
    e restituisca PipelineConfig per input validi."""

    def test_raises_on_invalid_confidence(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            _build_config({"detection_confidence": "not_a_number"})

    def test_raises_on_dos_refinement_passes(self):
        with pytest.raises(ValueError, match="max_refinement_passes"):
            _build_config({"max_refinement_passes": 999999})

    def test_raises_on_path_traversal_model(self):
        with pytest.raises(ValueError, match="yolo_model"):
            _build_config({"yolo_model": "../../../etc/shadow"})

    def test_returns_config_for_valid_input(self):
        config = _build_config({"detection_confidence": 0.5, "enable_tracking": True})

        assert config.detection_confidence == pytest.approx(0.5)
        assert config.enable_tracking is True

    def test_returns_defaults_for_empty_input(self):
        config = _build_config({})

        # Verifica che sia un PipelineConfig con i default
        from config import PipelineConfig
        default = PipelineConfig()
        assert config.detection_confidence == default.detection_confidence
        assert config.operation_mode == default.operation_mode
