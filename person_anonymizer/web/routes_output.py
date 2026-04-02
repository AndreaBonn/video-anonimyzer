"""Route per download output e configurazione."""

from dataclasses import asdict

from flask import Blueprint, jsonify, send_file, current_app

from person_anonymizer.web.extensions import limiter, validate_job_id

output_bp = Blueprint("output", __name__)


@output_bp.route("/api/config/defaults")
@limiter.limit("60 per minute")
def config_defaults():
    """Restituisce i valori di default dei parametri configurabili."""
    from person_anonymizer.config import PipelineConfig
    from person_anonymizer.web.config_validator import _ALLOWED_FIELDS

    cfg = PipelineConfig()
    defaults = {}
    for k, v in asdict(cfg).items():
        if k in _ALLOWED_FIELDS:
            defaults[k] = list(v) if isinstance(v, tuple) else v
    return jsonify(defaults)


@output_bp.route("/api/download/<job_id>/<file_type>")
@limiter.limit("30 per minute")
def download_file(job_id, file_type):
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400
    output_dir = current_app.output_dir
    job_out = output_dir / job_id
    if not job_out.exists():
        return jsonify({"error": "Job output non trovato"}), 404

    type_map = {
        "video": "_anonymized.mp4",
        "debug": "_debug.mp4",
        "report": "_report.csv",
        "annotations": "_annotations.json",
    }

    suffix = type_map.get(file_type)
    if not suffix:
        return jsonify({"error": f"Tipo non valido: {file_type}"}), 400

    for f in job_out.iterdir():
        if f.name.endswith(suffix):
            resp = send_file(str(f), as_attachment=True)
            resp.headers["Cache-Control"] = "no-store"
            return resp

    return jsonify({"error": f"File {file_type} non trovato"}), 404


@output_bp.route("/api/outputs/<job_id>")
@limiter.limit("60 per minute")
def list_outputs(job_id):
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400
    output_dir = current_app.output_dir
    job_out = output_dir / job_id
    if not job_out.exists():
        return jsonify({"files": []})

    files = []
    for f in sorted(job_out.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            file_type = "unknown"
            if f.suffix == ".json":
                file_type = "annotations"
            elif f.suffix == ".csv":
                file_type = "report"
            elif "debug" in f.name:
                file_type = "debug"
            elif "anonymized" in f.name:
                file_type = "video"
            files.append(
                {
                    "name": f.name,
                    "type": file_type,
                    "size_mb": round(size_mb, 2),
                }
            )

    return jsonify({"files": files})
