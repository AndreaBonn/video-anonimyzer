"""
Flask web app per Person Anonymizer.
Serve la GUI e gestisce upload, pipeline, SSE progress e download.
"""

import os
import re
import sys
import uuid
import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from werkzeug.utils import secure_filename

# Aggiungi parent dir al path per importare person_anonymizer
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from web.sse_manager import SSEManager
from web.pipeline_runner import PipelineRunner

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB max upload

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}

sse_manager = SSEManager()
pipeline_runner = PipelineRunner(sse_manager, OUTPUT_DIR)


# ---------- Helper sicurezza ----------


def validate_job_id(job_id: str) -> bool:
    """Verifica che job_id sia un hex di 12 caratteri minuscoli."""
    if not job_id or len(job_id) != 12:
        return False
    return bool(re.match(r"^[a-f0-9]{12}$", job_id))


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'"
    )
    return response


# ---------- Pagina principale ----------


@app.route("/")
def index():
    return render_template("index.html")


# ---------- Upload video ----------


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Nessun file video inviato"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Nome file vuoto"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": f"Formato non supportato: {ext}",
                    "supported": sorted(SUPPORTED_EXTENSIONS),
                }
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    safe_name = secure_filename(f.filename)
    if not safe_name:
        return jsonify({"error": "Nome file non valido"}), 400
    dest = job_dir / safe_name
    f.save(str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)

    return jsonify(
        {"job_id": job_id, "filename": safe_name, "size_mb": round(size_mb, 2), "path": str(dest)}
    )


# ---------- Upload JSON annotazioni ----------


@app.route("/api/upload-json", methods=["POST"])
def upload_json():
    if "json_file" not in request.files:
        return jsonify({"error": "Nessun file JSON inviato"}), 400

    f = request.files["json_file"]
    if not f.filename or not f.filename.endswith(".json"):
        return jsonify({"error": "File deve essere .json"}), 400

    job_id = request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id mancante"}), 400
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Job non trovato"}), 404

    safe_name = secure_filename(f.filename)
    if not safe_name:
        return jsonify({"error": "Nome file non valido"}), 400
    dest = job_dir / safe_name
    f.save(str(dest))

    return jsonify({"json_path": str(dest), "filename": safe_name})


# ---------- Avvia pipeline ----------


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Payload JSON mancante"}), 400

    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id mancante"}), 400
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "video_path mancante"}), 400

    resolved = Path(video_path).resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        return jsonify({"error": "Path non autorizzato"}), 403

    if not resolved.exists():
        return jsonify({"error": "Video non trovato"}), 404

    config = data.get("config", {})
    review_json = data.get("review_json")

    ok, msg = pipeline_runner.start(job_id, video_path, config, review_json)
    if not ok:
        return jsonify({"error": msg}), 409

    return jsonify({"status": "started", "job_id": job_id})


# ---------- SSE Progress stream ----------


@app.route("/api/progress")
def progress_stream():
    job_id = request.args.get("job_id")
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    def generate():
        q = sse_manager.subscribe(job_id)
        try:
            while True:
                event = q.get()
                if event is None:
                    break
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        finally:
            sse_manager.unsubscribe(job_id, q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- Stop pipeline ----------


@app.route("/api/stop", methods=["POST"])
def stop_pipeline():
    data = request.get_json() or {}
    job_id = data.get("job_id")
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400
    ok = pipeline_runner.stop(job_id)
    if ok:
        return jsonify({"status": "stopping"})
    return jsonify({"error": "Nessuna pipeline attiva per questo job"}), 404


# ---------- Stato ----------


@app.route("/api/status")
def status():
    return jsonify(pipeline_runner.get_status())


# ---------- Review manuale via web ----------


@app.route("/api/review/status")
def review_status():
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"active": False})
    meta = rs.get_metadata()
    meta["active"] = True
    return jsonify(meta)


@app.route("/api/review/frame/<int:frame_idx>")
def review_frame(frame_idx):
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    max_w = request.args.get("max_width", 1280, type=int)
    jpeg_bytes, scale = rs.get_frame_jpeg(frame_idx, max_width=max_w)
    if jpeg_bytes is None:
        return jsonify({"error": "Frame non trovato"}), 404
    return Response(
        jpeg_bytes,
        mimetype="image/jpeg",
        headers={"X-Scale-Factor": str(scale)},
    )


@app.route("/api/review/annotations")
def review_annotations():
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    annotations = rs.get_annotations()
    # Converti chiavi intere in stringhe per JSON
    out = {}
    for fidx, fdata in annotations.items():
        out[str(fidx)] = fdata
    return jsonify(out)


@app.route("/api/review/annotations/<int:frame_idx>", methods=["PUT"])
def review_update_annotations(frame_idx):
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    data = request.get_json()
    if not data:
        return jsonify({"error": "Payload JSON mancante"}), 400
    rs.update_annotations(frame_idx, data)
    return jsonify({"ok": True})


@app.route("/api/review/confirm", methods=["POST"])
def review_confirm():
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    annotations = rs.get_annotations()
    rs.complete(annotations)
    return jsonify({"ok": True})


# ---------- Config defaults ----------


@app.route("/api/config/defaults")
def config_defaults():
    """Restituisce i valori di default di tutti i parametri configurabili."""
    from config import PipelineConfig
    from dataclasses import asdict

    cfg = PipelineConfig()
    # Converti tuple in liste per la serializzazione JSON
    defaults = {}
    for k, v in asdict(cfg).items():
        defaults[k] = list(v) if isinstance(v, tuple) else v
    return jsonify(defaults)


# ---------- Download output ----------


@app.route("/api/download/<job_id>/<file_type>")
def download_file(job_id, file_type):
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400
    job_out = OUTPUT_DIR / job_id
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


# ---------- Lista file output di un job ----------


@app.route("/api/outputs/<job_id>")
def list_outputs(job_id):
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400
    job_out = OUTPUT_DIR / job_id
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


# ---------- Main ----------

if __name__ == "__main__":
    print(f"\n  Person Anonymizer Web GUI")
    print(f"  Apri http://127.0.0.1:5000 nel browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
