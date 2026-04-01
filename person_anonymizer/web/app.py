"""
Flask web app per Person Anonymizer.
Serve la GUI e gestisce upload, pipeline, SSE progress e download.
"""

import os
import re
import sys
import uuid
import json
import shutil
import threading
import time as _time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

# Aggiungi parent dir al path per importare person_anonymizer
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from config import SUPPORTED_EXTENSIONS
from web.sse_manager import SSEManager
from web.pipeline_runner import PipelineRunner

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB max upload
app.config.setdefault("RATELIMIT_ENABLED", True)

limiter = Limiter(app=app, key_func=get_remote_address, storage_uri="memory://", default_limits=[])

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _cleanup_old_jobs(max_age_seconds=3600):
    """Rimuove job directory più vecchie di max_age_seconds."""
    now = _time.time()
    for base_dir in (UPLOAD_DIR, OUTPUT_DIR):
        if not base_dir.exists():
            continue
        for job_dir in base_dir.iterdir():
            if job_dir.is_dir():
                try:
                    age = now - job_dir.stat().st_mtime
                    if age > max_age_seconds:
                        shutil.rmtree(job_dir, ignore_errors=True)
                except OSError:
                    pass


def _start_cleanup_thread(interval=600):
    """Avvia thread daemon per pulizia periodica."""

    def _loop():
        while True:
            _cleanup_old_jobs()
            _time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="cleanup")
    t.start()


_start_cleanup_thread()

sse_manager = SSEManager()
pipeline_runner = PipelineRunner(sse_manager, OUTPUT_DIR)


# ---------- Error handlers ----------


@app.errorhandler(500)
def internal_error(e):
    app.logger.exception("Internal server error")
    return jsonify({"error": "Errore interno del server"}), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File troppo grande", "max_mb": 2048}), 413


# ---------- Helper sicurezza ----------


def validate_job_id(job_id: str | None) -> bool:
    """Verifica che job_id sia un hex di 12 caratteri minuscoli."""
    if not job_id or len(job_id) != 12:
        return False
    return bool(re.match(r"^[a-f0-9]{12}$", job_id))


@app.before_request
def add_request_id():
    import uuid as _uuid

    request.request_id = _uuid.uuid4().hex[:16]


@app.before_request
def csrf_check():
    """Verifica CSRF per richieste mutation via header X-Requested-With."""
    if request.method in ("POST", "PUT", "DELETE"):
        # Le richieste multipart (upload file) e JSON (API) da fetch
        # devono includere X-Requested-With che i browser non inviano cross-origin
        if request.endpoint and request.endpoint not in ("static",):
            if not request.headers.get("X-Requested-With"):
                origin = request.headers.get("Origin", "")
                host = request.headers.get("Host", "")
                # Permetti richieste same-origin (no Origin header) o con Origin che matcha Host
                if origin and not origin.endswith(f"://{host}"):
                    return jsonify({"error": "CSRF check failed"}), 403


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'"
    )
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["X-Request-ID"] = getattr(request, "request_id", "unknown")
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
    return response


# ---------- Pagina principale ----------


@app.route("/")
def index():
    return render_template("index.html")


# ---------- Upload video ----------


@app.route("/api/upload", methods=["POST"])
@limiter.limit("10 per minute")
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

    return jsonify({"job_id": job_id, "filename": safe_name, "size_mb": round(size_mb, 2)})


# ---------- Upload JSON annotazioni ----------


@app.route("/api/upload-json", methods=["POST"])
@limiter.limit("20 per minute")
def upload_json():
    if "json_file" not in request.files:
        return jsonify({"error": "Nessun file JSON inviato"}), 400

    f = request.files["json_file"]
    safe_name = secure_filename(f.filename or "")
    if not safe_name:
        return jsonify({"error": "Nome file non valido"}), 400
    if not safe_name.endswith(".json"):
        return jsonify({"error": "File deve essere .json"}), 400

    job_id = request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id mancante"}), 400
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Job non trovato"}), 404

    dest = job_dir / safe_name
    f.save(str(dest))

    # Valida contenuto JSON
    try:
        content = dest.read_text(encoding="utf-8")
        if len(content) > 100 * 1024 * 1024:  # max 100 MB
            dest.unlink(missing_ok=True)
            return jsonify({"error": "File JSON troppo grande (max 100 MB)"}), 400
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            dest.unlink(missing_ok=True)
            return jsonify({"error": "Il JSON deve essere un oggetto"}), 400
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"JSON non valido: {e}"}), 400

    return jsonify({"filename": safe_name})


# ---------- Avvia pipeline ----------


@app.route("/api/start", methods=["POST"])
@limiter.limit("5 per minute")
def start_pipeline():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Payload JSON mancante"}), 400

    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id mancante"}), 400
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    video_filename = data.get("video_filename")
    if not video_filename:
        return jsonify({"error": "video_filename mancante"}), 400
    safe_video = secure_filename(video_filename)
    if not safe_video:
        return jsonify({"error": "video_filename non valido"}), 400

    resolved = (UPLOAD_DIR / job_id / safe_video).resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        return jsonify({"error": "Path non autorizzato"}), 403
    if not resolved.exists():
        return jsonify({"error": "Video non trovato"}), 404
    video_path = str(resolved)

    review_json = None
    review_json_filename = data.get("review_json_filename")
    if review_json_filename:
        safe_json = secure_filename(review_json_filename)
        if not safe_json:
            return jsonify({"error": "review_json_filename non valido"}), 400
        resolved_json = (UPLOAD_DIR / job_id / safe_json).resolve()
        if not str(resolved_json).startswith(str(UPLOAD_DIR.resolve())):
            return jsonify({"error": "Path review_json non autorizzato"}), 403
        if not resolved_json.exists():
            return jsonify({"error": "File JSON non trovato"}), 404
        review_json = str(resolved_json)

    config = data.get("config", {})

    ok, msg = pipeline_runner.start(job_id, video_path, config, review_json)
    if not ok:
        return jsonify({"error": msg}), 409

    return jsonify({"status": "started", "job_id": job_id})


# ---------- SSE Progress stream ----------


@app.route("/api/progress")
@limiter.limit("10 per minute")
def progress_stream():
    job_id = request.args.get("job_id")
    if not validate_job_id(job_id):
        return jsonify({"error": "job_id non valido"}), 400

    def generate():
        import queue as q_module
        import time as _time

        max_duration = 7200  # 2 ore max
        start = _time.monotonic()

        try:
            q = sse_manager.subscribe(job_id)
        except RuntimeError:
            yield f'event: error\ndata: {{"message": "Troppi client connessi"}}\n\n'
            return
        try:
            while True:
                try:
                    event = q.get(timeout=60)
                except q_module.Empty:
                    yield ": heartbeat\n\n"
                    if _time.monotonic() - start > max_duration:
                        yield 'event: timeout\ndata: {"message": "Connessione SSE scaduta dopo 2h"}\n\n'
                        break
                    continue
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
@limiter.limit("10 per minute")
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
@limiter.limit("60 per minute")
def status():
    return jsonify(pipeline_runner.get_status())


# ---------- Review manuale via web ----------


@app.route("/api/review/status")
@limiter.limit("60 per minute")
def review_status():
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"active": False})
    meta = rs.get_metadata()
    meta["active"] = True
    return jsonify(meta)


@app.route("/api/review/frame/<int:frame_idx>")
@limiter.limit("120 per minute")
def review_frame(frame_idx):
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    meta = rs.get_metadata()
    if not (0 <= frame_idx < meta["total_frames"]):
        return jsonify({"error": "frame_idx fuori range"}), 400
    max_w = min(request.args.get("max_width", 1280, type=int), 1920)
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


def _validate_annotation_frame(data: dict) -> tuple[bool, str]:
    """Valida la struttura di una annotazione frame."""
    if not isinstance(data, dict):
        return False, "Il payload deve essere un dizionario"
    for key in ("auto", "manual"):
        polys = data.get(key, [])
        if not isinstance(polys, list):
            return False, f"'{key}' deve essere una lista"
        for poly in polys:
            if not isinstance(poly, list) or len(poly) < 3:
                return False, f"Ogni poligono in '{key}' deve avere almeno 3 punti"
            for pt in poly:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    return False, "Ogni punto deve essere [x, y]"
                x, y = pt
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    return False, "Coordinate devono essere numeriche"
                if not (-10000 <= x <= 10000 and -10000 <= y <= 10000):
                    return False, "Coordinate fuori range ammesso"
    return True, ""


@app.route("/api/review/annotations/<int:frame_idx>", methods=["PUT"])
@limiter.limit("60 per minute")
def review_update_annotations(frame_idx):
    rs = pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    meta = rs.get_metadata()
    if not (0 <= frame_idx < meta["total_frames"]):
        return jsonify({"error": "frame_idx fuori range"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "Payload JSON mancante"}), 400
    valid, msg = _validate_annotation_frame(data)
    if not valid:
        return jsonify({"error": msg}), 422
    rs.update_annotations(frame_idx, data)
    return jsonify({"ok": True})


@app.route("/api/review/confirm", methods=["POST"])
@limiter.limit("5 per minute")
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
    """Restituisce i valori di default dei parametri configurabili."""
    from config import PipelineConfig
    from dataclasses import asdict
    from web.pipeline_runner import _ALLOWED_FIELDS

    cfg = PipelineConfig()
    defaults = {}
    for k, v in asdict(cfg).items():
        if k in _ALLOWED_FIELDS:
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
    import warnings

    warnings.warn(
        "Server di sviluppo Werkzeug in uso. Per produzione usa: "
        "gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 'person_anonymizer.web.app:app'",
        stacklevel=1,
    )
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    print(f"\n  Person Anonymizer Web GUI")
    print(f"  Apri http://{host}:{port} nel browser\n")
    app.run(host=host, port=port, debug=False, threaded=True)
