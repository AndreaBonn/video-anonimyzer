"""
Flask web app per Person Anonymizer.
Serve la GUI e gestisce upload, pipeline, SSE progress e download.
"""

import os
import uuid
import json
import shutil
import threading
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

from person_anonymizer.config import SUPPORTED_EXTENSIONS
from person_anonymizer.web.sse_manager import SSEManager
from person_anonymizer.web.pipeline_runner import PipelineRunner
from person_anonymizer.web.extensions import limiter, validate_job_id
from person_anonymizer.web.middleware import register_middleware
from person_anonymizer.web.routes_review import review_bp
from person_anonymizer.web.routes_output import output_bp

app = Flask(__name__)
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    import warnings

    warnings.warn(
        "FLASK_SECRET_KEY non impostata — uso chiave temporanea. "
        "Le sessioni saranno invalidate al riavvio. "
        "Imposta FLASK_SECRET_KEY per produzione.",
        stacklevel=1,
    )
    _secret_key = os.urandom(32)
app.secret_key = _secret_key
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB max upload
app.config.setdefault("RATELIMIT_ENABLED", True)

# Inizializza limiter sull'app
limiter.init_app(app)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Firme magic bytes per i formati video supportati
_VIDEO_SIGNATURES: dict[bytes, set[str]] = {
    b"\x00\x00\x00": {".mp4", ".m4v", ".mov"},  # ftyp container (ISO BMFF)
    b"\x1a\x45\xdf": {".mkv", ".webm"},  # EBML header (Matroska)
    b"RIFF": {".avi"},  # RIFF container
}


def _cleanup_old_jobs(max_age_seconds=3600):
    """Rimuove job directory più vecchie di max_age_seconds."""
    now = time.time()
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
            try:
                _cleanup_old_jobs()
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="cleanup")
    t.start()


_start_cleanup_thread()

sse_manager = SSEManager()
pipeline_runner = PipelineRunner(sse_manager, OUTPUT_DIR)

# Espone risorse condivise sull'app per i blueprint
app.pipeline_runner = pipeline_runner
app.sse_manager = sse_manager
app.output_dir = OUTPUT_DIR

# Registra middleware e blueprint
register_middleware(app)
app.register_blueprint(review_bp)
app.register_blueprint(output_bp)


# ---------- Error handlers ----------


@app.errorhandler(500)
def internal_error(e):
    app.logger.exception("Internal server error")
    return jsonify({"error": "Errore interno del server"}), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File troppo grande", "max_mb": 2048}), 413


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

    # Validazione magic bytes del file video
    header = f.read(12)
    f.seek(0)
    signature_valid = False
    for sig, valid_exts in _VIDEO_SIGNATURES.items():
        if header.startswith(sig) and ext in valid_exts:
            signature_valid = True
            break
    if not signature_valid:
        return jsonify({"error": "Il contenuto del file non corrisponde all'estensione"}), 400

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

    # Leggi e valida in memoria prima di salvare (evita TOCTOU)
    content = f.read()
    if len(content) > 100 * 1024 * 1024:  # max 100 MB
        return jsonify({"error": "File JSON troppo grande (max 100 MB)"}), 400
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return jsonify({"error": "Il JSON deve essere un oggetto"}), 400
    except (json.JSONDecodeError, UnicodeDecodeError):
        return jsonify({"error": "Il file non contiene JSON valido"}), 400

    dest = job_dir / safe_name
    dest.write_bytes(content)
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
        import time

        max_duration = 7200  # 2 ore max
        start = time.monotonic()

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
                    if time.monotonic() - start > max_duration:
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
