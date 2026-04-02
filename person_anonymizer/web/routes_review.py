"""Route per la revisione manuale delle annotazioni."""

from flask import Blueprint, Response, current_app, jsonify, request

from person_anonymizer.web.extensions import limiter

review_bp = Blueprint("review", __name__)


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


@review_bp.route("/api/review/status")
@limiter.limit("60 per minute")
def review_status():
    rs = current_app.pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"active": False})
    meta = rs.get_metadata()
    meta["active"] = True
    return jsonify(meta)


@review_bp.route("/api/review/frame/<int:frame_idx>")
@limiter.limit("120 per minute")
def review_frame(frame_idx):
    rs = current_app.pipeline_runner.review_state
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


@review_bp.route("/api/review/annotations")
@limiter.limit("60 per minute")
def review_annotations():
    rs = current_app.pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    annotations = rs.get_annotations()
    # Converti chiavi intere in stringhe per JSON
    out = {}
    for fidx, fdata in annotations.items():
        out[str(fidx)] = fdata
    return jsonify(out)


@review_bp.route("/api/review/annotations/<int:frame_idx>", methods=["PUT"])
@limiter.limit("60 per minute")
def review_update_annotations(frame_idx):
    rs = current_app.pipeline_runner.review_state
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


@review_bp.route("/api/review/confirm", methods=["POST"])
@limiter.limit("5 per minute")
def review_confirm():
    rs = current_app.pipeline_runner.review_state
    if not rs.is_active:
        return jsonify({"error": "Nessuna review attiva"}), 404
    annotations = rs.get_annotations()
    rs.complete(annotations)
    return jsonify({"ok": True})
