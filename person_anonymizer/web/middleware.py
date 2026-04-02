"""Middleware di sicurezza per la web app Flask."""

import uuid

from flask import jsonify, request


def register_middleware(app):
    """Registra middleware di sicurezza sull'app Flask."""
    app.before_request(add_request_id)
    app.before_request(csrf_check)
    app.after_request(add_security_headers)


def add_request_id():
    request.request_id = uuid.uuid4().hex[:16]


def csrf_check():
    """Verifica CSRF per richieste mutation via header X-Requested-With."""
    from flask import current_app

    if current_app.config.get("TESTING"):
        return None
    if request.method in ("POST", "PUT", "DELETE"):
        if request.endpoint and request.endpoint not in ("static",):
            if not request.headers.get("X-Requested-With"):
                return jsonify({"error": "CSRF check failed"}), 403


def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
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
