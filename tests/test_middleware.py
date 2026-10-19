"""Behavioral coverage for CSRF protection outside testing mode."""

import pytest
from flask import Flask

from person_anonymizer.web.middleware import register_middleware


@pytest.fixture
def client():
    """Provide a dedicated application with production middleware enabled."""
    app = Flask(__name__)
    app.config["TESTING"] = False
    register_middleware(app)

    @app.route("/x", methods=["GET", "POST", "PUT", "DELETE"])
    def endpoint():
        return {"status": "ok"}

    with app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_csrf_mutation_requires_header(client, method):
    response = client.open(path="/x", method=method)
    assert response.status_code == 403
    assert response.get_json() == {"error": "CSRF check failed"}

    response = client.open(path="/x", method=method, headers={"X-Requested-With": "XMLHttpRequest"})
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_csrf_get_without_header_returns_success(client):
    response = client.get("/x")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
