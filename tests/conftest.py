"""Configurazione pytest."""

import os

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-not-for-production")
