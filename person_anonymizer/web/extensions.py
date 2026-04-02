"""Estensioni Flask condivise tra app e blueprint."""

import re

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per minute"],
)


def validate_job_id(job_id: str | None) -> bool:
    """Verifica che job_id sia un hex di 12 caratteri minuscoli."""
    if not job_id or len(job_id) != 12:
        return False
    return bool(re.match(r"^[a-f0-9]{12}$", job_id))
