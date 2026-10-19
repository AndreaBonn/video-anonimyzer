"""Configurazione pytest."""

import os
from importlib import import_module
from threading import Thread
from unittest.mock import patch

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-not-for-production")

_start_thread = Thread.start


def _start_without_storage_cleanup(thread: Thread) -> None:
    # App import starts cleanup before fixtures can isolate storage directories.
    if thread.name != "cleanup":
        _start_thread(thread)


with patch.object(Thread, "start", new=_start_without_storage_cleanup):
    import_module("person_anonymizer.web.app")
