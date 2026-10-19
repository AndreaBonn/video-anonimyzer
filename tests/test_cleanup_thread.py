"""Guard for the storage cleanup thread suppressed by conftest.py."""

import threading
from unittest.mock import patch

import pytest


def test_app_cleanup_thread_name_matches_the_one_conftest_suppresses():
    # conftest.py stops the thread named "cleanup" from starting at app import,
    # otherwise the suite deletes real job folders under web/outputs. If the
    # thread is renamed, the suppression stops working and this test turns red.
    from person_anonymizer.web import app as web_app

    with patch.object(web_app.threading, "Thread") as thread_cls:
        web_app._start_cleanup_thread()

    assert thread_cls.call_args.kwargs["name"] == "cleanup"
    thread_cls.return_value.start.assert_called_once()
    assert not any(t.name == "cleanup" for t in threading.enumerate())


def test_cleanup_loop_removes_jobs_older_than_one_hour_in_both_storage_dirs():
    from person_anonymizer.web import app as web_app

    class StopLoop(Exception):
        pass

    with patch.object(web_app.threading, "Thread") as thread_cls:
        web_app._start_cleanup_thread(interval=600)
    loop = thread_cls.call_args.kwargs["target"]

    with (
        patch.object(web_app, "cleanup_old_jobs") as cleanup,
        patch.object(web_app.time, "time", return_value=1000.0),
        patch.object(web_app.time, "sleep", side_effect=StopLoop) as sleep,
        pytest.raises(StopLoop),
    ):
        loop()

    cleanup.assert_called_once_with(
        base_dirs=(web_app.UPLOAD_DIR, web_app.OUTPUT_DIR), max_age_s=3600, now=1000.0
    )
    sleep.assert_called_once_with(600)
