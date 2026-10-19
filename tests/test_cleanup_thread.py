"""Guard for the storage cleanup thread suppressed by conftest.py."""

import threading
from unittest.mock import patch


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
