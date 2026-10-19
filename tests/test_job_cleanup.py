"""Tests for web.job_cleanup."""

import os

from person_anonymizer.web.job_cleanup import cleanup_old_jobs

NOW = 1_800_000_000.0
MAX_AGE = 3600


def _make_dir(base, name, age_s):
    path = base / name
    path.mkdir(parents=True)
    os.utime(path, (NOW - age_s, NOW - age_s))
    return path


def test_dir_older_than_limit_is_removed_and_reported(tmp_path):
    old = _make_dir(tmp_path, "old", MAX_AGE + 1)

    removed = cleanup_old_jobs([tmp_path], max_age_s=MAX_AGE, now=NOW)

    assert removed == [old]
    assert not old.exists()


def test_dir_younger_than_limit_is_kept(tmp_path):
    recent = _make_dir(tmp_path, "recent", MAX_AGE - 1)

    removed = cleanup_old_jobs([tmp_path], max_age_s=MAX_AGE, now=NOW)

    assert removed == []
    assert recent.exists()


def test_plain_file_is_ignored_even_when_old(tmp_path):
    f = tmp_path / "old.txt"
    f.write_text("x")
    os.utime(f, (NOW - 10 * MAX_AGE, NOW - 10 * MAX_AGE))

    removed = cleanup_old_jobs([tmp_path], max_age_s=MAX_AGE, now=NOW)

    assert removed == []
    assert f.exists()


def test_missing_base_dir_is_skipped_and_others_processed(tmp_path):
    other = tmp_path / "outputs"
    old = _make_dir(other, "old", MAX_AGE + 1)

    removed = cleanup_old_jobs([tmp_path / "missing", other], max_age_s=MAX_AGE, now=NOW)

    assert removed == [old]
