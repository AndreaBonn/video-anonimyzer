"""Removal of expired job directories (no thread is started here)."""

import shutil
from collections.abc import Iterable
from pathlib import Path


def cleanup_old_jobs(base_dirs: Iterable[Path], max_age_s: float, now: float) -> list[Path]:
    """Remove job directories whose mtime is older than ``max_age_s``.

    Parameters
    ----------
    base_dirs : Iterable[Path]
        Directories whose direct subdirectories are job folders.
    max_age_s : float
        Age in seconds above which a job folder is removed.
    now : float
        Current epoch time, injected for determinism.

    Returns
    -------
    list[Path]
        Job directories selected for removal. Plain files are ignored.
    """
    removed: list[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for job_dir in base_dir.iterdir():
            if _is_expired(job_dir, max_age_s=max_age_s, now=now):
                shutil.rmtree(job_dir, ignore_errors=True)
                removed.append(job_dir)
    return removed


def _is_expired(job_dir: Path, max_age_s: float, now: float) -> bool:
    try:
        return job_dir.is_dir() and now - job_dir.stat().st_mtime > max_age_s
    except OSError:
        return False
