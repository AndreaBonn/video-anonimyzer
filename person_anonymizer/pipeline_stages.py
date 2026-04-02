"""Fasi della pipeline — re-export per backward compatibility."""

from tqdm import tqdm  # noqa: F401 — riferimento patchato da TqdmCapture

from .stage_detection import run_detection_loop
from .stage_refinement import run_refinement_loop
from .stage_review import run_manual_review_stage

__all__ = ["run_detection_loop", "run_refinement_loop", "run_manual_review_stage"]
