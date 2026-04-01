"""Modelli dati e tipi per la pipeline di anonimizzazione."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .preprocessing import MotionDetector
    from .tracking import TemporalSmoother
    from .web.review_state import ReviewState
    from .web.sse_manager import SSEManager


class PipelineError(Exception):
    """Errore generico della pipeline."""


class PipelineInputError(PipelineError):
    """Errore di input (file non trovato, formato non supportato, ecc.)."""


@dataclass
class OutputPaths:
    """Percorsi dei file di output della pipeline."""

    output: str
    temp_video: str
    temp_debug: str
    debug: str
    report: str
    json: str


@dataclass
class VideoMeta:
    """Metadati del video sorgente."""

    fps: float
    frame_w: int
    frame_h: int
    total_frames: int


@dataclass
class PipelineResult:
    """Risultati della pipeline da salvare."""

    annotations: dict
    report_data: dict
    review_stats: dict
    method: str
    mode: str
    enable_debug: bool
    enable_report: bool
    ffmpeg_available: bool
    actual_refinement_passes: int
    refinement_annotations_added: int


@dataclass
class FrameProcessors:
    """Processori inizializzati per il loop di detection."""

    clahe_obj: object  # cv2.CLAHE
    motion_detector: object  # MotionDetector | None
    patches: list
    tracker: object  # BYTETracker | None
    smoother: object  # TemporalSmoother | None
    do_interpolation: bool


@dataclass
class PipelineContext:
    """Contesto tipizzato per la pipeline — sostituisce SimpleNamespace."""

    input: str
    mode: str | None = None
    method: str | None = None
    output: str | None = None
    no_debug: bool = False
    no_report: bool = False
    review: str | None = None
    normalize: bool = False
    stop_event: threading.Event | None = None
    review_state: ReviewState | None = None
    sse_manager: SSEManager | None = None
    job_id: str | None = None


__all__ = [
    "PipelineError",
    "PipelineInputError",
    "OutputPaths",
    "VideoMeta",
    "PipelineResult",
    "FrameProcessors",
    "PipelineContext",
]
