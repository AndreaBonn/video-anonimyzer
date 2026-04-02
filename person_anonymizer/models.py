"""Modelli dati e tipi per la pipeline di anonimizzazione."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2

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

    clahe_obj: cv2.CLAHE | None = None
    motion_detector: MotionDetector | None = None
    patches: list = field(default_factory=list)
    tracker: object = None  # BYTETracker (no type stub disponibile)
    smoother: TemporalSmoother | None = None
    do_interpolation: bool = False


@dataclass
class FrameDetectionResult:
    """Risultato dell'elaborazione di un singolo frame."""

    polygons: list
    intensities: list
    tracked: list
    sw_hits: int
    ms_hits: int
    active_ids: set
    prev_interp_frame: object  # np.ndarray | None
    motion_count: int


@dataclass
class FisheyeContext:
    """Contesto di correzione fish-eye."""

    enabled: bool = False
    undist_map1: object = None  # np.ndarray | None
    undist_map2: object = None  # np.ndarray | None

    def undistort(self, frame):
        """Applica undistortion se abilitato. Restituisce il frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR da processare.

        Returns
        -------
        np.ndarray
            Frame con undistortion applicata, oppure il frame originale.
        """
        if self.enabled and self.undist_map1 is not None:
            import cv2

            return cv2.remap(frame, self.undist_map1, self.undist_map2, cv2.INTER_LINEAR)
        return frame


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
    "FrameDetectionResult",
    "FisheyeContext",
    "PipelineContext",
]
