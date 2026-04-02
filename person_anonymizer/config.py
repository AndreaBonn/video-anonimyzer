"""
Configurazione centralizzata per Person Anonymizer.

Contiene la dataclass PipelineConfig con tutti i parametri della pipeline,
e i valori di default. Sostituisce le 42 variabili globali a livello modulo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VERSION", "SUPPORTED_EXTENSIONS", "PipelineConfig"]

VERSION = "7.1.0"

SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class PipelineConfig:
    """Configurazione completa della pipeline di anonimizzazione."""

    # --- Modalità operativa ---
    operation_mode: str = "manual"

    # --- Oscuramento ---
    anonymization_method: str = "pixelation"
    anonymization_intensity: int = 10
    person_padding: int = 15
    edge_padding_multiplier: float = 2.5
    edge_threshold: float = 0.05

    # --- Rilevamento ---
    detection_confidence: float = 0.20
    nms_iou_internal: float = 0.45
    nms_iou_threshold: float = 0.55
    yolo_model: str = "yolov8x.pt"

    # --- Correzione fish-eye ---
    enable_fisheye_correction: bool = True
    camera_matrix: np.ndarray | None = None
    dist_coefficients: np.ndarray | None = None

    # --- Frame differencing ---
    enable_motion_detection: bool = False
    motion_threshold: int = 25
    motion_min_area: int = 500
    motion_padding: int = 60

    # --- Sliding window ---
    enable_sliding_window: bool = True
    sliding_window_grid: int = 3
    sliding_window_overlap: float = 0.3

    # --- Multi-scale + TTA ---
    inference_scales: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    tta_augmentations: list[str] = field(default_factory=lambda: ["flip_h"])

    # --- Miglioramento qualità ---
    quality_clahe_clip: float = 2.0
    quality_clahe_grid: tuple[int, int] = (8, 8)
    quality_darkness_threshold: int = 60

    # --- Tracking ---
    enable_tracking: bool = True
    track_max_age: int = 45
    track_match_thresh: float = 0.6

    # --- Temporal smoothing ---
    enable_temporal_smoothing: bool = True
    smoothing_alpha: float = 0.35
    ghost_frames: int = 10
    ghost_expansion: float = 1.15

    # --- Intensità adattiva ---
    enable_adaptive_intensity: bool = True
    adaptive_reference_height: int = 80

    # --- Interpolazione sub-frame ---
    enable_subframe_interpolation: bool = False
    interpolation_fps_threshold: int = 15

    # --- Verifica post-rendering ---
    enable_post_render_check: bool = True
    post_render_check_confidence: float = 0.45
    max_refinement_passes: int = 3
    refinement_overlap_threshold: float = 0.5

    # --- Revisione manuale (colori BGR) ---
    review_auto_color: tuple[int, int, int] = (0, 255, 0)
    review_manual_color: tuple[int, int, int] = (0, 120, 255)
    review_drawing_color: tuple[int, int, int] = (255, 255, 0)
    review_fill_alpha: float = 0.35
    review_window_max_width: int = 1280

    # --- Output e debug ---
    enable_debug_video: bool = True
    enable_confidence_report: bool = True

    def __post_init__(self):
        """Valida i parametri di configurazione."""
        if not (0.01 <= self.detection_confidence <= 0.99):
            raise ValueError(
                "detection_confidence deve essere tra 0.01 e 0.99, "
                f"ricevuto {self.detection_confidence}"
            )
        if not (1 <= self.anonymization_intensity <= 100):
            raise ValueError(
                "anonymization_intensity deve essere tra 1 e 100, "
                f"ricevuto {self.anonymization_intensity}"
            )
        if self.person_padding < 0 or self.person_padding > 200:
            raise ValueError(
                f"person_padding deve essere tra 0 e 200, ricevuto {self.person_padding}"
            )
        if not (0.0 < self.nms_iou_internal < 1.0):
            raise ValueError(
                f"nms_iou_internal deve essere tra 0 e 1 (escl.), ricevuto {self.nms_iou_internal}"
            )
        if not (0.0 < self.nms_iou_threshold < 1.0):
            raise ValueError(
                "nms_iou_threshold deve essere tra 0 e 1 (escl.), "
                f"ricevuto {self.nms_iou_threshold}"
            )
        if self.operation_mode not in ("manual", "auto"):
            raise ValueError(
                f"operation_mode deve essere 'manual' o 'auto', ricevuto '{self.operation_mode}'"
            )
        if self.anonymization_method not in ("pixelation", "blur"):
            raise ValueError(
                "anonymization_method deve essere 'pixelation' o 'blur', "
                f"ricevuto '{self.anonymization_method}'"
            )
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha deve essere tra 0 (escl.) e 1, ricevuto {self.smoothing_alpha}"
            )
        if self.ghost_frames < 0 or self.ghost_frames > 120:
            raise ValueError(f"ghost_frames deve essere tra 0 e 120, ricevuto {self.ghost_frames}")
        if not (1.0 <= self.ghost_expansion <= 2.0):
            raise ValueError(
                f"ghost_expansion deve essere tra 1.0 e 2.0, ricevuto {self.ghost_expansion}"
            )
        if not (1 <= self.max_refinement_passes <= 10):
            raise ValueError(
                "max_refinement_passes deve essere tra 1 e 10, "
                f"ricevuto {self.max_refinement_passes}"
            )
        if self.sliding_window_grid < 1 or self.sliding_window_grid > 10:
            raise ValueError(
                f"sliding_window_grid deve essere tra 1 e 10, ricevuto {self.sliding_window_grid}"
            )
        if not self.inference_scales:
            raise ValueError("inference_scales non può essere vuota")
        if self.adaptive_reference_height < 1:
            raise ValueError(
                "adaptive_reference_height deve essere >= 1, "
                f"ricevuto {self.adaptive_reference_height}"
            )
