"""
Configurazione centralizzata per Person Anonymizer.

Contiene la dataclass PipelineConfig con tutti i parametri della pipeline,
e i valori di default. Sostituisce le 42 variabili globali a livello modulo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VERSION = "7.1"

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
    inference_scales: list = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    tta_augmentations: list = field(default_factory=lambda: ["flip_h"])

    # --- Miglioramento qualità ---
    quality_clahe_clip: float = 2.0
    quality_clahe_grid: tuple = (8, 8)
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
    review_auto_color: tuple = (0, 255, 0)
    review_manual_color: tuple = (0, 120, 255)
    review_drawing_color: tuple = (255, 255, 0)
    review_fill_alpha: float = 0.35
    review_window_max_width: int = 1280

    # --- Output e debug ---
    enable_debug_video: bool = True
    enable_confidence_report: bool = True
