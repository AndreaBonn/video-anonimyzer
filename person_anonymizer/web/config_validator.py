"""Validatori di configurazione per i parametri della web form."""

import os
import re

__all__ = [
    "_CONFIG_VALIDATORS",
    "_BOOL_FIELDS",
    "_ALLOWED_FIELDS",
    "validate_config_params",
]

_CONFIG_VALIDATORS = {
    "operation_mode": lambda v: isinstance(v, str) and v in ("auto", "manual"),
    "anonymization_method": lambda v: isinstance(v, str) and v in ("pixelation", "blur"),
    "anonymization_intensity": lambda v: isinstance(v, (int, float)) and 1 <= v <= 100,
    "person_padding": lambda v: isinstance(v, (int, float)) and 0 <= v <= 200,
    "detection_confidence": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.99,
    "nms_iou_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "nms_iou_internal": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "detection_backend": lambda v: isinstance(v, str) and v in ("yolo", "yolo+sam3", "sam3"),
    "yolo_model": lambda v: isinstance(v, str) and v in ("yolov8x.pt", "yolov8n.pt"),
    "sam3_model": lambda v: (
        isinstance(v, str) and len(v) > 0 and os.path.basename(v) == v and v.endswith(".pt")
    ),
    "sam3_text_prompt": lambda v: (
        isinstance(v, str) and 0 < len(v) <= 100 and bool(re.match(r"^[a-zA-Z0-9 _-]+$", v))
    ),
    "sam3_mask_simplify_epsilon": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "sam3_min_mask_area": lambda v: isinstance(v, int) and v >= 1,
    "sliding_window_grid": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "sliding_window_overlap": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 0.9,
    "max_refinement_passes": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "smoothing_alpha": lambda v: isinstance(v, (int, float)) and 0.0 < v <= 1.0,
    "ghost_frames": lambda v: isinstance(v, int) and 0 <= v <= 120,
    "ghost_expansion": lambda v: isinstance(v, (int, float)) and 1.0 <= v <= 2.0,
    "track_max_age": lambda v: isinstance(v, int) and 1 <= v <= 300,
    "track_match_thresh": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "adaptive_reference_height": lambda v: isinstance(v, int) and 10 <= v <= 500,
    "post_render_check_confidence": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.99,
    "refinement_overlap_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "edge_padding_multiplier": lambda v: isinstance(v, (int, float)) and 1.0 <= v <= 5.0,
    "edge_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 0.5,
    "motion_threshold": lambda v: isinstance(v, int) and 1 <= v <= 255,
    "motion_min_area": lambda v: isinstance(v, int) and 1 <= v <= 100000,
    "motion_padding": lambda v: isinstance(v, int) and 0 <= v <= 500,
    "quality_clahe_clip": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 10.0,
    "quality_darkness_threshold": lambda v: isinstance(v, int) and 0 <= v <= 255,
    "interpolation_fps_threshold": lambda v: isinstance(v, int) and 1 <= v <= 120,
    "inference_scales": lambda v: isinstance(v, list)
    and all(isinstance(s, (int, float)) and 0.5 <= s <= 5.0 for s in v)
    and 1 <= len(v) <= 10,
    "tta_augmentations": lambda v: isinstance(v, list)
    and all(isinstance(a, str) and a in ("flip_h",) for a in v),
    "quality_clahe_grid": lambda v: isinstance(v, (list, tuple))
    and len(v) == 2
    and all(isinstance(x, int) and 1 <= x <= 32 for x in v),
}

# Campi booleani: accettano solo bool
_BOOL_FIELDS = {
    "enable_fisheye_correction",
    "enable_motion_detection",
    "enable_sliding_window",
    "enable_tracking",
    "enable_temporal_smoothing",
    "enable_adaptive_intensity",
    "enable_subframe_interpolation",
    "enable_post_render_check",
    "enable_debug_video",
    "enable_confidence_report",
}

# Campi ammessi nella costruzione di PipelineConfig dalla web form
_ALLOWED_FIELDS = {
    "operation_mode",
    "anonymization_method",
    "anonymization_intensity",
    "person_padding",
    "detection_confidence",
    "nms_iou_threshold",
    "yolo_model",
    "enable_fisheye_correction",
    "enable_motion_detection",
    "motion_threshold",
    "motion_min_area",
    "motion_padding",
    "enable_sliding_window",
    "sliding_window_grid",
    "sliding_window_overlap",
    "inference_scales",
    "tta_augmentations",
    "quality_clahe_clip",
    "quality_clahe_grid",
    "quality_darkness_threshold",
    "enable_tracking",
    "track_max_age",
    "track_match_thresh",
    "enable_temporal_smoothing",
    "smoothing_alpha",
    "ghost_frames",
    "ghost_expansion",
    "enable_adaptive_intensity",
    "adaptive_reference_height",
    "enable_subframe_interpolation",
    "interpolation_fps_threshold",
    "enable_post_render_check",
    "post_render_check_confidence",
    "max_refinement_passes",
    "refinement_overlap_threshold",
    "enable_debug_video",
    "enable_confidence_report",
    "edge_padding_multiplier",
    "edge_threshold",
    "nms_iou_internal",
    "detection_backend",
    "sam3_model",
    "sam3_text_prompt",
    "sam3_mask_simplify_epsilon",
    "sam3_min_mask_area",
}


def validate_config_params(web_config: dict) -> tuple[bool, str]:
    """
    Valida i parametri di configurazione prima di applicarli.

    Parameters
    ----------
    web_config : dict
        Parametri dalla form web.

    Returns
    -------
    tuple[bool, str]
        (True, "") se valido, (False, messaggio_errore) altrimenti.
    """
    for key, value in web_config.items():
        if key in _BOOL_FIELDS:
            if not isinstance(value, bool):
                return False, f"Parametro '{key}' deve essere booleano"
        elif key in _CONFIG_VALIDATORS:
            if not _CONFIG_VALIDATORS[key](value):
                return False, f"Parametro '{key}' non valido: {value!r}"
    return True, ""
