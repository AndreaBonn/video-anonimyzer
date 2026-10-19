"""Funzioni di salvataggio output e caricamento annotazioni."""

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .anonymization import resolve_intensity
from .config import VERSION
from .models import OutputPaths, PipelineContext, PipelineInputError, PipelineResult, VideoMeta

__all__ = [
    "save_outputs",
    "load_annotations_from_json",
    "build_annotations_payload",
    "report_rows",
]

REPORT_FIELDS = [
    "frame_number",
    "persons_detected",
    "avg_confidence",
    "min_confidence",
    "max_confidence",
    "motion_zones",
    "sliding_window_hits",
    "multiscale_hits",
    "post_check_alerts",
]


def load_annotations_from_json(review_json, config):
    """Carica annotazioni da JSON. Restituisce (annotations, mode_override).

    Parameters
    ----------
    review_json : str
        Percorso del file JSON con le annotazioni.
    config : PipelineConfig
        Configurazione della pipeline.

    Returns
    -------
    tuple
        (annotations dict, mode_override str)

    Raises
    ------
    PipelineInputError
        Se il file JSON non esiste.
    """
    if not os.path.isfile(review_json):
        raise PipelineInputError(f"File JSON non trovato: {review_json}")

    print(f"\n[FASE 1/5] Caricamento annotazioni da {Path(review_json).name}...")
    try:
        with open(review_json) as f:
            json_data = json.load(f)

        annotations = {}
        for fidx_str, frame_ann in json_data.get("frames", {}).items():
            fidx = int(fidx_str)
            auto_polys = [[tuple(pt) for pt in poly] for poly in frame_ann.get("auto", [])]
            manual_polys = [[tuple(pt) for pt in poly] for poly in frame_ann.get("manual", [])]
            intensities = []
            for poly in auto_polys:
                ys = [pt[1] for pt in poly]
                box_h = max(ys) - min(ys) if ys else 0
                intensities.append(resolve_intensity(config, box_h))
            annotations[fidx] = {
                "auto": auto_polys,
                "manual": manual_polys,
                "intensities": intensities,
            }

    except (ValueError, AttributeError, TypeError) as err:
        raise PipelineInputError(
            f"JSON annotazioni non valido: {Path(review_json).name}: {err}"
        ) from err

    total_polys = sum(
        len(a.get("auto", [])) + len(a.get("manual", [])) for a in annotations.values()
    )
    print(f"\n  Annotazioni caricate: {len(annotations)} frame, {total_polys} poligoni totali")
    return annotations, "manual"


def report_rows(report_data: dict) -> list[dict]:
    """Return the CSV report rows ordered by frame index.

    Parameters
    ----------
    report_data : dict
        Per-frame report entries keyed by frame index.

    Returns
    -------
    list[dict]
        One row per frame, sorted by ascending frame index.
    """
    return [report_data[fidx] for fidx in sorted(report_data)]


def _frame_payload(ann: dict) -> dict:
    """Serialize one frame's annotations; ``intensities`` only when non-empty."""
    frame_data = {
        "auto": [[list(pt) for pt in poly] for poly in ann.get("auto", [])],
        "manual": [[list(pt) for pt in poly] for poly in ann.get("manual", [])],
    }
    if ann.get("intensities"):
        frame_data["intensities"] = ann["intensities"]
    return frame_data


def _pipeline_config_payload(config, method: str) -> dict:
    """Serialize the config values recorded in the annotations JSON."""
    return {
        "yolo_model": config.yolo_model,
        "detection_confidence": config.detection_confidence,
        "nms_iou_threshold": config.nms_iou_threshold,
        "nms_iou_internal": config.nms_iou_internal,
        "inference_scales": config.inference_scales,
        "sliding_window_grid": config.sliding_window_grid,
        "padding": config.person_padding,
        "anonymization_method": method,
        "base_intensity": config.anonymization_intensity,
        "adaptive_reference_height": config.adaptive_reference_height,
        "smoothing_alpha": config.smoothing_alpha,
        "ghost_frames": config.ghost_frames,
    }


def build_annotations_payload(
    *, config, result: PipelineResult, input_path, meta: VideoMeta
) -> dict:
    """Build the annotations JSON document (schema 2.0) as a plain dict.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    result : PipelineResult
        Pipeline results (annotations, mode, refinement and review stats).
    input_path : str
        Path of the source video (only the file name is stored).
    meta : VideoMeta
        Source video metadata.

    Returns
    -------
    dict
        JSON-serializable payload; ``generated`` is the current local time.
    """
    return {
        "schema_version": "2.0",
        "tool_version": VERSION,
        "video": {
            "filename": Path(input_path).name,
            "total_frames": meta.total_frames,
            "fps": meta.fps,
            "resolution": [meta.frame_w, meta.frame_h],
        },
        "pipeline_config": _pipeline_config_payload(config, result.method),
        "refinement": {
            "max_passes": config.max_refinement_passes,
            "actual_passes": result.actual_refinement_passes,
            "annotations_added": result.refinement_annotations_added,
            "overlap_threshold": config.refinement_overlap_threshold,
        },
        "mode": result.mode,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "review_stats": result.review_stats,
        "frames": {str(fidx): _frame_payload(ann) for fidx, ann in result.annotations.items()},
    }


def _has_debug_video(result: PipelineResult, paths: OutputPaths) -> bool:
    """True when debug output is enabled and the temp debug video exists."""
    return result.enable_debug and os.path.exists(paths.temp_debug)


def _write_final_videos(result: PipelineResult, input_path, paths: OutputPaths) -> None:
    """Encode (ffmpeg available) or copy the final video and, if enabled, the debug one."""
    # Imported here so tests can patch the functions on the postprocessing module.
    from .postprocessing import encode_with_audio, encode_without_audio

    if result.ffmpeg_available:
        encode_with_audio(paths.temp_video, input_path, paths.output)
        if _has_debug_video(result, paths):
            encode_without_audio(paths.temp_debug, paths.debug)
    else:
        shutil.copy(paths.temp_video, paths.output)
        if _has_debug_video(result, paths):
            shutil.copy(paths.temp_debug, paths.debug)


def _write_report_csv(report_data: dict, report_path) -> None:
    """Write the per-frame CSV report."""
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows(report_data))


def _write_annotations_json(payload: dict, json_path) -> None:
    """Write the annotations payload as indented JSON."""
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def _remove_temp_files(paths: OutputPaths) -> None:
    """Remove temporary videos; a failure is reported and never raised."""
    for tmp in (paths.temp_video, paths.temp_debug):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError as e:
            print(f"  ATTENZIONE: impossibile rimuovere {tmp}: {e}")


def save_outputs(
    ctx: PipelineContext,
    result: PipelineResult,
    config,
    input_path,
    paths: OutputPaths,
    meta: VideoMeta,
):
    """Encoding H.264, salvataggio CSV/JSON, cleanup temp.

    Parameters
    ----------
    ctx : PipelineContext
        Contesto della pipeline (usato per ctx.normalize).
    result : PipelineResult
        Risultati della pipeline.
    config : PipelineConfig
        Configurazione della pipeline.
    input_path : str
        Percorso del video sorgente.
    paths : OutputPaths
        Percorsi dei file di output.
    meta : VideoMeta
        Metadati del video.
    """
    try:
        _write_final_videos(result, input_path, paths)

        if result.enable_report:
            _write_report_csv(result.report_data, paths.report)

        if result.mode == "manual" or ctx.normalize:
            payload = build_annotations_payload(
                config=config, result=result, input_path=input_path, meta=meta
            )
            _write_annotations_json(payload, paths.json)
    finally:
        _remove_temp_files(paths)
