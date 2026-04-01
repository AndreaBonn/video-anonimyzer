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

__all__ = ["save_outputs", "load_annotations_from_json"]


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
        annotations[fidx] = {"auto": auto_polys, "manual": manual_polys, "intensities": intensities}

    total_polys = sum(
        len(a.get("auto", [])) + len(a.get("manual", [])) for a in annotations.values()
    )
    print(f"\n  Annotazioni caricate: {len(annotations)} frame, {total_polys} poligoni totali")
    return annotations, "manual"


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
    from .postprocessing import encode_with_audio, encode_without_audio

    try:
        if result.ffmpeg_available:
            encode_with_audio(paths.temp_video, input_path, paths.output)
            if result.enable_debug and os.path.exists(paths.temp_debug):
                encode_without_audio(paths.temp_debug, paths.debug)
        else:
            shutil.copy(paths.temp_video, paths.output)
            if result.enable_debug and os.path.exists(paths.temp_debug):
                shutil.copy(paths.temp_debug, paths.debug)

        if result.enable_report:
            with open(paths.report, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_number",
                        "persons_detected",
                        "avg_confidence",
                        "min_confidence",
                        "max_confidence",
                        "motion_zones",
                        "sliding_window_hits",
                        "multiscale_hits",
                        "post_check_alerts",
                    ],
                )
                writer.writeheader()
                for fidx in sorted(result.report_data.keys()):
                    writer.writerow(result.report_data[fidx])

        if result.mode == "manual" or ctx.normalize:
            json_annotations = {
                "schema_version": "2.0",
                "tool_version": VERSION,
                "video": {
                    "filename": Path(input_path).name,
                    "total_frames": meta.total_frames,
                    "fps": meta.fps,
                    "resolution": [meta.frame_w, meta.frame_h],
                },
                "pipeline_config": {
                    "yolo_model": config.yolo_model,
                    "detection_confidence": config.detection_confidence,
                    "nms_iou_threshold": config.nms_iou_threshold,
                    "nms_iou_internal": config.nms_iou_internal,
                    "inference_scales": config.inference_scales,
                    "sliding_window_grid": config.sliding_window_grid,
                    "padding": config.person_padding,
                    "anonymization_method": result.method,
                    "base_intensity": config.anonymization_intensity,
                    "adaptive_reference_height": config.adaptive_reference_height,
                    "smoothing_alpha": config.smoothing_alpha,
                    "ghost_frames": config.ghost_frames,
                },
                "refinement": {
                    "max_passes": config.max_refinement_passes,
                    "actual_passes": result.actual_refinement_passes,
                    "annotations_added": result.refinement_annotations_added,
                    "overlap_threshold": config.refinement_overlap_threshold,
                },
                "mode": result.mode,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "review_stats": result.review_stats,
                "frames": {},
            }
            for fidx, ann in result.annotations.items():
                frame_data = {
                    "auto": [[list(pt) for pt in poly] for poly in ann.get("auto", [])],
                    "manual": [[list(pt) for pt in poly] for poly in ann.get("manual", [])],
                }
                if ann.get("intensities"):
                    frame_data["intensities"] = ann["intensities"]
                json_annotations["frames"][str(fidx)] = frame_data
            with open(paths.json, "w") as f:
                json.dump(json_annotations, f, indent=2)

    finally:
        for tmp in (paths.temp_video, paths.temp_debug):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError as e:
                print(f"  ATTENZIONE: impossibile rimuovere {tmp}: {e}")
