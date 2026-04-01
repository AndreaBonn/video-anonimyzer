#!/usr/bin/env python3
"""
Person Anonymizer Tool v7.1
Pipeline principale di anonimizzazione. Importa i componenti dai moduli.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config import VERSION, SUPPORTED_EXTENSIONS, PipelineConfig
from preprocessing import (
    build_undistortion_maps,
    undistort_frame,
    enhance_frame,
    MotionDetector,
    interpolate_frames,
    should_interpolate,
)
from detection import (
    get_window_patches,
    run_sliding_window,
    run_multiscale_inference,
    apply_nms,
    compute_iou_boxes,
    run_full_detection,
    detect_and_rescale,
)
from tracking import create_tracker, update_tracker, TemporalSmoother
from anonymization import (
    compute_adaptive_intensity,
    obscure_polygon,
    draw_debug_polygons,
    box_to_polygon,
    polygon_to_bbox,
)
from rendering import render_video, _compute_review_stats
from postprocessing import (
    encode_with_audio,
    encode_without_audio,
    run_post_render_check,
    filter_artifact_detections,
    normalize_annotations,
)


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


# ============================================================
# SOTTO-FUNZIONI DI run_pipeline
# ============================================================


def _load_annotations_from_json(review_json, config):
    """Carica annotazioni da JSON. Restituisce (annotations, mode_override)."""
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
            if config.enable_adaptive_intensity:
                intensities.append(
                    compute_adaptive_intensity(
                        box_h, config.anonymization_intensity, config.adaptive_reference_height
                    )
                )
            else:
                intensities.append(config.anonymization_intensity)
        annotations[fidx] = {"auto": auto_polys, "manual": manual_polys, "intensities": intensities}

    total_polys = sum(
        len(a.get("auto", [])) + len(a.get("manual", [])) for a in annotations.values()
    )
    print(f"\n  Annotazioni caricate: {len(annotations)} frame, {total_polys} poligoni totali")
    return annotations, "manual"


def _run_detection_loop(
    cap, total_frames, model, config, fisheye_enabled, undist_map1, undist_map2
):
    """Loop di detection frame-per-frame. Restituisce (annotations, report_data, stats)."""
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    do_interpolation = config.enable_subframe_interpolation and should_interpolate(
        fps, config.interpolation_fps_threshold
    )
    clahe_obj = cv2.createCLAHE(
        clipLimit=config.quality_clahe_clip, tileGridSize=config.quality_clahe_grid
    )
    motion_detector = (
        MotionDetector(config.motion_threshold, config.motion_min_area, config.motion_padding)
        if config.enable_motion_detection
        else None
    )
    patches = (
        get_window_patches(
            frame_w, frame_h, config.sliding_window_grid, config.sliding_window_overlap
        )
        if config.enable_sliding_window
        else []
    )
    tracker = create_tracker(fps, config) if config.enable_tracking else None
    smoother = (
        TemporalSmoother(config.smoothing_alpha, config.ghost_frames, config.ghost_expansion)
        if config.enable_temporal_smoothing
        else None
    )

    annotations, report_data = {}, {}
    unique_ids, total_instances, frames_zero_det, all_confs, corrupted = set(), 0, 0, [], []
    prev_interp = None

    print(f"\n[FASE 1/5] Rilevamento automatico...")
    pbar = tqdm(total=total_frames, desc="Elaborazione", unit=" frame")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_idx < total_frames - 1:
                corrupted.append(frame_idx)
                frame_idx += 1
                pbar.update(1)
                continue
            break

        if fisheye_enabled:
            frame = undistort_frame(frame, undist_map1, undist_map2)
        enhanced = enhance_frame(frame, clahe_obj, config.quality_darkness_threshold)
        motion_regions = motion_detector.get_motion_regions(enhanced) if motion_detector else None

        all_boxes = []
        if do_interpolation and prev_interp is not None:
            n_interp = max(1, int(config.interpolation_fps_threshold / fps) - 1)
            for vf in interpolate_frames(prev_interp, enhanced, n_interp):
                vf_boxes, _, _ = run_full_detection(
                    model,
                    vf,
                    config.detection_confidence,
                    frame_w,
                    frame_h,
                    motion_regions,
                    patches,
                    config,
                )
                all_boxes.extend(vf_boxes)
        prev_interp = enhanced.copy()

        real_boxes, sw_hits, ms_hits = run_full_detection(
            model,
            enhanced,
            config.detection_confidence,
            frame_w,
            frame_h,
            motion_regions,
            patches,
            config,
        )
        all_boxes.extend(real_boxes)
        nms_boxes = apply_nms(all_boxes, config.nms_iou_threshold)

        tracked = (
            update_tracker(tracker, nms_boxes, (frame_h, frame_w, 3))
            if (config.enable_tracking and tracker)
            else [(i, b[0], b[1], b[2], b[3], b[4]) for i, b in enumerate(nms_boxes)]
        )

        frame_polygons, frame_intensities, active_ids = [], [], set()
        for tid, x1, y1, x2, y2, conf in tracked:
            active_ids.add(tid)
            unique_ids.add(tid)
            if config.enable_temporal_smoothing and smoother:
                x1, y1, x2, y2 = smoother.smooth(tid, x1, y1, x2, y2)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
            box_h = y2 - y1
            intensity = (
                compute_adaptive_intensity(
                    box_h, config.anonymization_intensity, config.adaptive_reference_height
                )
                if config.enable_adaptive_intensity
                else config.anonymization_intensity
            )
            frame_polygons.append(
                box_to_polygon(
                    x1, y1, x2, y2, config.person_padding, frame_w, frame_h, config=config
                )
            )
            frame_intensities.append(intensity)
            all_confs.append(conf)

        if smoother:
            smoother.clear_stale(active_ids)
            for gtid, gx1, gy1, gx2, gy2 in smoother.get_ghost_boxes():
                gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(frame_w, gx2), min(frame_h, gy2)
                ghost_h = gy2 - gy1
                if ghost_h <= 0:
                    continue
                g_int = (
                    compute_adaptive_intensity(
                        ghost_h, config.anonymization_intensity, config.adaptive_reference_height
                    )
                    if config.enable_adaptive_intensity
                    else config.anonymization_intensity
                )
                frame_polygons.append(
                    box_to_polygon(
                        gx1, gy1, gx2, gy2, config.person_padding, frame_w, frame_h, config=config
                    )
                )
                frame_intensities.append(g_int)

        annotations[frame_idx] = {
            "auto": frame_polygons,
            "manual": [],
            "intensities": frame_intensities,
        }
        n_det = len(frame_polygons)
        total_instances += n_det
        if n_det == 0:
            frames_zero_det += 1
        confs = [t[5] for t in tracked]
        report_data[frame_idx] = {
            "frame_number": frame_idx,
            "persons_detected": n_det,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "min_confidence": float(min(confs)) if confs else 0.0,
            "max_confidence": float(max(confs)) if confs else 0.0,
            "motion_zones": len(motion_regions) if motion_regions else 0,
            "sliding_window_hits": sw_hits,
            "multiscale_hits": ms_hits,
            "post_check_alerts": 0,
        }
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    zero_pct = (frames_zero_det / total_frames * 100) if total_frames > 0 else 0
    print(f"\n  Persone tracciate (ID unici):   {len(unique_ids)}")
    print(f"  Istanze totali rilevate:        {total_instances:,}")
    print(f"  Frame con 0 rilevamenti:        {frames_zero_det}  ({zero_pct:.1f}%)")
    print(f"  Confidenza media:               {avg_conf:.2f}")
    if corrupted:
        print(
            f"  ATTENZIONE: {len(corrupted)} frame corrotti (saltati): {corrupted[:10]}{'...' if len(corrupted) > 10 else ''}"
        )

    return annotations, report_data, {"unique_ids": unique_ids, "total_instances": total_instances}


def _run_refinement_loop(
    input_path,
    annotations,
    model,
    config,
    fps,
    frame_w,
    frame_h,
    method,
    fisheye_enabled,
    undist_map1,
    undist_map2,
    report_data,
    temp_video_path,
):
    """Loop di auto-refinement. Restituisce (annotations, actual_passes, annotations_added)."""
    if not config.enable_post_render_check:
        print(f"\n[FASE 2/5] Auto refinement — saltato (verifica disabilitata)")
        return annotations, 0, 0

    actual_passes, annotations_added = 0, 0
    for pass_num in range(1, config.max_refinement_passes + 1):
        actual_passes = pass_num
        pass_label = f"pass {pass_num}/{config.max_refinement_passes}"

        print(f"\n[FASE 2/5] Auto refinement — rendering ({pass_label})...")
        render_video(
            input_path,
            temp_video_path,
            annotations,
            fps,
            frame_w,
            frame_h,
            method,
            fisheye_enabled,
            undist_map1,
            undist_map2,
            config,
            debug_path=None,
            desc=f"Rendering ({pass_label})",
        )

        print(f"\n  Verifica post-rendering ({pass_label})...")
        alert_frames = run_post_render_check(
            temp_video_path, model, config.post_render_check_confidence, report_data, config
        )

        if not alert_frames:
            print(f"\n  Nessun rilevamento residuo — rendering OK.")
            break

        genuine_alerts, n_artifacts, n_genuine = filter_artifact_detections(
            alert_frames, annotations, config.refinement_overlap_threshold
        )
        print(f"\n  Rilevamenti post-render: {n_artifacts + n_genuine}")
        print(f"  Artefatti filtrati (IoU >= {config.refinement_overlap_threshold}): {n_artifacts}")
        print(f"  Residui genuini: {n_genuine}")

        if not genuine_alerts:
            print(f"  Tutti i rilevamenti sono artefatti della pixelazione — rendering OK.")
            break

        if pass_num == config.max_refinement_passes:
            print(f"\n  Raggiunto limite di {config.max_refinement_passes} pass.")
            print(f"  Residui genuini rimasti in {len(genuine_alerts)} frame:")
            for fidx, boxes in genuine_alerts:
                print(f"    Frame {fidx}: {len(boxes)} persona/e")
            print(
                "  -> Verranno mostrati nella revisione manuale."
                if method == "manual"
                else "  -> Rieseguire con --mode manual per correzione."
            )
            break

        added_this_pass = 0
        for fidx, boxes in genuine_alerts:
            if fidx not in annotations:
                annotations[fidx] = {"auto": [], "manual": [], "intensities": []}
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                annotations[fidx]["auto"].append(
                    box_to_polygon(
                        x1,
                        y1,
                        x2,
                        y2,
                        padding=config.person_padding,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        config=config,
                    )
                )
                if config.enable_adaptive_intensity:
                    annotations[fidx]["intensities"].append(
                        compute_adaptive_intensity(
                            y2 - y1,
                            config.anonymization_intensity,
                            config.adaptive_reference_height,
                        )
                    )
                added_this_pass += 1

        annotations_added += added_this_pass
        print(f"\n  Aggiunte {added_this_pass} annotazioni — ri-rendering in corso...")

    return annotations, actual_passes, annotations_added


def _run_manual_review(
    args,
    input_path,
    annotations,
    config,
    total_frames,
    fps,
    frame_w,
    frame_h,
    fisheye_enabled,
    undist_map1,
    undist_map2,
):
    """Revisione manuale (web o CLI). Restituisce (annotations, review_stats)."""
    review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
    web_review_state = getattr(args, "_review_state", None)

    if web_review_state is not None:
        print(f"\n[FASE 3/5] Revisione manuale — in attesa di conferma dal browser...")
        web_review_state.setup(
            input_path,
            annotations,
            total_frames,
            frame_w,
            frame_h,
            fps,
            fisheye_enabled,
            undist_map1,
            undist_map2,
        )
        sse_mgr = getattr(args, "_sse_manager", None)
        web_job_id = getattr(args, "_job_id", None)
        sse_mgr.emit(
            web_job_id,
            "review_ready",
            {"total_frames": total_frames, "frame_w": frame_w, "frame_h": frame_h, "fps": fps},
        )
        original = {
            fidx: {"auto": list(d.get("auto", [])), "manual": list(d.get("manual", []))}
            for fidx, d in annotations.items()
        }
        annotations = web_review_state.wait_for_completion()
        review_stats = _compute_review_stats(original, annotations, total_frames)
        print(f"\n  Revisione completata:")
        print(f"  Poligoni aggiunti:     {review_stats['added']}")
        print(f"  Poligoni rimossi:      {review_stats['removed']}")
        print(
            f"  Frame modificati:      {review_stats['frames_modified']}  ({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
        )
    else:
        print(f"\n[FASE 3/5] Revisione manuale — apertura interfaccia...")
        print(
            "  -> Usa Spazio per navigare, Click per disegnare, D per eliminare, Q per confermare."
        )
        from manual_reviewer import run_manual_review

        review_config = {
            "auto_color": config.review_auto_color,
            "manual_color": config.review_manual_color,
            "drawing_color": config.review_drawing_color,
            "fill_alpha": config.review_fill_alpha,
            "max_width": config.review_window_max_width,
        }
        annotations, review_stats = run_manual_review(
            input_path, annotations, review_config, fisheye_enabled, undist_map1, undist_map2
        )
        print(f"\n  Revisione completata:")
        print(f"  Frame revisionati:     {review_stats['frames_reviewed']} / {total_frames}")
        print(f"  Poligoni aggiunti:     {review_stats['added']}")
        print(f"  Poligoni rimossi:      {review_stats['removed']}")
        print(
            f"  Frame modificati:      {review_stats['frames_modified']}  ({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
        )

    return annotations, review_stats


def _save_outputs(
    args,
    annotations,
    report_data,
    review_stats,
    config,
    method,
    mode,
    input_path,
    paths: OutputPaths,
    meta: VideoMeta,
    enable_debug,
    enable_report,
    ffmpeg_available,
    actual_refinement_passes,
    refinement_annotations_added,
):
    """Encoding H.264, salvataggio CSV/JSON, cleanup temp."""
    try:
        if ffmpeg_available:
            encode_with_audio(paths.temp_video, input_path, paths.output)
            if enable_debug and os.path.exists(paths.temp_debug):
                encode_without_audio(paths.temp_debug, paths.debug)
        else:
            shutil.copy(paths.temp_video, paths.output)
            if enable_debug and os.path.exists(paths.temp_debug):
                shutil.copy(paths.temp_debug, paths.debug)

        if enable_report:
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
                for fidx in sorted(report_data.keys()):
                    writer.writerow(report_data[fidx])

        if mode == "manual" or args.normalize:
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
                    "anonymization_method": method,
                    "base_intensity": config.anonymization_intensity,
                    "adaptive_reference_height": config.adaptive_reference_height,
                    "smoothing_alpha": config.smoothing_alpha,
                    "ghost_frames": config.ghost_frames,
                },
                "refinement": {
                    "max_passes": config.max_refinement_passes,
                    "actual_passes": actual_refinement_passes,
                    "annotations_added": refinement_annotations_added,
                    "overlap_threshold": config.refinement_overlap_threshold,
                },
                "mode": mode,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "review_stats": review_stats,
                "frames": {},
            }
            for fidx, ann in annotations.items():
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


# ============================================================
# PIPELINE PRINCIPALE
# ============================================================


def run_pipeline(args, config=None):
    """Esegue la pipeline completa di anonimizzazione."""
    if config is None:
        config = PipelineConfig()

    input_path = args.input
    mode = args.mode or config.operation_mode
    method = args.method or config.anonymization_method
    enable_debug = not args.no_debug
    enable_report = not args.no_report
    review_json = args.review

    if args.normalize and not review_json:
        raise PipelineInputError("--normalize richiede --review <json>")

    if not os.path.isfile(input_path):
        raise PipelineInputError(f"File non trovato: {input_path}")

    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise PipelineInputError(
            f"Formato non supportato '{ext}'. Supportati: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        print("ATTENZIONE: ffmpeg non trovato nel PATH.")
        print("  L'audio non verrà preservato nel video di output.")
        print("  Installare ffmpeg: sudo apt install ffmpeg")

    input_stem = Path(input_path).stem
    input_dir = Path(input_path).parent
    if args.output:
        output_path = args.output
        output_dir = Path(output_path).parent
    else:
        output_dir = input_dir
        output_path = str(output_dir / f"{input_stem}_anonymized.mp4")

    temp_video_path = str(output_dir / f"{input_stem}_temp_noaudio.avi")
    temp_debug_path = str(output_dir / f"{input_stem}_temp_debug.avi")
    debug_path = str(output_dir / f"{input_stem}_debug.mp4")
    report_path = str(output_dir / f"{input_stem}_report.csv")
    json_path = str(output_dir / f"{input_stem}_annotations.json")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise PipelineInputError(f"Impossibile aprire il video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        raise PipelineInputError("Impossibile determinare il numero di frame")

    res_label = f"{frame_w}x{frame_h}"
    if frame_h >= 2160:
        res_label = "4K"
    elif frame_h >= 1080:
        res_label = "1080p"
    elif frame_h >= 720:
        res_label = "720p"
    elif frame_h >= 480:
        res_label = "480p"

    fisheye_enabled = config.enable_fisheye_correction and not (
        config.camera_matrix is None or config.dist_coefficients is None
    )
    do_interpolation = config.enable_subframe_interpolation and should_interpolate(
        fps, config.interpolation_fps_threshold
    )

    if do_interpolation and fps < 3:
        print(
            "ATTENZIONE: fps molto basso (< 3). L'interpolazione potrebbe produrre artefatti. Considera di disabilitarla."
        )

    method_label = (
        f"{method} adattivo (base: {config.anonymization_intensity}px, ref: {config.adaptive_reference_height}px)"
        if config.enable_adaptive_intensity
        else f"{method} ({config.anonymization_intensity}px)"
    )

    print(f"\nPerson Anonymizer v{VERSION}")
    print("-" * 40)
    print(
        f"Input:          {Path(input_path).name}  ({total_frames} frame, {fps:.0f}fps, {res_label})"
    )
    print(f"Output:         {Path(output_path).name}")
    print(f"Modalità:      {mode}")
    print(f"Metodo:         {method_label}")
    print(f"Modello:        {config.yolo_model}  |  Confidenza: {config.detection_confidence}")
    print(
        f"Scale:          [{', '.join(f'{s}x' for s in config.inference_scales)}] + {', '.join(config.tta_augmentations)}"
    )
    sw_status = (
        f"griglia {config.sliding_window_grid}x{config.sliding_window_grid}, overlap {int(config.sliding_window_overlap * 100)}%"
        if config.enable_sliding_window
        else "disabilitato"
    )
    print(f"Sliding window: {sw_status}")
    print(f"Fish-eye:       {'abilitato' if fisheye_enabled else 'disabilitato'}")
    print(
        f"Tracking:       ByteTrack (max_age: {config.track_max_age})"
        if config.enable_tracking
        else "Tracking:       disabilitato"
    )
    print(
        f"Smoothing:      EMA (alpha: {config.smoothing_alpha})"
        if config.enable_temporal_smoothing
        else "Smoothing:      disabilitato"
    )
    interp_status = (
        f"disabilitata (fps >= {config.interpolation_fps_threshold})"
        if not do_interpolation
        else f"abilitata ({fps:.0f}fps < {config.interpolation_fps_threshold}fps)"
    )
    print(f"Interpolazione: {interp_status}")
    print("-" * 40)

    print(f"\nCaricamento modello {config.yolo_model}...")
    model = YOLO(config.yolo_model)

    undist_map1, undist_map2 = None, None
    if fisheye_enabled:
        undist_map1, undist_map2 = build_undistortion_maps(
            config.camera_matrix, config.dist_coefficients, frame_w, frame_h
        )

    report_data = {}
    review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}

    # ============================================
    # FASE 1 — RILEVAMENTO AUTOMATICO (o caricamento JSON)
    # ============================================
    start_time = time.time()

    if review_json:
        annotations, mode = _load_annotations_from_json(review_json, config)
        if args.normalize:
            print(f"\n  Normalizzazione annotazioni...")
            annotations, (n_before, n_after) = normalize_annotations(annotations, config)
            print(f"  Poligoni prima:  {n_before}")
            print(f"  Rettangoli dopo: {n_after}  (riduzione: {n_before - n_after})")
            mode = "auto"
        cap.release()
    else:
        annotations, report_data, _ = _run_detection_loop(
            cap, total_frames, model, config, fisheye_enabled, undist_map1, undist_map2
        )

    # ============================================
    # FASE 2 — AUTO REFINEMENT LOOP
    # ============================================
    if not review_json:
        annotations, actual_refinement_passes, refinement_annotations_added = _run_refinement_loop(
            input_path,
            annotations,
            model,
            config,
            fps,
            frame_w,
            frame_h,
            method,
            fisheye_enabled,
            undist_map1,
            undist_map2,
            report_data,
            temp_video_path,
        )
    else:
        actual_refinement_passes, refinement_annotations_added = 0, 0

    # ============================================
    # FASE 3 — REVISIONE MANUALE
    # ============================================
    if mode == "manual":
        annotations, review_stats = _run_manual_review(
            args,
            input_path,
            annotations,
            config,
            total_frames,
            fps,
            frame_w,
            frame_h,
            fisheye_enabled,
            undist_map1,
            undist_map2,
        )
    else:
        print(f"\n[FASE 3/5] Revisione manuale — saltata (modalità auto)")

    # ============================================
    # FASE 4 — RENDERING FINALE
    # ============================================
    print(f"\n[FASE 4/5] Rendering finale...")
    render_video(
        input_path,
        temp_video_path,
        annotations,
        fps,
        frame_w,
        frame_h,
        method,
        fisheye_enabled,
        undist_map1,
        undist_map2,
        config,
        debug_path=temp_debug_path if enable_debug else None,
        desc="Rendering finale",
    )

    # ============================================
    # FASE 5 — POST-PROCESSING
    # ============================================
    print(f"\n[FASE 5/5] Post-processing...")
    paths = OutputPaths(
        output=output_path,
        temp_video=temp_video_path,
        temp_debug=temp_debug_path,
        debug=debug_path,
        report=report_path,
        json=json_path,
    )
    meta = VideoMeta(fps=fps, frame_w=frame_w, frame_h=frame_h, total_frames=total_frames)
    _save_outputs(
        args,
        annotations,
        report_data,
        review_stats,
        config,
        method,
        mode,
        input_path,
        paths,
        meta,
        enable_debug,
        enable_report,
        ffmpeg_available,
        actual_refinement_passes,
        refinement_annotations_added,
    )

    total_time = time.time() - start_time
    minutes, seconds = int(total_time // 60), int(total_time % 60)
    print("-" * 40)
    print(f"Completato  —  Tempo totale: {minutes}m {seconds}s")
    print(f"  Output:       {Path(paths.output).name}")
    if enable_debug:
        print(f"  Debug:        {Path(paths.debug).name}")
    if enable_report:
        print(f"  Report:       {Path(paths.report).name}")
    if mode == "manual" or args.normalize:
        print(f"  Annotazioni:  {Path(paths.json).name}")
    print()


# ============================================================
# CLI
# ============================================================


def parse_args():
    """Parser argomenti CLI."""
    parser = argparse.ArgumentParser(
        description=f"Person Anonymizer v{VERSION} — Oscuramento automatico persone in video di sorveglianza",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Percorso del video da elaborare")
    parser.add_argument(
        "-M",
        "--mode",
        choices=["manual", "auto"],
        default=None,
        help="Modalità operativa (default: da config)",
    )
    parser.add_argument("-o", "--output", default=None, help="Percorso file di output")
    parser.add_argument(
        "-m",
        "--method",
        choices=["pixelation", "blur"],
        default=None,
        help="Metodo di oscuramento (default: da config)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disabilita video debug")
    parser.add_argument("--no-report", action="store_true", help="Disabilita CSV report")
    parser.add_argument(
        "--review",
        default=None,
        help="Ricarica annotazioni da JSON esistente, salta la detection e apre solo la revisione manuale",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalizza i poligoni in rettangoli e unifica le aree sovrapposte. Richiede --review.",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    try:
        run_pipeline(args)
    except PipelineInputError as e:
        print(f"\nErrore: {e}")
        sys.exit(1)
    except PipelineError as e:
        print(f"\nErrore pipeline: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente (Ctrl+C).")
        sys.exit(1)


if __name__ == "__main__":
    main()
