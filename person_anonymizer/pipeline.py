"""Pipeline principale di anonimizzazione — orchestratore."""

import os
import shutil
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import SUPPORTED_EXTENSIONS, VERSION, PipelineConfig
from .models import (
    FisheyeContext,
    OutputPaths,
    PipelineContext,
    PipelineInputError,
    PipelineResult,
    VideoMeta,
)
from .output import load_annotations_from_json, save_outputs
from .pipeline_stages import run_detection_loop, run_manual_review_stage, run_refinement_loop
from .postprocessing import normalize_annotations
from .preprocessing import build_undistortion_maps, should_interpolate
from .rendering import render_video

__all__ = ["run_pipeline"]


def run_pipeline(ctx: PipelineContext, config=None):
    """Esegue la pipeline completa di anonimizzazione.

    Parameters
    ----------
    ctx : PipelineContext
        Contesto della pipeline con input, mode, method, output, flag e stato web.
    config : PipelineConfig | None
        Configurazione della pipeline. Se None, usa i default.

    Raises
    ------
    PipelineInputError
        Se il file di input non esiste o non è supportato.
    PipelineError
        Per errori generici durante l'elaborazione.
    """
    if config is None:
        config = PipelineConfig()

    input_path = ctx.input
    mode = ctx.mode or config.operation_mode
    method = ctx.method or config.anonymization_method
    enable_debug = not ctx.no_debug
    enable_report = not ctx.no_report
    review_json = ctx.review
    stop_event = ctx.stop_event

    if ctx.normalize and not review_json:
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
    if ctx.output:
        output_path = ctx.output
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

    do_interpolation = config.enable_subframe_interpolation and should_interpolate(
        fps, config.interpolation_fps_threshold
    )

    if do_interpolation and fps < 3:
        print(
            "ATTENZIONE: fps molto basso (< 3). L'interpolazione potrebbe produrre artefatti. "
            "Considera di disabilitarla."
        )

    method_label = (
        f"{method} adattivo (base: {config.anonymization_intensity}px, "
        f"ref: {config.adaptive_reference_height}px)"
        if config.enable_adaptive_intensity
        else f"{method} ({config.anonymization_intensity}px)"
    )

    print(f"\nPerson Anonymizer v{VERSION}")
    print("-" * 40)
    print(
        f"Input:          {Path(input_path).name}  "
        f"({total_frames} frame, {fps:.0f}fps, {res_label})"
    )
    print(f"Output:         {Path(output_path).name}")
    print(f"Modalità:      {mode}")
    print(f"Metodo:         {method_label}")
    print(f"Modello:        {config.yolo_model}  |  Confidenza: {config.detection_confidence}")
    print(
        f"Scale:          [{', '.join(f'{s}x' for s in config.inference_scales)}] + "
        f"{', '.join(config.tta_augmentations)}"
    )
    sw_status = (
        f"griglia {config.sliding_window_grid}x{config.sliding_window_grid}, "
        f"overlap {int(config.sliding_window_overlap * 100)}%"
        if config.enable_sliding_window
        else "disabilitato"
    )
    print(f"Sliding window: {sw_status}")
    fe_label = "abilitato" if config.enable_fisheye_correction else "disabilitato"
    print(f"Fish-eye:       {fe_label}")
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

    fisheye = FisheyeContext()
    if (
        config.enable_fisheye_correction
        and config.camera_matrix is not None
        and config.dist_coefficients is not None
    ):
        undist_map1, undist_map2 = build_undistortion_maps(
            config.camera_matrix, config.dist_coefficients, frame_w, frame_h
        )
        fisheye = FisheyeContext(enabled=True, undist_map1=undist_map1, undist_map2=undist_map2)

    report_data = {}
    review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}

    # ============================================
    # FASE 1 — RILEVAMENTO AUTOMATICO (o caricamento JSON)
    # ============================================
    start_time = time.time()

    if review_json:
        annotations, mode = load_annotations_from_json(review_json, config)
        if ctx.normalize:
            print("\n  Normalizzazione annotazioni...")
            annotations, (n_before, n_after) = normalize_annotations(annotations, config)
            print(f"  Poligoni prima:  {n_before}")
            print(f"  Rettangoli dopo: {n_after}  (riduzione: {n_before - n_after})")
            mode = "auto"
        cap.release()
    else:
        annotations, report_data, _ = run_detection_loop(
            cap, total_frames, model, config, fisheye, stop_event
        )

    # ============================================
    # FASE 2 — AUTO REFINEMENT LOOP
    # ============================================
    if not review_json:
        annotations, actual_refinement_passes, refinement_annotations_added = run_refinement_loop(
            input_path,
            annotations,
            model,
            config,
            fps,
            frame_w,
            frame_h,
            method,
            fisheye,
            report_data,
            temp_video_path,
            stop_event,
        )
    else:
        actual_refinement_passes, refinement_annotations_added = 0, 0

    # ============================================
    # FASE 3 — REVISIONE MANUALE
    # ============================================
    if mode == "manual":
        annotations, review_stats = run_manual_review_stage(
            ctx,
            input_path,
            annotations,
            config,
            total_frames,
            fps,
            frame_w,
            frame_h,
            fisheye,
        )
    else:
        print("\n[FASE 3/5] Revisione manuale — saltata (modalità auto)")

    # ============================================
    # FASE 4 — RENDERING FINALE
    # ============================================
    print("\n[FASE 4/5] Rendering finale...")
    render_video(
        input_path,
        temp_video_path,
        annotations,
        fps,
        frame_w,
        frame_h,
        method,
        fisheye,
        config,
        debug_path=temp_debug_path if enable_debug else None,
        desc="Rendering finale",
        stop_event=stop_event,
    )

    # ============================================
    # FASE 5 — POST-PROCESSING
    # ============================================
    print("\n[FASE 5/5] Post-processing...")
    paths = OutputPaths(
        output=output_path,
        temp_video=temp_video_path,
        temp_debug=temp_debug_path,
        debug=debug_path,
        report=report_path,
        json=json_path,
    )
    meta = VideoMeta(fps=fps, frame_w=frame_w, frame_h=frame_h, total_frames=total_frames)
    result = PipelineResult(
        annotations=annotations,
        report_data=report_data,
        review_stats=review_stats,
        method=method,
        mode=mode,
        enable_debug=enable_debug,
        enable_report=enable_report,
        ffmpeg_available=ffmpeg_available,
        actual_refinement_passes=actual_refinement_passes,
        refinement_annotations_added=refinement_annotations_added,
    )
    save_outputs(ctx, result, config, input_path, paths, meta)

    total_time = time.time() - start_time
    minutes, seconds = int(total_time // 60), int(total_time % 60)
    print("-" * 40)
    print(f"Completato  —  Tempo totale: {minutes}m {seconds}s")
    print(f"  Output:       {Path(paths.output).name}")
    if enable_debug:
        print(f"  Debug:        {Path(paths.debug).name}")
    if enable_report:
        print(f"  Report:       {Path(paths.report).name}")
    if mode == "manual" or ctx.normalize:
        print(f"  Annotazioni:  {Path(paths.json).name}")
    print()
