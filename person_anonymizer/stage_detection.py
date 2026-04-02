"""Fase di detection: init processori, loop frame-per-frame."""

import cv2
import numpy as np
from tqdm import tqdm

from .anonymization import box_to_polygon, resolve_intensity
from .config import PipelineConfig
from .detection import apply_nms, run_full_detection, get_window_patches
from .models import FrameDetectionResult, FrameProcessors
from .preprocessing import MotionDetector, should_interpolate, interpolate_frames, enhance_frame
from .tracking import create_tracker, TemporalSmoother, update_tracker

__all__ = ["run_detection_loop"]


def _init_frame_processors(fps, frame_w, frame_h, config):
    """Inizializza i processori per il loop di detection.

    Parameters
    ----------
    fps : float
        Frame rate del video.
    frame_w : int
        Larghezza frame in pixel.
    frame_h : int
        Altezza frame in pixel.
    config : PipelineConfig
        Configurazione della pipeline.

    Returns
    -------
    FrameProcessors
        Processori inizializzati.
    """
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
    return FrameProcessors(
        clahe_obj=clahe_obj,
        motion_detector=motion_detector,
        patches=patches,
        tracker=tracker,
        smoother=smoother,
        do_interpolation=do_interpolation,
    )


def _process_single_frame(frame, model, config, frame_w, frame_h, proc, prev_interp, fps):
    """Processa un singolo frame: detection, tracking, smoothing, ghost boxes.

    Returns
    -------
    tuple
        (frame_polygons, frame_intensities, tracked, sw_hits, ms_hits, active_ids,
         new_prev_interp, motion_count)
    """
    enhanced = enhance_frame(frame, proc.clahe_obj, config.quality_darkness_threshold)
    motion_regions = (
        proc.motion_detector.get_motion_regions(enhanced) if proc.motion_detector else None
    )
    motion_count = len(motion_regions) if motion_regions else 0

    all_boxes = []
    if proc.do_interpolation and prev_interp is not None:
        n_interp = max(1, int(config.interpolation_fps_threshold / fps) - 1)
        for vf in interpolate_frames(prev_interp, enhanced, n_interp):
            vf_boxes, _, _ = run_full_detection(
                model,
                vf,
                config.detection_confidence,
                frame_w,
                frame_h,
                motion_regions,
                proc.patches,
                config,
            )
            all_boxes.extend(vf_boxes)
    new_prev_interp = enhanced.copy()

    real_boxes, sw_hits, ms_hits = run_full_detection(
        model,
        enhanced,
        config.detection_confidence,
        frame_w,
        frame_h,
        motion_regions,
        proc.patches,
        config,
    )
    all_boxes.extend(real_boxes)
    nms_boxes = apply_nms(all_boxes, config.nms_iou_threshold)

    tracked = (
        update_tracker(proc.tracker, nms_boxes, (frame_h, frame_w, 3))
        if (config.enable_tracking and proc.tracker)
        else [(i, b[0], b[1], b[2], b[3], b[4]) for i, b in enumerate(nms_boxes)]
    )

    frame_polygons, frame_intensities, active_ids = [], [], set()
    for tid, x1, y1, x2, y2, conf in tracked:
        active_ids.add(tid)
        if config.enable_temporal_smoothing and proc.smoother:
            x1, y1, x2, y2 = proc.smoother.smooth(tid, x1, y1, x2, y2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
        box_h = y2 - y1
        intensity = resolve_intensity(config, box_h)
        frame_polygons.append(
            box_to_polygon(x1, y1, x2, y2, config.person_padding, frame_w, frame_h, config=config)
        )
        frame_intensities.append(intensity)

    if proc.smoother:
        proc.smoother.clear_stale(active_ids)
        for gtid, gx1, gy1, gx2, gy2 in proc.smoother.get_ghost_boxes():
            gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(frame_w, gx2), min(frame_h, gy2)
            ghost_h = gy2 - gy1
            if ghost_h <= 0:
                continue
            g_int = resolve_intensity(config, ghost_h)
            frame_polygons.append(
                box_to_polygon(
                    gx1, gy1, gx2, gy2, config.person_padding, frame_w, frame_h, config=config
                )
            )
            frame_intensities.append(g_int)

    return FrameDetectionResult(
        polygons=frame_polygons,
        intensities=frame_intensities,
        tracked=tracked,
        sw_hits=sw_hits,
        ms_hits=ms_hits,
        active_ids=active_ids,
        prev_interp_frame=new_prev_interp,
        motion_count=motion_count,
    )


def run_detection_loop(cap, total_frames, model, config, fisheye, stop_event=None):
    """Loop di detection frame-per-frame.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Capture object del video.
    total_frames : int
        Numero totale di frame.
    model : YOLO
        Modello di detection.
    config : PipelineConfig
        Configurazione della pipeline.
    fisheye : FisheyeContext
        Contesto di correzione fish-eye.
    stop_event : threading.Event | None
        Evento di stop per interruzione asincrona.

    Returns
    -------
    tuple
        (annotations dict, report_data dict, stats dict)
    """
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    proc = _init_frame_processors(fps, frame_w, frame_h, config)

    annotations, report_data = {}, {}
    unique_ids, total_instances, frames_zero_det, all_confs, corrupted = set(), 0, 0, [], []
    prev_interp = None

    print(f"\n[FASE 1/5] Rilevamento automatico...")
    pbar = tqdm(total=total_frames, desc="Elaborazione", unit=" frame")
    frame_idx = 0

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("\n  Pipeline interrotta dall'utente.")
                break

            ret, frame = cap.read()
            if not ret:
                if frame_idx < total_frames - 1:
                    corrupted.append(frame_idx)
                    frame_idx += 1
                    pbar.update(1)
                    continue
                break

            frame = fisheye.undistort(frame)

            result = _process_single_frame(
                frame, model, config, frame_w, frame_h, proc, prev_interp, fps
            )
            prev_interp = result.prev_interp_frame

            for tid, *_ in result.tracked:
                unique_ids.add(tid)
            confs = [t[5] for t in result.tracked]
            all_confs.extend(confs)

            annotations[frame_idx] = {
                "auto": result.polygons,
                "manual": [],
                "intensities": result.intensities,
            }
            n_det = len(result.polygons)
            total_instances += n_det
            if n_det == 0:
                frames_zero_det += 1
            report_data[frame_idx] = {
                "frame_number": frame_idx,
                "persons_detected": n_det,
                "avg_confidence": float(np.mean(confs)) if confs else 0.0,
                "min_confidence": float(min(confs)) if confs else 0.0,
                "max_confidence": float(max(confs)) if confs else 0.0,
                "motion_zones": result.motion_count,
                "sliding_window_hits": result.sw_hits,
                "multiscale_hits": result.ms_hits,
                "post_check_alerts": 0,
            }
            frame_idx += 1
            pbar.update(1)

        pbar.close()
    finally:
        cap.release()

    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    zero_pct = (frames_zero_det / total_frames * 100) if total_frames > 0 else 0
    print(f"\n  Persone tracciate (ID unici):   {len(unique_ids)}")
    print(f"  Istanze totali rilevate:        {total_instances:,}")
    print(f"  Frame con 0 rilevamenti:        {frames_zero_det}  ({zero_pct:.1f}%)")
    print(f"  Confidenza media:               {avg_conf:.2f}")
    if corrupted:
        print(
            f"  ATTENZIONE: {len(corrupted)} frame corrotti (saltati): "
            f"{corrupted[:10]}{'...' if len(corrupted) > 10 else ''}"
        )

    return annotations, report_data, {"unique_ids": unique_ids, "total_instances": total_instances}
