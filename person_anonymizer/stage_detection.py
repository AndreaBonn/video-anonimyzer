"""Fase di detection: init processori, loop frame-per-frame."""

from dataclasses import dataclass

import cv2
from tqdm import tqdm

from .anonymization import box_to_polygon, resolve_intensity
from .detection import apply_nms, get_window_patches, run_full_detection
from .detection_report import DetectionTotals, report_row
from .models import FrameDetectionResult, FrameProcessors
from .preprocessing import MotionDetector, enhance_frame, interpolate_frames, should_interpolate
from .tracking import TemporalSmoother, create_tracker, update_tracker

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


@dataclass(frozen=True)
class _DetectionSetup:
    """Valori invarianti per l'intero run di detection, condivisi dai passi per frame."""

    model: object
    config: object
    frame_w: int
    frame_h: int
    fps: float
    proc: FrameProcessors


def clamp_box(box, frame_w, frame_h):
    """Limita un box ``(x1, y1, x2, y2)`` ai bordi del frame.

    Returns
    -------
    tuple
        Box con x1, y1 >= 0 e x2 <= frame_w, y2 <= frame_h.
    """
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)


def _box_height(box):
    _, y1, _, y2 = box
    return y2 - y1


def _box_polygon(box, config, frame_w, frame_h):
    """Poligono con padding per un box ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = box
    return box_to_polygon(x1, y1, x2, y2, config.person_padding, frame_w, frame_h, config=config)


def _detect(frame, motion_regions, setup):
    """Detection completa (multi-scala, sliding window) su un frame."""
    return run_full_detection(
        setup.model,
        frame,
        setup.config.detection_confidence,
        setup.frame_w,
        setup.frame_h,
        motion_regions,
        setup.proc.patches,
        setup.config,
    )


def _collect_boxes(enhanced, prev_interp, motion_regions, setup):
    """Box dei frame virtuali interpolati più quelli del frame reale.

    Returns
    -------
    tuple
        (box, sliding_window_hits, multiscale_hits); gli hit sono del solo frame reale.
    """
    boxes = []
    if setup.proc.do_interpolation and prev_interp is not None:
        n_interp = max(1, int(setup.config.interpolation_fps_threshold / setup.fps) - 1)
        for virtual_frame in interpolate_frames(prev_interp, enhanced, n_interp):
            virtual_boxes, _, _ = _detect(virtual_frame, motion_regions, setup)
            boxes.extend(virtual_boxes)
    real_boxes, sw_hits, ms_hits = _detect(enhanced, motion_regions, setup)
    boxes.extend(real_boxes)
    return boxes, sw_hits, ms_hits


def _track(nms_boxes, setup):
    """Assegna ID ai box NMS: una voce ``(tid, x1, y1, x2, y2, conf)`` per box, stesso ordine."""
    if setup.config.enable_tracking and setup.proc.tracker:
        return update_tracker(setup.proc.tracker, nms_boxes, (setup.frame_h, setup.frame_w, 3))
    return [(i, b[0], b[1], b[2], b[3], b[4]) for i, b in enumerate(nms_boxes)]


def _smooth_and_clamp(tracked, setup):
    """Smoothing temporale opzionale e clamp ai bordi.

    Returns
    -------
    tuple
        (box ``(x1, y1, x2, y2)``, insieme degli ID attivi).
    """
    smoother = setup.proc.smoother if setup.config.enable_temporal_smoothing else None
    boxes, active_ids = [], set()
    for tid, x1, y1, x2, y2, _conf in tracked:
        active_ids.add(tid)
        if smoother:
            x1, y1, x2, y2 = smoother.smooth(tid, x1, y1, x2, y2)
        boxes.append(clamp_box((x1, y1, x2, y2), setup.frame_w, setup.frame_h))
    return boxes, active_ids


def _boxes_to_polygons(frame, boxes, setup):
    """Poligoni e intensità per i box: SAM3 in batch se attivo, altrimenti box con padding.

    Il refiner SAM3 restituisce un poligono per ciascun box in ingresso.
    """
    config = setup.config
    refiner = setup.proc.sam3_refiner
    if refiner is not None and boxes:
        polygons = refiner.refine_boxes(frame, boxes)
    else:
        polygons = [_box_polygon(box, config, setup.frame_w, setup.frame_h) for box in boxes]
    intensities = [resolve_intensity(config, _box_height(box)) for box in boxes]
    return polygons, intensities


def _ghost_polygons(ghost_boxes, config, frame_w, frame_h):
    """Poligoni e intensità dei ghost box ``(tid, x1, y1, x2, y2)``.

    I ghost con altezza <= 0 dopo il clamp sono scartati.
    """
    polygons, intensities = [], []
    for _tid, x1, y1, x2, y2 in ghost_boxes:
        box = clamp_box((x1, y1, x2, y2), frame_w, frame_h)
        height = _box_height(box)
        if height <= 0:
            continue
        polygons.append(_box_polygon(box, config, frame_w, frame_h))
        intensities.append(resolve_intensity(config, height))
    return polygons, intensities


def _ghosts(active_ids, setup):
    """Rimuove dallo smoother gli ID non più attivi e restituisce i poligoni dei ghost box."""
    smoother = setup.proc.smoother
    if not smoother:
        return [], []
    smoother.clear_stale(active_ids)
    return _ghost_polygons(smoother.get_ghost_boxes(), setup.config, setup.frame_w, setup.frame_h)


def _process_single_frame(frame, prev_interp, setup):
    """Processa un singolo frame: detection, tracking, smoothing, ghost boxes.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR già corretto dalla distorsione.
    prev_interp : np.ndarray | None
        Frame migliorato precedente, per l'interpolazione sub-frame.
    setup : _DetectionSetup
        Parametri invarianti del run.

    Returns
    -------
    FrameDetectionResult
        Poligoni, intensità e metriche del frame.
    """
    proc = setup.proc
    enhanced = enhance_frame(frame, proc.clahe_obj, setup.config.quality_darkness_threshold)
    regions = proc.motion_detector.get_motion_regions(enhanced) if proc.motion_detector else None
    next_prev_interp = enhanced.copy()

    boxes, sw_hits, ms_hits = _collect_boxes(enhanced, prev_interp, regions, setup)
    nms_boxes = apply_nms(boxes, setup.config.nms_iou_threshold)
    tracked = _track(nms_boxes, setup)
    clamped, active_ids = _smooth_and_clamp(tracked, setup)
    polygons, intensities = _boxes_to_polygons(frame, clamped, setup)
    ghost_polygons, ghost_intensities = _ghosts(active_ids, setup)

    return FrameDetectionResult(
        polygons=polygons + ghost_polygons,
        intensities=intensities + ghost_intensities,
        tracked=tracked,
        sw_hits=sw_hits,
        ms_hits=ms_hits,
        active_ids=active_ids,
        prev_interp_frame=next_prev_interp,
        motion_count=len(regions) if regions else 0,
    )


def _build_setup(cap, model, config, sam3_refiner):
    """Legge le dimensioni dal video e costruisce i parametri invarianti del run."""
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    proc = _init_frame_processors(fps, frame_w, frame_h, config)
    proc.sam3_refiner = sam3_refiner
    return _DetectionSetup(
        model=model, config=config, frame_w=frame_w, frame_h=frame_h, fps=fps, proc=proc
    )


def run_detection_loop(
    cap, total_frames, model, config, fisheye, stop_event=None, sam3_refiner=None
):
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
    sam3_refiner : Sam3ImageRefiner | None
        Refiner SAM3 per modalità ``yolo+sam3``. Se None, usa ``box_to_polygon``.

    Returns
    -------
    tuple
        (annotations dict, report_data dict, stats dict)
    """
    setup = _build_setup(cap, model, config, sam3_refiner)
    annotations, report_data, totals = {}, {}, DetectionTotals()
    prev_interp = None

    print("\n[FASE 1/5] Rilevamento automatico...")
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
                    totals.corrupted.append(frame_idx)
                    frame_idx += 1
                    pbar.update(1)
                    continue
                break

            result = _process_single_frame(fisheye.undistort(frame), prev_interp, setup)
            prev_interp = result.prev_interp_frame
            totals.record(result)

            annotations[frame_idx] = {
                "auto": result.polygons,
                "manual": [],
                "intensities": result.intensities,
            }
            report_data[frame_idx] = report_row(frame_idx, result)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
    finally:
        cap.release()

    print()
    for line in totals.summary_lines(total_frames):
        print(line)

    stats = {"unique_ids": totals.unique_ids, "total_instances": totals.total_instances}
    return annotations, report_data, stats
