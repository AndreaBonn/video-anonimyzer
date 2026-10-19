"""Pipeline principale di anonimizzazione — orchestratore."""

import logging
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from .backend_factory import load_detection_backend
from .config import SUPPORTED_EXTENSIONS, PipelineConfig
from .models import (
    FisheyeContext,
    OutputPaths,
    PipelineContext,
    PipelineInputError,
    PipelineResult,
    PipelineStoppedError,
    PostCheckOutcome,
    VideoMeta,
)
from .output import load_annotations_from_json, save_outputs
from .pipeline_report import RunLabels, build_output_paths, format_banner, format_summary
from .pipeline_stages import run_detection_loop, run_manual_review_stage, run_refinement_loop
from .postprocessing import normalize_annotations
from .preprocessing import build_undistortion_maps, should_interpolate
from .rendering import render_video

_log = logging.getLogger(__name__)

__all__ = ["run_pipeline"]

_DEFAULT_FPS = 25.0
_LOW_FPS_LIMIT = 3
_EMPTY_REVIEW_STATS = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}


@dataclass(frozen=True)
class _RunSetup:
    """Values fixed for the whole run, shared by the stage helpers."""

    ctx: PipelineContext
    config: PipelineConfig
    meta: VideoMeta
    fisheye: FisheyeContext
    method: str
    mode: str
    paths: OutputPaths


def _raise_if_stopped(stop_event, temp_paths: tuple[str, ...]) -> None:
    """Interrompe la pipeline fra una fase e l'altra se l'utente lo ha chiesto.

    Le fasi escono dal proprio loop con risultati parziali: proseguire
    salverebbe un video anonimizzato solo in parte. Per lo stesso motivo
    vengono rimossi i render intermedi già scritti su disco.
    """
    if stop_event is None or not stop_event.is_set():
        return
    for path in temp_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            # The stop must still be reported as a stop, not as a generic error
            _log.warning("Impossibile rimuovere il file temporaneo %s", path, exc_info=True)
    raise PipelineStoppedError("Pipeline interrotta dall'utente")


def _validate_inputs(ctx: PipelineContext) -> None:
    if ctx.normalize and not ctx.review:
        raise PipelineInputError("--normalize richiede --review <json>")
    if not os.path.isfile(ctx.input):
        raise PipelineInputError(f"File non trovato: {ctx.input}")
    ext = Path(ctx.input).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PipelineInputError(f"Formato non supportato '{ext}'. Supportati: {supported}")


def _warn_if_no_ffmpeg() -> bool:
    available = shutil.which("ffmpeg") is not None
    if not available:
        print("ATTENZIONE: ffmpeg non trovato nel PATH.")
        print("  L'audio non verrà preservato nel video di output.")
        print("  Installare ffmpeg: sudo apt install ffmpeg")
    return available


def _open_video(input_path: str) -> tuple[cv2.VideoCapture, VideoMeta]:
    """Open the input video and read its metadata.

    The capture is released here when opening or probing fails, so the caller
    owns it only once this function returns.
    """
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise PipelineInputError(f"Impossibile aprire il video: {input_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise PipelineInputError("Impossibile determinare il numero di frame")
        meta = VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS,
            frame_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=total_frames,
        )
    except BaseException:
        cap.release()
        raise
    return cap, meta


def _announce(setup: _RunSetup) -> None:
    """Print the run banner, preceded by the low-fps warning when it applies."""
    ctx, config, meta = setup.ctx, setup.config, setup.meta
    interpolating = config.enable_subframe_interpolation and should_interpolate(
        meta.fps, config.interpolation_fps_threshold
    )
    if interpolating and meta.fps < _LOW_FPS_LIMIT:
        print(
            "ATTENZIONE: fps molto basso (< 3). L'interpolazione potrebbe produrre artefatti. "
            "Considera di disabilitarla."
        )
    labels = RunLabels(
        input_name=Path(ctx.input).name,
        output_name=Path(setup.paths.output).name,
        mode=setup.mode,
        method=setup.method,
    )
    print(format_banner(meta=meta, labels=labels, config=config, do_interpolation=interpolating))


def build_fisheye(config: PipelineConfig, frame_w: int, frame_h: int) -> FisheyeContext:
    """Return the undistortion context, disabled unless correction and calibration are set."""
    camera, dist = config.camera_matrix, config.dist_coefficients
    if not config.enable_fisheye_correction or camera is None or dist is None:
        return FisheyeContext()
    map1, map2 = build_undistortion_maps(camera, dist, frame_w, frame_h)
    return FisheyeContext(enabled=True, undist_map1=map1, undist_map2=map2)


def _detect_or_load(cap, setup: _RunSetup, backend) -> tuple[dict, dict, str]:
    """Phase 1: load annotations from JSON or detect. Returns the final mode."""
    ctx, config, mode = setup.ctx, setup.config, setup.mode
    if ctx.review:
        annotations, mode = load_annotations_from_json(ctx.review, config)
        if ctx.normalize:
            print("\n  Normalizzazione annotazioni...")
            annotations, (n_before, n_after) = normalize_annotations(annotations, config)
            print(f"  Poligoni prima:  {n_before}")
            print(f"  Rettangoli dopo: {n_after}  (riduzione: {n_before - n_after})")
            mode = "auto"
        cap.release()
        return annotations, {}, mode
    if config.detection_backend == "sam3" and backend.sam3_video_detector is not None:
        cap.release()
        detector = backend.sam3_video_detector
        annotations, report_data = detector.detect_video(ctx.input, config, ctx.stop_event)
    else:
        annotations, report_data, _ = run_detection_loop(
            cap,
            setup.meta.total_frames,
            backend.yolo_model,
            config,
            setup.fisheye,
            ctx.stop_event,
            sam3_refiner=backend.sam3_refiner,
        )
    return annotations, report_data, mode


def _refine(setup: _RunSetup, annotations: dict, model, report_data: dict) -> tuple:
    """Phase 2: auto-refinement, skipped when annotations come from JSON."""
    ctx, meta = setup.ctx, setup.meta
    if ctx.review:
        return annotations, 0, 0, PostCheckOutcome(status="skipped")
    annotations, passes, added, post_check = run_refinement_loop(
        ctx.input,
        annotations,
        model,
        setup.config,
        meta.fps,
        meta.frame_w,
        meta.frame_h,
        setup.method,
        setup.fisheye,
        report_data,
        setup.paths.temp_video,
        ctx.stop_event,
    )
    if post_check.status == "residual":
        print(
            "  -> Verranno mostrati nella revisione manuale."
            if setup.mode == "manual"
            else "  -> Rieseguire con --mode manual per correzione."
        )
    return annotations, passes, added, post_check


def _review(setup: _RunSetup, annotations: dict, mode: str) -> tuple[dict, dict]:
    """Phase 3: manual review, or a no-op with zeroed stats in auto mode."""
    if mode != "manual":
        print("\n[FASE 3/5] Revisione manuale — saltata (modalità auto)")
        return annotations, dict(_EMPTY_REVIEW_STATS)
    ctx, meta = setup.ctx, setup.meta
    return run_manual_review_stage(
        ctx,
        ctx.input,
        annotations,
        setup.config,
        meta.total_frames,
        meta.fps,
        meta.frame_w,
        meta.frame_h,
        setup.fisheye,
    )


def _render(setup: _RunSetup, annotations: dict) -> None:
    print("\n[FASE 4/5] Rendering finale...")
    ctx, meta, paths = setup.ctx, setup.meta, setup.paths
    render_video(
        ctx.input,
        paths.temp_video,
        annotations,
        meta.fps,
        meta.frame_w,
        meta.frame_h,
        setup.method,
        setup.fisheye,
        setup.config,
        debug_path=None if ctx.no_debug else paths.temp_debug,
        desc="Rendering finale",
        stop_event=ctx.stop_event,
    )


def run_pipeline(ctx: PipelineContext, config=None) -> PostCheckOutcome:
    """Esegue la pipeline completa di anonimizzazione.

    Eccezione ai 30 righe: fasi, controlli di stop e ciclo di ``cap`` sono la spec dell'ordine.

    Parameters
    ----------
    ctx : PipelineContext
        Contesto della pipeline con input, mode, method, output, flag e stato web.
    config : PipelineConfig | None
        Configurazione della pipeline. Se None, usa i default.

    Returns
    -------
    PostCheckOutcome
        Esito della verifica automatica post-render (fase 2).

    Raises
    ------
    PipelineInputError
        Se il file di input non esiste o non è supportato.
    PipelineStoppedError
        Se ``ctx.stop_event`` viene impostato durante l'elaborazione.
    PipelineError
        Per errori generici durante l'elaborazione.
    """
    if config is None:
        config = PipelineConfig()
    _validate_inputs(ctx)
    ffmpeg_available = _warn_if_no_ffmpeg()
    mode = ctx.mode or config.operation_mode
    method = ctx.method or config.anonymization_method
    paths = build_output_paths(input_path=ctx.input, output=ctx.output)
    temps = (paths.temp_video, paths.temp_debug)

    cap, meta = _open_video(ctx.input)
    # The banner is printed before the backend loads and the fisheye maps are
    # built after it, as before the refactor: stdout order and which error
    # surfaces first are part of the CLI and web contract.
    setup = _RunSetup(ctx, config, meta, FisheyeContext(), method, mode, paths)
    try:
        _announce(setup)
        backend = load_detection_backend(config)
        setup = replace(setup, fisheye=build_fisheye(config, meta.frame_w, meta.frame_h))
        start_time = time.time()
        annotations, report_data, mode = _detect_or_load(cap, setup, backend)
    finally:
        cap.release()

    _raise_if_stopped(ctx.stop_event, temps)
    annotations, passes, added, post_check = _refine(
        setup, annotations, backend.yolo_model, report_data
    )
    _raise_if_stopped(ctx.stop_event, temps)
    annotations, review_stats = _review(setup, annotations, mode)
    _raise_if_stopped(ctx.stop_event, temps)
    _render(setup, annotations)
    _raise_if_stopped(ctx.stop_event, temps)

    print("\n[FASE 5/5] Post-processing...")
    result = PipelineResult(
        annotations=annotations,
        report_data=report_data,
        review_stats=review_stats,
        method=method,
        mode=mode,
        enable_debug=not ctx.no_debug,
        enable_report=not ctx.no_report,
        ffmpeg_available=ffmpeg_available,
        actual_refinement_passes=passes,
        refinement_annotations_added=added,
    )
    save_outputs(ctx, result, config, ctx.input, paths, meta)
    elapsed = time.time() - start_time
    print(format_summary(elapsed_s=elapsed, paths=paths, result=result, normalized=ctx.normalize))
    return post_check
