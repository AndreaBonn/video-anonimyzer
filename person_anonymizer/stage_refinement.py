"""Fase di auto-refinement: loop di ri-rendering e verifica post-render."""

from dataclasses import dataclass
from typing import Any, NamedTuple

from .anonymization import box_to_polygon, resolve_intensity
from .models import PostCheckOutcome
from .postprocessing import filter_artifact_detections, run_post_render_check
from .rendering import render_video

__all__ = ["run_refinement_loop"]


@dataclass(frozen=True)
class RefinementJob:
    """Inputs that stay constant across the passes of a refinement loop."""

    input_path: str
    temp_video_path: str
    model: Any
    fps: float
    frame_w: int
    frame_h: int
    method: str
    fisheye: Any
    report_data: dict
    stop_event: Any = None


class ResidualAnnotation(NamedTuple):
    """Polygon and intensity to append for one genuine residual detection."""

    polygon: Any
    intensity: Any


def build_residual_additions(genuine_alerts, config, frame_w, frame_h):
    """Turn genuine residual detections into annotations to append, per frame.

    Pure: nothing is mutated. A frame with an empty box list still gets an
    (empty) entry, so that applying the result creates the frame record.

    Parameters
    ----------
    genuine_alerts : list
        Pairs (frame_idx, boxes) where each box starts with (x1, y1, x2, y2).
    config : PipelineConfig
        Configurazione della pipeline.
    frame_w : int
        Larghezza frame.
    frame_h : int
        Altezza frame.

    Returns
    -------
    dict
        {frame_idx: [ResidualAnnotation, ...]} in detection order.
    """
    additions = {}
    for fidx, boxes in genuine_alerts:
        residuals = additions.setdefault(fidx, [])
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            polygon = box_to_polygon(
                x1,
                y1,
                x2,
                y2,
                padding=config.person_padding,
                frame_w=frame_w,
                frame_h=frame_h,
                config=config,
            )
            residuals.append(
                ResidualAnnotation(polygon=polygon, intensity=resolve_intensity(config, y2 - y1))
            )
    return additions


def apply_residual_additions(annotations, additions):
    """Append ``additions`` to ``annotations`` in place and return how many were added.

    In place on purpose: callers keep a reference to the same dict across passes.
    """
    added = 0
    for fidx, residuals in additions.items():
        if fidx not in annotations:
            annotations[fidx] = {"auto": [], "manual": [], "intensities": []}
        for residual in residuals:
            annotations[fidx]["auto"].append(residual.polygon)
            annotations[fidx]["intensities"].append(residual.intensity)
            added += 1
    return added


def _run_refinement_pass(job, annotations, config, pass_label):
    """Render, run the post-render check and drop pixelation artifacts.

    Returns the genuine residual alerts; an empty list means the render is clean.
    """
    print(f"\n[FASE 2/5] Auto refinement — rendering ({pass_label})...")
    render_video(
        job.input_path,
        job.temp_video_path,
        annotations,
        job.fps,
        job.frame_w,
        job.frame_h,
        job.method,
        job.fisheye,
        config,
        debug_path=None,
        desc=f"Rendering ({pass_label})",
        stop_event=job.stop_event,
    )

    print(f"\n  Verifica post-rendering ({pass_label})...")
    alert_frames = run_post_render_check(
        job.temp_video_path, job.model, config.post_render_check_confidence, job.report_data, config
    )

    if not alert_frames:
        print("\n  Nessun rilevamento residuo — rendering OK.")
        return []

    genuine_alerts, n_artifacts, n_genuine = filter_artifact_detections(
        alert_frames, annotations, config.refinement_overlap_threshold
    )
    print(f"\n  Rilevamenti post-render: {n_artifacts + n_genuine}")
    print(f"  Artefatti filtrati (IoU >= {config.refinement_overlap_threshold}): {n_artifacts}")
    print(f"  Residui genuini: {n_genuine}")

    if not genuine_alerts:
        print("  Tutti i rilevamenti sono artefatti della pixelazione — rendering OK.")
    return genuine_alerts


def _report_residual_limit(genuine_alerts, config):
    """Print the residuals left at the pass limit and return the matching outcome."""
    print(f"\n  Raggiunto limite di {config.max_refinement_passes} pass.")
    print(f"  Residui genuini rimasti in {len(genuine_alerts)} frame:")
    for fidx, boxes in genuine_alerts:
        print(f"    Frame {fidx}: {len(boxes)} persona/e")
    return PostCheckOutcome(status="residual", residual_frames=len(genuine_alerts))


def run_refinement_loop(
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
    stop_event=None,
):
    """Loop di auto-refinement.

    Oltre le 30 righe: il loop dei pass e la regola dell'ultimo pass si leggono in un punto solo.

    Parameters
    ----------
    input_path : str
        Percorso del video sorgente.
    annotations : dict
        Annotazioni correnti.
    model : YOLO
        Modello di detection.
    config : PipelineConfig
        Configurazione della pipeline.
    fps : float
        Frame rate del video.
    frame_w : int
        Larghezza frame.
    frame_h : int
        Altezza frame.
    method : str
        Metodo di anonimizzazione.
    fisheye : FisheyeContext
        Contesto di correzione fish-eye.
    report_data : dict
        Dati del report frame-per-frame.
    temp_video_path : str
        Percorso del video temporaneo.
    stop_event : threading.Event | None
        Evento di stop.

    Returns
    -------
    tuple
        (annotations, actual_passes, annotations_added, PostCheckOutcome)
    """
    if not config.enable_post_render_check:
        print("\n[FASE 2/5] Auto refinement — saltato (verifica disabilitata)")
        return annotations, 0, 0, PostCheckOutcome(status="skipped")

    job = RefinementJob(
        input_path=input_path,
        temp_video_path=temp_video_path,
        model=model,
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
        method=method,
        fisheye=fisheye,
        report_data=report_data,
        stop_event=stop_event,
    )
    actual_passes, annotations_added = 0, 0
    outcome = PostCheckOutcome(status="skipped")
    for pass_num in range(1, config.max_refinement_passes + 1):
        if stop_event is not None and stop_event.is_set():
            print("\n  Pipeline interrotta dall'utente.")
            break

        actual_passes = pass_num
        pass_label = f"pass {pass_num}/{config.max_refinement_passes}"
        genuine_alerts = _run_refinement_pass(job, annotations, config, pass_label)

        if not genuine_alerts:
            outcome = PostCheckOutcome(status="clean")
            break

        # Last pass: residuals are reported, never added (no further render would use them).
        if pass_num == config.max_refinement_passes:
            outcome = _report_residual_limit(genuine_alerts, config)
            break

        additions = build_residual_additions(genuine_alerts, config, frame_w, frame_h)
        added_this_pass = apply_residual_additions(annotations, additions)
        annotations_added += added_this_pass
        print(f"\n  Aggiunte {added_this_pass} annotazioni — ri-rendering in corso...")

    return annotations, actual_passes, annotations_added, outcome
