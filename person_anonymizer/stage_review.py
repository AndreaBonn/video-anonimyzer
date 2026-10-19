"""Fase di revisione manuale (web o CLI)."""

from typing import Any, NamedTuple

from .models import FisheyeContext, PipelineContext, VideoMeta
from .rendering import compute_review_stats

__all__ = ["format_review_summary", "run_manual_review_stage"]


class _ReviewTarget(NamedTuple):
    """What both review front-ends need to know about the video under review."""

    input_path: str
    fisheye: FisheyeContext
    meta: VideoMeta


def format_review_summary(
    stats: dict[str, Any], total_frames: int, *, show_reviewed: bool
) -> list[str]:
    """Build the console lines printed after a manual review.

    Parameters
    ----------
    stats : dict
        Review statistics (added, removed, frames_modified and, when
        ``show_reviewed`` is set, frames_reviewed).
    total_frames : int
        Total frames of the video, used for the modified percentage.
    show_reviewed : bool
        Whether to include the "Frame revisionati" line (CLI only).

    Returns
    -------
    list of str
        Lines to print in order; the first one is the section header.
    """
    pct = f"({stats['frames_modified'] / total_frames * 100:.1f}%)" if total_frames > 0 else "(N/A)"
    lines = ["\n  Revisione completata:"]
    if show_reviewed:
        lines.append(f"  Frame revisionati:     {stats['frames_reviewed']} / {total_frames}")
    lines += [
        f"  Poligoni aggiunti:     {stats['added']}",
        f"  Poligoni rimossi:      {stats['removed']}",
        f"  Frame modificati:      {stats['frames_modified']}  {pct}",
    ]
    return lines


def _review_in_browser(ctx: PipelineContext, target: _ReviewTarget, annotations: dict):
    """Web branch: hand the session to the browser and wait for confirmation."""
    state, meta = ctx.review_state, target.meta
    print("\n[FASE 3/5] Revisione manuale — in attesa di conferma dal browser...")
    state.setup(
        target.input_path,
        annotations,
        meta.total_frames,
        meta.frame_w,
        meta.frame_h,
        meta.fps,
        target.fisheye,
    )
    ctx.sse_manager.emit(
        ctx.job_id,
        "review_ready",
        {
            "total_frames": meta.total_frames,
            "frame_w": meta.frame_w,
            "frame_h": meta.frame_h,
            "fps": meta.fps,
        },
    )
    original = {
        fidx: {"auto": list(d.get("auto", [])), "manual": list(d.get("manual", []))}
        for fidx, d in annotations.items()
    }
    # Uno stop arrivato prima di setup() ha già chiamato cancel(), che setup() ha azzerato
    if ctx.stop_event is not None and ctx.stop_event.is_set():
        state.cancel()
    annotations = state.wait_for_completion()
    review_stats = compute_review_stats(original, annotations, meta.total_frames)
    for line in format_review_summary(review_stats, meta.total_frames, show_reviewed=False):
        print(line)
    return annotations, review_stats


def _review_in_window(target: _ReviewTarget, annotations: dict, config):
    """CLI branch: open the OpenCV reviewer window."""
    print("\n[FASE 3/5] Revisione manuale — apertura interfaccia...")
    print("  -> Usa Spazio per navigare, Click per disegnare, D per eliminare, Q per confermare.")
    from .manual_reviewer import run_manual_review

    annotations, review_stats = run_manual_review(
        target.input_path, annotations, config, target.fisheye
    )
    for line in format_review_summary(review_stats, target.meta.total_frames, show_reviewed=True):
        print(line)
    return annotations, review_stats


def run_manual_review_stage(
    ctx: PipelineContext,
    input_path,
    annotations,
    config,
    total_frames,
    fps,
    frame_w,
    frame_h,
    fisheye,
):
    """Revisione manuale (web o CLI).

    Parameters
    ----------
    ctx : PipelineContext
        Contesto della pipeline con review_state, sse_manager e job_id.
    input_path : str
        Percorso del video sorgente.
    annotations : dict
        Annotazioni correnti.
    config : PipelineConfig
        Configurazione della pipeline.
    total_frames : int
        Numero totale di frame.
    fps : float
        Frame rate del video.
    frame_w : int
        Larghezza frame.
    frame_h : int
        Altezza frame.
    fisheye : FisheyeContext
        Contesto di correzione fish-eye.

    Returns
    -------
    tuple
        (annotations, review_stats)
    """
    target = _ReviewTarget(
        input_path=input_path,
        fisheye=fisheye,
        meta=VideoMeta(fps=fps, frame_w=frame_w, frame_h=frame_h, total_frames=total_frames),
    )
    if ctx.review_state is not None:
        return _review_in_browser(ctx, target, annotations)
    return _review_in_window(target, annotations, config)
