"""Fase di revisione manuale (web o CLI)."""

from .models import PipelineContext
from .rendering import compute_review_stats

__all__ = ["run_manual_review_stage"]


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
    review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
    web_review_state = ctx.review_state

    if web_review_state is not None:
        print("\n[FASE 3/5] Revisione manuale — in attesa di conferma dal browser...")
        web_review_state.setup(
            input_path,
            annotations,
            total_frames,
            frame_w,
            frame_h,
            fps,
            fisheye,
        )
        sse_mgr = ctx.sse_manager
        web_job_id = ctx.job_id
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
        review_stats = compute_review_stats(original, annotations, total_frames)
        pct = (
            f"({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
            if total_frames > 0
            else "(N/A)"
        )
        print("\n  Revisione completata:")
        print(f"  Poligoni aggiunti:     {review_stats['added']}")
        print(f"  Poligoni rimossi:      {review_stats['removed']}")
        print(f"  Frame modificati:      {review_stats['frames_modified']}  {pct}")
    else:
        print("\n[FASE 3/5] Revisione manuale — apertura interfaccia...")
        print(
            "  -> Usa Spazio per navigare, Click per disegnare, D per eliminare, Q per confermare."
        )
        from .manual_reviewer import run_manual_review

        annotations, review_stats = run_manual_review(input_path, annotations, config, fisheye)
        pct_cli = (
            f"({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
            if total_frames > 0
            else "(N/A)"
        )
        print("\n  Revisione completata:")
        print(f"  Frame revisionati:     {review_stats['frames_reviewed']} / {total_frames}")
        print(f"  Poligoni aggiunti:     {review_stats['added']}")
        print(f"  Poligoni rimossi:      {review_stats['removed']}")
        print(f"  Frame modificati:      {review_stats['frames_modified']}  {pct_cli}")

    return annotations, review_stats
