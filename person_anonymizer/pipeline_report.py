"""Pure helpers for pipeline output paths and console report text."""

from pathlib import Path
from typing import NamedTuple

from .config import VERSION, PipelineConfig
from .models import OutputPaths, PipelineResult, VideoMeta

_RULE = "-" * 40

# (minimum frame height, label), checked from the largest down
_RESOLUTION_LABELS = ((2160, "4K"), (1080, "1080p"), (720, "720p"), (480, "480p"))


class RunLabels(NamedTuple):
    """Display strings of a run that the banner prints verbatim."""

    input_name: str
    output_name: str
    mode: str
    method: str


def resolution_label(frame_w: int, frame_h: int) -> str:
    """Return a short label for the frame height, or ``WxH`` below 480 lines."""
    for min_height, label in _RESOLUTION_LABELS:
        if frame_h >= min_height:
            return label
    return f"{frame_w}x{frame_h}"


def build_output_paths(input_path: str, output: str | None) -> OutputPaths:
    """Derive every output and temporary path of a run.

    Without an explicit ``output`` everything lands next to the input; with
    one, the temporary and side files follow the output directory.
    """
    stem = Path(input_path).stem
    if output:
        output_path, base_dir = output, Path(output).parent
    else:
        base_dir = Path(input_path).parent
        output_path = str(base_dir / f"{stem}_anonymized.mp4")
    return OutputPaths(
        output=output_path,
        temp_video=str(base_dir / f"{stem}_temp_noaudio.avi"),
        temp_debug=str(base_dir / f"{stem}_temp_debug.avi"),
        debug=str(base_dir / f"{stem}_debug.mp4"),
        report=str(base_dir / f"{stem}_report.csv"),
        json=str(base_dir / f"{stem}_annotations.json"),
    )


def _method_label(config: PipelineConfig, method: str) -> str:
    if config.enable_adaptive_intensity:
        return (
            f"{method} adattivo (base: {config.anonymization_intensity}px, "
            f"ref: {config.adaptive_reference_height}px)"
        )
    return f"{method} ({config.anonymization_intensity}px)"


def _detection_lines(config: PipelineConfig) -> list[str]:
    scales = ", ".join(f"{s}x" for s in config.inference_scales)
    if config.enable_sliding_window:
        grid = config.sliding_window_grid
        overlap = int(config.sliding_window_overlap * 100)
        sw_status = f"griglia {grid}x{grid}, overlap {overlap}%"
    else:
        sw_status = "disabilitato"
    fisheye = "abilitato" if config.enable_fisheye_correction else "disabilitato"
    return [
        f"Backend:        {config.detection_backend}",
        f"Modello:        {config.yolo_model}  |  Confidenza: {config.detection_confidence}",
        f"Scale:          [{scales}] + {', '.join(config.tta_augmentations)}",
        f"Sliding window: {sw_status}",
        f"Fish-eye:       {fisheye}",
    ]


def _temporal_lines(config: PipelineConfig, fps: float, do_interpolation: bool) -> list[str]:
    threshold = config.interpolation_fps_threshold
    tracking = (
        f"ByteTrack (max_age: {config.track_max_age})" if config.enable_tracking else "disabilitato"
    )
    smoothing = (
        f"EMA (alpha: {config.smoothing_alpha})"
        if config.enable_temporal_smoothing
        else "disabilitato"
    )
    interpolation = (
        f"abilitata ({fps:.0f}fps < {threshold}fps)"
        if do_interpolation
        else f"disabilitata (fps >= {threshold})"
    )
    return [
        f"Tracking:       {tracking}",
        f"Smoothing:      {smoothing}",
        f"Interpolazione: {interpolation}",
    ]


def format_banner(
    meta: VideoMeta, labels: RunLabels, config: PipelineConfig, do_interpolation: bool
) -> str:
    """Build the multi-line banner printed before the first stage."""
    res_label = resolution_label(frame_w=meta.frame_w, frame_h=meta.frame_h)
    lines = [
        f"\nPerson Anonymizer v{VERSION}",
        _RULE,
        f"Input:          {labels.input_name}  "
        f"({meta.total_frames} frame, {meta.fps:.0f}fps, {res_label})",
        f"Output:         {labels.output_name}",
        f"Modalità:      {labels.mode}",
        f"Metodo:         {_method_label(config=config, method=labels.method)}",
        *_detection_lines(config=config),
        *_temporal_lines(config=config, fps=meta.fps, do_interpolation=do_interpolation),
        _RULE,
    ]
    return "\n".join(lines)


def format_summary(
    elapsed_s: float, paths: OutputPaths, result: PipelineResult, normalized: bool
) -> str:
    """Build the closing summary, ending with the blank line the UI expects."""
    minutes, seconds = int(elapsed_s // 60), int(elapsed_s % 60)
    lines = [
        _RULE,
        f"Completato  —  Tempo totale: {minutes}m {seconds}s",
        f"  Output:       {Path(paths.output).name}",
    ]
    if result.enable_debug:
        lines.append(f"  Debug:        {Path(paths.debug).name}")
    if result.enable_report:
        lines.append(f"  Report:       {Path(paths.report).name}")
    if result.mode == "manual" or normalized:
        lines.append(f"  Annotazioni:  {Path(paths.json).name}")
    return "\n".join(lines) + "\n"
