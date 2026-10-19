"""Pure drawing helpers of the manual review window: each returns a new array."""

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "OverlayStyle",
    "draw_annotations",
    "draw_pending_polygon",
    "draw_status_bars",
    "format_info_text",
]

INFO_BAR_HEIGHT = 30
HELP_BAR_HEIGHT = 45
BAR_COLOR = (40, 40, 40)
WHITE = (255, 255, 255)
HELP_TEXT_COLOR = (200, 200, 200)
INFO_TEXT_ORIGIN = (5, 20)
INFO_FONT_SCALE = 0.5
HELP_FONT_SCALE = 0.38
HELP_LINE1_OFFSET = 15
HELP_LINE2_OFFSET = 35
VERTEX_RADIUS = 4
PENDING_VERTEX_RADIUS = 5
OUTLINE_THICKNESS = 2
CURSOR_LINE_THICKNESS = 1
HELP_LINE1 = "  <-/-> Naviga  |  Spazio Avanti  |  Invio Chiudi poligono  |  Q Esci"
HELP_LINE2 = (
    "  Click Aggiungi punto  |  Backspace Annulla punto  |  D Elimina  |  Esc Annulla poligono"
)


@dataclass(frozen=True)
class OverlayStyle:
    """Colors (BGR) and fill opacity of the review overlay."""

    auto_color: tuple[int, int, int]
    manual_color: tuple[int, int, int]
    drawing_color: tuple[int, int, int]
    fill_alpha: float

    @classmethod
    def from_config(cls, config) -> "OverlayStyle":
        """Build the style from the `review_*` fields of a PipelineConfig."""
        return cls(
            auto_color=config.review_auto_color,
            manual_color=config.review_manual_color,
            drawing_color=config.review_drawing_color,
            fill_alpha=config.review_fill_alpha,
        )


def draw_annotations(frame: np.ndarray, annotation: dict, style: OverlayStyle) -> np.ndarray:
    """
    Draw auto and manual polygons (translucent fill, outline, white vertices).

    Returns a new frame; `frame` is left untouched.
    """
    result = frame.copy()
    overlay = frame.copy()
    layers = (
        (annotation.get("auto", []), style.auto_color),
        (annotation.get("manual", []), style.manual_color),
    )
    for polygons, color in layers:
        for polygon in polygons:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(result, [pts], True, color, OUTLINE_THICKNESS)
            for vertex in polygon:
                cv2.circle(result, tuple(vertex), VERTEX_RADIUS, WHITE, -1)
    cv2.addWeighted(overlay, style.fill_alpha, result, 1 - style.fill_alpha, 0, result)
    return result


def draw_pending_polygon(
    frame: np.ndarray, points: list[tuple[int, int]], cursor: tuple[int, int], color
) -> np.ndarray:
    """
    Draw the polygon being drawn, with a rubber-band line to the cursor.

    Points and cursor are in frame coordinates. Returns a new frame, a plain
    copy when there are no points.
    """
    result = frame.copy()
    if not points:
        return result
    for i, point in enumerate(points):
        cv2.circle(result, tuple(point), PENDING_VERTEX_RADIUS, WHITE, -1)
        if i > 0:
            cv2.line(result, tuple(points[i - 1]), tuple(point), color, OUTLINE_THICKNESS)
    cv2.line(result, tuple(points[-1]), cursor, color, CURSOR_LINE_THICKNESS, cv2.LINE_AA)
    return result


def format_info_text(frame_idx: int, total_frames: int, annotation: dict, delete_mode: bool) -> str:
    """Build the top bar text: 1-based frame counter, polygon counts and mode."""
    n_auto = len(annotation.get("auto", []))
    n_manual = len(annotation.get("manual", []))
    mode_label = "[DEL]" if delete_mode else "[NORM]"
    return (
        f"  Frame {frame_idx + 1} / {total_frames}  |  "
        f"Poligoni: {n_auto + n_manual} "
        f"({n_auto} auto, {n_manual} man)  |  {mode_label}"
    )


def draw_status_bars(frame: np.ndarray, info_text: str) -> np.ndarray:
    """Draw the info bar at the top and the help bar at the bottom on a copy of the frame."""
    result = frame.copy()
    height, width = result.shape[:2]
    cv2.rectangle(result, (0, 0), (width, INFO_BAR_HEIGHT), BAR_COLOR, -1)
    cv2.putText(
        result, info_text, INFO_TEXT_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, INFO_FONT_SCALE, WHITE, 1
    )
    y_start = height - HELP_BAR_HEIGHT
    cv2.rectangle(result, (0, y_start), (width, height), BAR_COLOR, -1)
    for text, offset in ((HELP_LINE1, HELP_LINE1_OFFSET), (HELP_LINE2, HELP_LINE2_OFFSET)):
        cv2.putText(
            result,
            text,
            (5, y_start + offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            HELP_FONT_SCALE,
            HELP_TEXT_COLOR,
            1,
        )
    return result
