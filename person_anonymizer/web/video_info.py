"""Read basic metadata of an uploaded video for display in the web UI."""

import math
from pathlib import Path

import cv2


def read_video_info(path: Path) -> dict:
    """Return duration and resolution of a video file.

    Parameters
    ----------
    path : Path
        Video file to inspect.

    Returns
    -------
    dict
        ``duration_s``, ``width``, ``height``; all ``None`` when OpenCV cannot
        decode the file (the upload is still accepted, the UI just omits them).
    """
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    # Some containers report NaN/inf: jsonify would emit an invalid "NaN" token
    if not (math.isfinite(fps) and math.isfinite(frames)):
        return {"duration_s": None, "width": None, "height": None}
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        return {"duration_s": None, "width": None, "height": None}
    return {"duration_s": round(frames / fps, 1), "width": width, "height": height}
