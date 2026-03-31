"""
Rendering video anonimizzato e calcolo statistiche di review.

Contiene le funzioni per applicare le annotazioni di oscuramento ai frame
del video originale e produrre il file di output (con video debug opzionale),
oltre alla funzione di confronto statistico pre/post review manuale.
"""

import cv2
from tqdm import tqdm

from anonymization import draw_debug_polygons, obscure_polygon
from config import PipelineConfig
from preprocessing import undistort_frame


def render_video(
    input_path,
    output_path,
    annotations,
    fps,
    frame_w,
    frame_h,
    method,
    fisheye_enabled,
    undist_map1,
    undist_map2,
    config: PipelineConfig,
    debug_path=None,
    desc="Rendering",
):
    """
    Renderizza il video anonimizzato dal video originale.

    Parameters
    ----------
    input_path : str
        Percorso video sorgente (originale, non anonimizzato).
    output_path : str
        Percorso video di output (lossless FFV1).
    annotations : dict
        Annotazioni {frame_idx: {"auto": [...], "manual": [...], "intensities": [...]}}.
    fps : float
        Frame per secondo del video.
    frame_w : int
        Larghezza frame.
    frame_h : int
        Altezza frame.
    method : str
        Metodo di oscuramento ("pixelation" o "blur").
    fisheye_enabled : bool
        Se True, applica correzione fish-eye.
    undist_map1 : ndarray or None
        Mappa undistortion 1.
    undist_map2 : ndarray or None
        Mappa undistortion 2.
    config : PipelineConfig
        Configurazione della pipeline (intensità oscuramento, adaptive mode).
    debug_path : str or None
        Se specificato, genera anche il video debug.
    desc : str
        Descrizione per la barra di progresso.
    """
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    debug_writer = None
    if debug_path:
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (frame_w, frame_h))
    pbar = tqdm(total=total_frames, desc=desc, unit=" frame")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fisheye_enabled:
            frame = undistort_frame(frame, undist_map1, undist_map2)

        render_frame = frame.copy()
        ann = annotations.get(frame_idx, {"auto": [], "manual": [], "intensities": []})

        auto_polys = ann.get("auto", [])
        manual_polys = ann.get("manual", [])
        intensities = ann.get("intensities", [])

        for i, poly in enumerate(auto_polys):
            if config.enable_adaptive_intensity and i < len(intensities):
                intensity = intensities[i]
            else:
                intensity = config.anonymization_intensity
            if method == "blur" and intensity % 2 == 0:
                intensity += 1
            render_frame = obscure_polygon(render_frame, poly, method, intensity)

        for poly in manual_polys:
            intensity = config.anonymization_intensity
            if method == "blur" and intensity % 2 == 0:
                intensity += 1
            render_frame = obscure_polygon(render_frame, poly, method, intensity)

        out_writer.write(render_frame)

        if debug_writer:
            debug_frame = draw_debug_polygons(frame, auto_polys, manual_polys, config)
            debug_writer.write(debug_frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()
    if debug_writer:
        debug_writer.release()


def _compute_review_stats(original, reviewed, total_frames):
    """Calcola statistiche di review confrontando annotazioni prima e dopo.

    Parameters
    ----------
    original : dict
        Annotazioni originali {frame_idx: {auto: [...], manual: [...]}}.
    reviewed : dict
        Annotazioni dopo la review.
    total_frames : int
        Numero totale di frame nel video.

    Returns
    -------
    dict
        Contiene added, removed, frames_modified, frames_reviewed.
    """
    added = 0
    removed = 0
    frames_modified = 0

    all_frames = set(original.keys()) | set(reviewed.keys())
    for fidx in all_frames:
        orig_auto = original.get(fidx, {}).get("auto", [])
        orig_manual = original.get(fidx, {}).get("manual", [])
        orig_count = len(orig_auto) + len(orig_manual)

        rev_auto = reviewed.get(fidx, {}).get("auto", [])
        rev_manual = reviewed.get(fidx, {}).get("manual", [])
        rev_count = len(rev_auto) + len(rev_manual)

        diff = rev_count - orig_count
        if diff > 0:
            added += diff
        elif diff < 0:
            removed += abs(diff)

        if rev_count != orig_count:
            frames_modified += 1

    return {
        "added": added,
        "removed": removed,
        "frames_modified": frames_modified,
        "frames_reviewed": total_frames,
    }
