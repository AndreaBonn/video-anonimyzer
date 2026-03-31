"""
Modulo anonymization per Person Anonymizer.

Contiene le funzioni di oscuramento (pixelation/blur) su poligoni,
debug visivo, conversione box<->poligono e intensità adattiva.
"""

import cv2
import numpy as np

from config import PipelineConfig


# ============================================================
# INTENSITA' ADATTIVA
# ============================================================


def compute_adaptive_intensity(box_height, base_intensity, reference_height):
    """
    Calcola intensità di oscuramento proporzionale all'altezza della figura.

    Per persone piccole (< 50px), garantisce un'intensità minima sufficiente
    a produrre al massimo 4 blocchi visibili, assicurando che l'oscuramento
    sia efficace anche su figure lontane dalla camera.
    """
    if reference_height <= 0:
        return base_intensity
    scale_factor = box_height / reference_height
    adaptive = int(base_intensity * scale_factor)
    # Soglia minima: max tra 3 e box_height//4 (garantisce max 4 blocchi)
    min_intensity = max(3, int(box_height) // 4)
    return max(min_intensity, adaptive)


# ============================================================
# OSCURAMENTO
# ============================================================


def obscure_polygon(frame, points, method, intensity):
    """Applica oscuramento (pixelation o blur) dentro un poligono."""
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    if w == 0 or h == 0:
        return frame

    if method == "pixelation":
        roi = frame[y : y + h, x : x + w].copy()
        block = max(1, intensity)
        small = cv2.resize(
            roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR
        )
        obscured_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # blur
        k = intensity if intensity % 2 == 1 else intensity + 1
        obscured_roi = cv2.GaussianBlur(frame[y : y + h, x : x + w], (k, k), 0)

    mask_roi = mask[y : y + h, x : x + w]
    frame[y : y + h, x : x + w] = np.where(
        mask_roi[:, :, np.newaxis] == 255, obscured_roi, frame[y : y + h, x : x + w]
    )
    return frame


# ============================================================
# DEBUG VISIVO
# ============================================================


def draw_debug_polygons(frame, auto_polygons, manual_polygons, config: PipelineConfig):
    """
    Disegna poligoni colorati per il video debug.

    Parameters
    ----------
    frame : ndarray
        Frame BGR su cui disegnare.
    auto_polygons : list
        Lista di poligoni rilevati automaticamente.
    manual_polygons : list
        Lista di poligoni aggiunti manualmente.
    config : PipelineConfig
        Configurazione della pipeline; usa ``review_auto_color``,
        ``review_manual_color`` e ``review_fill_alpha``.

    Returns
    -------
    ndarray
        Frame con poligoni disegnati.
    """
    debug_frame = frame.copy()
    overlay = debug_frame.copy()
    for poly in auto_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], config.review_auto_color)
        cv2.polylines(debug_frame, [pts], True, config.review_auto_color, 2)
    for poly in manual_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], config.review_manual_color)
        cv2.polylines(debug_frame, [pts], True, config.review_manual_color, 2)
    cv2.addWeighted(
        overlay, config.review_fill_alpha, debug_frame, 1 - config.review_fill_alpha, 0, debug_frame
    )
    return debug_frame


# ============================================================
# BOX -> POLIGONO
# ============================================================


def box_to_polygon(x1, y1, x2, y2, padding=0, frame_w=None, frame_h=None, config=None):
    """
    Converte bounding box in poligono 4 punti con padding direzionale.

    Applica padding maggiorato quando il box è vicino ai bordi del frame,
    per coprire persone parzialmente visibili in ingresso/uscita dalla scena.
    Il clamping avviene DOPO il padding.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Coordinate del bounding box.
    padding : int, optional
        Padding base in pixel (default: 0).
    frame_w : int or None, optional
        Larghezza del frame per il calcolo bordi.
    frame_h : int or None, optional
        Altezza del frame per il calcolo bordi.
    config : PipelineConfig or None, optional
        Se fornito, usa ``config.edge_threshold`` e
        ``config.edge_padding_multiplier``; altrimenti usa i valori
        di default hardcoded (0.05 e 2.5).

    Returns
    -------
    list of tuple
        Lista di 4 punti [(x1, y1), (x2, y1), (x2, y2), (x1, y2)].
    """
    edge_threshold = config.edge_threshold if config is not None else 0.05
    edge_padding_multiplier = config.edge_padding_multiplier if config is not None else 2.5

    if frame_w and frame_h:
        edge_x = int(frame_w * edge_threshold)
        edge_y = int(frame_h * edge_threshold)
        pad_l = int(padding * edge_padding_multiplier) if x1 < edge_x else padding
        pad_t = int(padding * edge_padding_multiplier) if y1 < edge_y else padding
        pad_r = int(padding * edge_padding_multiplier) if frame_w - x2 < edge_x else padding
        pad_b = int(padding * edge_padding_multiplier) if frame_h - y2 < edge_y else padding
    else:
        pad_l = pad_t = pad_r = pad_b = padding

    x1 = max(0, int(x1) - pad_l)
    y1 = max(0, int(y1) - pad_t)
    x2 = int(x2) + pad_r
    y2 = int(y2) + pad_b
    if frame_w is not None:
        x2 = min(x2, frame_w)
    if frame_h is not None:
        y2 = min(y2, frame_h)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_to_bbox(polygon):
    """
    Converte un poligono in bounding box [x1, y1, x2, y2].

    Parameters
    ----------
    polygon : list of tuple
        Lista di punti (x, y) del poligono.

    Returns
    -------
    list
        Bounding box [x1, y1, x2, y2].
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]
