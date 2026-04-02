"""
Normalizzazione annotazioni: merge poligoni sovrapposti.

Contiene le funzioni per il merge tramite Union-Find dei rettangoli
sovrapposti e la normalizzazione delle annotazioni per frame.
"""

import logging

import cv2
import numpy as np

from .anonymization import compute_adaptive_intensity
from .config import PipelineConfig

__all__ = [
    "_rects_overlap",
    "_merge_rects",
    "_merge_overlapping_rects",
    "normalize_annotations",
]

_log = logging.getLogger(__name__)


def _rects_overlap(r1, r2):
    """
    Verifica se due rettangoli (x, y, w, h) si sovrappongono.

    Parameters
    ----------
    r1 : tuple
        Primo rettangolo (x, y, w, h).
    r2 : tuple
        Secondo rettangolo (x, y, w, h).

    Returns
    -------
    bool
        True se i rettangoli si sovrappongono.
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def _merge_rects(r1, r2):
    """
    Unifica due rettangoli (x, y, w, h) nel bounding box che li contiene.

    Parameters
    ----------
    r1 : tuple
        Primo rettangolo (x, y, w, h).
    r2 : tuple
        Secondo rettangolo (x, y, w, h).

    Returns
    -------
    tuple
        Rettangolo unificato (x, y, w, h).
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    nx = min(x1, x2)
    ny = min(y1, y2)
    nx2 = max(x1 + w1, x2 + w2)
    ny2 = max(y1 + h1, y2 + h2)
    return (nx, ny, nx2 - nx, ny2 - ny)


def _merge_overlapping_rects(rects):
    """
    Merge di rettangoli sovrapposti tramite Union-Find.

    Parameters
    ----------
    rects : list of tuple
        Lista di rettangoli (x, y, w, h).

    Returns
    -------
    list of tuple
        Rettangoli non sovrapposti dopo il merge.
    """
    if not rects:
        return []

    n = len(rects)
    if n > 100:
        _log.warning(
            "_merge_overlapping_rects: %d rects, performance O(n²) potenzialmente lenta", n
        )
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # O(n²) overlap check: accettabile per n < 50 (tipico per sorveglianza).
    # Per n grande, usare sweep-line O(n log n) con ordinamento per x1.
    for i in range(n):
        for j in range(i + 1, n):
            if _rects_overlap(rects[i], rects[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = rects[i]
        else:
            groups[root] = _merge_rects(groups[root], rects[i])

    return list(groups.values())


def normalize_annotations(annotations, config: PipelineConfig):
    """
    Normalizza le annotazioni: converte poligoni in rettangoli
    e unifica le aree sovrapposte.

    Parameters
    ----------
    annotations : dict
        Dizionario {frame_idx: {"auto": [...], "manual": [...], "intensities": [...]}}.
    config : PipelineConfig
        Configurazione della pipeline (intensità adattiva, reference height).

    Returns
    -------
    dict
        Annotazioni normalizzate con rettangoli non sovrapposti.
    tuple
        (poligoni_prima, poligoni_dopo) per statistiche.
    """
    total_before = 0
    total_after = 0
    normalized = {}

    for fidx, ann in annotations.items():
        all_polys = ann.get("auto", []) + ann.get("manual", [])
        total_before += len(all_polys)

        if not all_polys:
            normalized[fidx] = {"auto": [], "manual": [], "intensities": []}
            continue

        # Converti ogni poligono nel suo bounding rectangle (x, y, w, h)
        rects = []
        for poly in all_polys:
            pts = np.array(poly, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            rects.append((x, y, w, h))

        # Merge iterativo dei rettangoli sovrapposti
        merged = _merge_overlapping_rects(rects)
        total_after += len(merged)

        # Converti rettangoli in poligoni e ricalcola intensità
        new_polys = []
        new_intensities = []
        for x, y, w, h in merged:
            poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            new_polys.append(poly)
            if config.enable_adaptive_intensity:
                intensity = compute_adaptive_intensity(
                    h, config.anonymization_intensity, config.adaptive_reference_height
                )
            else:
                intensity = config.anonymization_intensity
            new_intensities.append(intensity)

        normalized[fidx] = {"auto": new_polys, "manual": [], "intensities": new_intensities}

    return normalized, (total_before, total_after)
