"""
Modulo di rilevamento multi-strategia per la pipeline di anonimizzazione.

Contiene le funzioni di sliding window, inferenza multi-scala con TTA,
Non-Maximum Suppression, calcolo IoU e il punto di ingresso unificato
run_full_detection che orchestra tutte le strategie su un singolo frame.
"""

import cv2
import numpy as np

from config import PipelineConfig


# ============================================================
# RILEVAMENTO MULTI-STRATEGIA
# ============================================================


def get_window_patches(frame_w, frame_h, grid, overlap):
    """Genera patch per sliding window."""
    step_x = int(frame_w / grid * (1 - overlap))
    step_y = int(frame_h / grid * (1 - overlap))
    patch_w = int(frame_w / grid * (1 + overlap))
    patch_h = int(frame_h / grid * (1 + overlap))
    patches = []
    for row in range(grid):
        for col in range(grid):
            x1 = col * step_x
            y1 = row * step_y
            x2 = min(frame_w, x1 + patch_w)
            y2 = min(frame_h, y1 + patch_h)
            patches.append((x1, y1, x2, y2))
    return patches


def patch_intersects_motion(px1, py1, px2, py2, motion_regions):
    """Verifica se una patch interseca almeno una zona di movimento."""
    for mx1, my1, mx2, my2 in motion_regions:
        if px1 < mx2 and px2 > mx1 and py1 < my2 and py2 > my1:
            return True
    return False


def run_sliding_window(model, frame, patches, conf, motion_regions):
    """Esegue YOLO su ogni patch della griglia sliding window."""
    all_boxes = []
    hits = 0
    for px1, py1, px2, py2 in patches:
        if motion_regions is not None and len(motion_regions) > 0:
            if not patch_intersects_motion(px1, py1, px2, py2, motion_regions):
                continue
        patch = frame[py1:py2, px1:px2]
        results = model(patch, conf=conf, classes=[0], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            all_boxes.append([x1 + px1, y1 + py1, x2 + px1, y2 + py1, float(box.conf[0])])
        if len(results[0].boxes) > 0:
            hits += 1
    return all_boxes, hits


def detect_and_rescale(model, frame, conf, scale, base_imgsz=640):
    """
    Esegue YOLO con imgsz adattivo e riscala le coordinate.

    Per scale > 1.0, aumenta la risoluzione di input di YOLO fino a 1280,
    sfruttando effettivamente l'upscaling. Senza questo, YOLO comprime
    internamente a 640px e il beneficio del multi-scale è nullo.
    """
    effective_imgsz = min(int(base_imgsz * max(1.0, scale)), 1280)
    results = model(frame, conf=conf, classes=[0], verbose=False, imgsz=effective_imgsz)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        boxes.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale, float(box.conf[0])])
    return boxes


def run_multiscale_inference(model, frame, scales, augmentations, conf, orig_w, orig_h):
    """Esegue inferenza multi-scala con TTA."""
    all_boxes = []
    hits = 0
    for scale in scales:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        # INTER_CUBIC preserva più dettagli rispetto a INTER_LINEAR per
        # upscaling, migliorando il riconoscimento di persone piccole
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        boxes = detect_and_rescale(model, scaled, conf, scale)
        all_boxes.extend(boxes)
        if boxes:
            hits += 1
        for aug in augmentations:
            if aug == "flip_h":
                flipped = cv2.flip(scaled, 1)
                flip_boxes = detect_and_rescale(model, flipped, conf, scale)
                for b in flip_boxes:
                    b[0], b[2] = orig_w - b[2], orig_w - b[0]
                all_boxes.extend(flip_boxes)
                if flip_boxes:
                    hits += 1
    return all_boxes, hits


def apply_nms(boxes, iou_threshold):
    """Applica Non-Maximum Suppression."""
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_np]
    scores = list(boxes_np[:, 4])
    indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        return [boxes[i] for i in indices.flatten()]
    return []


def compute_iou_boxes(box_a, box_b):
    """
    Calcola IoU tra due bounding box [x1, y1, x2, y2].

    Parameters
    ----------
    box_a : list or tuple
        Primo box [x1, y1, x2, y2].
    box_b : list or tuple
        Secondo box [x1, y1, x2, y2].

    Returns
    -------
    float
        Intersection over Union (0.0 - 1.0).
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# DETECTION COMPLETA SU UN FRAME
# ============================================================


def run_full_detection(model, frame, conf, frame_w, frame_h, motion_regions, patches, config):
    """
    Esegue rilevamento sliding window + multi-scale + TTA su un frame.

    NMS a due stadi: prima interna per-strategia (deduplicazione),
    poi le box vengono combinate per la NMS finale cross-strategia.

    Parameters
    ----------
    model : YOLO
        Modello YOLO caricato.
    frame : ndarray
        Frame BGR corrente.
    conf : float
        Soglia di confidence per il rilevamento.
    frame_w : int
        Larghezza frame in pixel.
    frame_h : int
        Altezza frame in pixel.
    motion_regions : list or None
        Zone di movimento [(x1, y1, x2, y2), ...] o None per analisi completa.
    patches : list
        Patch sliding window pre-calcolate da get_window_patches.
    config : PipelineConfig
        Configurazione pipeline (enable_sliding_window, inference_scales,
        tta_augmentations, nms_iou_internal).

    Returns
    -------
    tuple (all_boxes, sw_hits, ms_hits)
    """
    all_boxes = []
    sw_hits = 0
    ms_hits = 0

    # Sliding window + NMS interna
    if config.enable_sliding_window and patches:
        sw_boxes, sw_hits = run_sliding_window(model, frame, patches, conf, motion_regions)
        sw_boxes = apply_nms(sw_boxes, config.nms_iou_internal)
        all_boxes.extend(sw_boxes)

    # Multi-scale + TTA + NMS interna
    ms_boxes, ms_hits = run_multiscale_inference(
        model, frame, config.inference_scales, config.tta_augmentations, conf, frame_w, frame_h
    )
    ms_boxes = apply_nms(ms_boxes, config.nms_iou_internal)
    all_boxes.extend(ms_boxes)

    return all_boxes, sw_hits, ms_hits
