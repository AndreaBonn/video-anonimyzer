#!/usr/bin/env python3
"""
Person Anonymizer Tool v7.1
Rileva persone in video di sorveglianza e le oscura con pixelation/blur.
Pipeline multi-strategia YOLO + revisione manuale OpenCV.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# CONFIGURAZIONE
# ============================================================

# --- MODALITA' OPERATIVA ---
OPERATION_MODE = "manual"

# --- OSCURAMENTO ---
ANONYMIZATION_METHOD = "pixelation"
ANONYMIZATION_INTENSITY = 10
PERSON_PADDING = 15
# Padding amplificato per persone ai bordi del frame (ingresso/uscita)
EDGE_PADDING_MULTIPLIER = 2.5
EDGE_THRESHOLD = 0.05  # 5% del frame considerato "bordo"

# --- RILEVAMENTO ---
# Confidence bassa (0.20): in anonimizzazione, recall >> precision.
# Una persona non oscurata è un rischio legale (GDPR), un falso positivo
# è solo un pezzo di muro pixelato. Il ByteTracker filtra il rumore.
DETECTION_CONFIDENCE = 0.20
# NMS interna per-strategia (più aggressiva)
NMS_IOU_INTERNAL = 0.45
# NMS finale cross-strategia (più permissiva)
NMS_IOU_THRESHOLD = 0.55
YOLO_MODEL = "yolov8x.pt"

# --- CORREZIONE FISH-EYE ---
ENABLE_FISHEYE_CORRECTION = True
CAMERA_MATRIX = None
DIST_COEFFICIENTS = None

# --- FRAME DIFFERENCING ---
ENABLE_MOTION_DETECTION = False
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 500
MOTION_PADDING = 60

# --- SLIDING WINDOW ---
ENABLE_SLIDING_WINDOW = True
SLIDING_WINDOW_GRID = 3
SLIDING_WINDOW_OVERLAP = 0.3

# --- MULTI-SCALE + TTA ---
INFERENCE_SCALES = [1.0, 1.5, 2.0, 2.5]
TTA_AUGMENTATIONS = ["flip_h"]

# --- MIGLIORAMENTO QUALITA' ---
QUALITY_CLAHE_CLIP = 2.0
QUALITY_CLAHE_GRID = (8, 8)
QUALITY_DARKNESS_THRESHOLD = 60  # CLAHE solo su frame con luminosità media < soglia

# --- TRACKING ---
ENABLE_TRACKING = True
TRACK_MAX_AGE = 45
TRACK_MATCH_THRESH = 0.6

# --- TEMPORAL SMOOTHING ---
ENABLE_TEMPORAL_SMOOTHING = True
SMOOTHING_ALPHA = 0.35  # Peso EMA (0-1): più alto = più reattivo, meno lag
GHOST_FRAMES = 10  # Frame di persistenza per persone temporaneamente occluse
GHOST_EXPANSION = 1.15  # Espansione del box ghost (15% per compensare movimento)

# --- INTENSITA' OSCURAMENTO ADATTIVA ---
ENABLE_ADAPTIVE_INTENSITY = True
ADAPTIVE_REFERENCE_HEIGHT = 80

# --- INTERPOLAZIONE SUB-FRAME ---
# DISABILITATA: la detection su frame blended (cv2.addWeighted) crea
# artefatti di ghosting che generano falsi positivi e bounding box
# imprecisi. Il tracking ByteTrack gestisce già la continuità temporale.
ENABLE_SUBFRAME_INTERPOLATION = False
INTERPOLATION_FPS_THRESHOLD = 15

# --- VERIFICA POST-RENDERING ---
ENABLE_POST_RENDER_CHECK = True
POST_RENDER_CHECK_CONFIDENCE = 0.45
MAX_REFINEMENT_PASSES = 3
REFINEMENT_OVERLAP_THRESHOLD = 0.5

# --- REVISIONE MANUALE (colori BGR) ---
REVIEW_AUTO_COLOR = (0, 255, 0)
REVIEW_MANUAL_COLOR = (0, 120, 255)
REVIEW_DRAWING_COLOR = (255, 255, 0)
REVIEW_FILL_ALPHA = 0.35
REVIEW_WINDOW_MAX_WIDTH = 1280

# --- OUTPUT E DEBUG ---
ENABLE_DEBUG_VIDEO = True
ENABLE_CONFIDENCE_REPORT = True

# ============================================================
# FORMATI VIDEO SUPPORTATI
# ============================================================
SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}

# ============================================================
# VERSIONE
# ============================================================
VERSION = "7.1"


# ============================================================
# UTILITA' PRE-PROCESSING
# ============================================================


def build_undistortion_maps(camera_matrix, dist_coefficients, frame_w, frame_h):
    """Costruisce le mappe di undistortion (una sola volta)."""
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefficients, (frame_w, frame_h), 1
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coefficients, None, new_matrix, (frame_w, frame_h), cv2.CV_16SC2
    )
    return map1, map2


def undistort_frame(frame, map1, map2):
    """Applica undistortion a un frame usando mappe pre-calcolate."""
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def enhance_frame(frame, clahe_obj, darkness_threshold):
    """
    Migliora il frame con CLAHE solo se necessario.

    CLAHE viene applicato solo su frame con luminosità media sotto la soglia,
    evitando di alterare frame ben illuminati che YOLO gestisce già bene.
    Lo sharpening è stato rimosso: crea artefatti (halos, amplificazione rumore)
    che peggiorano il riconoscimento di persone piccole.

    Parameters
    ----------
    frame : ndarray
        Frame BGR.
    clahe_obj : cv2.CLAHE
        Oggetto CLAHE pre-creato (riutilizzato tra frame).
    darkness_threshold : float
        Soglia luminosità media sotto la quale applicare CLAHE.

    Returns
    -------
    ndarray
        Frame migliorato (o originale se luminosità sufficiente).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= darkness_threshold:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = clahe_obj.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ============================================================
# MOTION DETECTION
# ============================================================


class MotionDetector:
    """Rileva zone di movimento tramite frame differencing."""

    def __init__(self, threshold, min_area, padding):
        self.prev_gray = None
        self.threshold = threshold
        self.min_area = min_area
        self.padding = padding

    def get_motion_regions(self, frame):
        """
        Restituisce lista di (x1, y1, x2, y2) con zone di movimento.
        None al primo frame (analisi completa).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            regions.append(
                (
                    max(0, x - self.padding),
                    max(0, y - self.padding),
                    x + w + self.padding,
                    y + h + self.padding,
                )
            )
        return regions


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
# INTERPOLAZIONE SUB-FRAME
# ============================================================


def interpolate_frames(frame_a, frame_b, n_steps):
    """Genera frame interpolati tra due frame consecutivi."""
    interpolated = []
    for i in range(1, n_steps + 1):
        alpha = i / (n_steps + 1)
        interp = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
        interpolated.append(interp)
    return interpolated


def should_interpolate(fps, threshold):
    """Verifica se l'interpolazione è necessaria."""
    return fps < threshold


# ============================================================
# TRACKING BYTETRACK
# ============================================================


def create_tracker(fps):
    """Crea un'istanza di ByteTracker."""
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
    except ImportError:
        from ultralytics.trackers import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

    # Parametri ottimizzati per sorveglianza:
    # - track_high_thresh alto: primo stage solo detection affidabili
    # - track_low_thresh basso: secondo stage recupera detection deboli
    #   SE matchano con track esistenti (riduce FN)
    # - buffer adattivo: 2 secondi indipendentemente dal fps
    tracker_args = IterableSimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.15,
        new_track_thresh=0.4,
        track_buffer=max(TRACK_MAX_AGE, int(fps * 2)),
        match_thresh=TRACK_MATCH_THRESH,
    )
    return BYTETracker(tracker_args, frame_rate=int(fps))


class SyntheticResults:
    """Bridge per fornire detections in formato compatibile con ByteTracker."""

    def __init__(self, boxes_with_conf, frame_shape):
        self.boxes_with_conf = boxes_with_conf
        self.frame_shape = frame_shape

    def to_tracker_format(self):
        """
        Restituisce array (N, 6): [x1, y1, x2, y2, conf, cls].
        """
        if not self.boxes_with_conf:
            return np.empty((0, 6), dtype=np.float32)
        arr = []
        for b in self.boxes_with_conf:
            arr.append([b[0], b[1], b[2], b[3], b[4], 0])
        return np.array(arr, dtype=np.float32)


def update_tracker(tracker, nms_boxes, frame_shape):
    """
    Aggiorna il tracker con i box NMS.

    Returns
    -------
    list of (track_id, x1, y1, x2, y2, conf)
    """
    if not nms_boxes:
        det = np.empty((0, 6), dtype=np.float32)
    else:
        det = []
        for b in nms_boxes:
            det.append([b[0], b[1], b[2], b[3], b[4], 0])
        det = np.array(det, dtype=np.float32)

    img_info = (frame_shape[0], frame_shape[1])
    img_size = (frame_shape[0], frame_shape[1])

    try:
        tracks = tracker.update(det, img_info, img_size)
    except Exception:
        # Fallback: restituisce i box senza tracking
        results = []
        for i, b in enumerate(nms_boxes):
            results.append((i, b[0], b[1], b[2], b[3], b[4]))
        return results

    results = []
    for t in tracks:
        try:
            tlbr = t.tlbr
            tid = t.track_id
            score = t.score
            results.append((tid, tlbr[0], tlbr[1], tlbr[2], tlbr[3], float(score)))
        except Exception:
            continue
    return results


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================


class TemporalSmoother:
    """
    Exponential Moving Average con ghost boxes per occlusioni temporanee.

    A differenza della media mobile semplice (lag di N/2 frame), l'EMA
    dà peso maggiore ai frame recenti. Quando una persona è temporaneamente
    occlusa, mantiene un "ghost box" leggermente espanso per continuare
    l'oscuramento, evitando gap nell'anonimizzazione.
    """

    def __init__(self, alpha, ghost_frames=10, ghost_expansion=1.15):
        self.alpha = alpha
        self.state = {}
        self.ghost_frames = ghost_frames
        self.ghost_expansion = ghost_expansion
        self.ghost_countdown = {}

    def smooth(self, track_id, x1, y1, x2, y2):
        """Applica EMA e restituisce coordinate smoothed."""
        curr = np.array([x1, y1, x2, y2], dtype=np.float64)
        if track_id not in self.state:
            self.state[track_id] = curr
        else:
            self.state[track_id] = self.alpha * curr + (1 - self.alpha) * self.state[track_id]
        # Reset ghost countdown se il track riappare
        self.ghost_countdown.pop(track_id, None)
        s = self.state[track_id]
        return int(s[0]), int(s[1]), int(s[2]), int(s[3])

    def get_ghost_boxes(self):
        """
        Restituisce box fantasma per track recentemente persi.

        Returns
        -------
        list of (track_id, x1, y1, x2, y2)
        """
        ghosts = []
        for tid in list(self.ghost_countdown):
            countdown = self.ghost_countdown[tid]
            if countdown > 0 and tid in self.state:
                s = self.state[tid]
                cx = (s[0] + s[2]) / 2
                cy = (s[1] + s[3]) / 2
                w = (s[2] - s[0]) * self.ghost_expansion
                h = (s[3] - s[1]) * self.ghost_expansion
                ghosts.append(
                    (tid, int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
                )
                self.ghost_countdown[tid] -= 1
            else:
                del self.ghost_countdown[tid]
                if tid in self.state:
                    del self.state[tid]
        return ghosts

    def clear_stale(self, active_ids):
        """Avvia ghost countdown per track non più attivi."""
        for tid in list(self.state):
            if tid not in active_ids and tid not in self.ghost_countdown:
                self.ghost_countdown[tid] = self.ghost_frames


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


def draw_debug_polygons(frame, auto_polygons, manual_polygons):
    """Disegna poligoni colorati per il video debug."""
    debug_frame = frame.copy()
    overlay = debug_frame.copy()
    for poly in auto_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], REVIEW_AUTO_COLOR)
        cv2.polylines(debug_frame, [pts], True, REVIEW_AUTO_COLOR, 2)
    for poly in manual_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], REVIEW_MANUAL_COLOR)
        cv2.polylines(debug_frame, [pts], True, REVIEW_MANUAL_COLOR, 2)
    cv2.addWeighted(overlay, REVIEW_FILL_ALPHA, debug_frame, 1 - REVIEW_FILL_ALPHA, 0, debug_frame)
    return debug_frame


# ============================================================
# REINTEGRO AUDIO
# ============================================================


def encode_with_audio(video_no_audio, original_video, output_path):
    """
    Codifica il video in H.264 (libx264) con qualità alta e reintegra l'audio.

    Usa CRF 18 (visually lossless) e preset 'slow' per massima qualità.
    Il codec mp4v usato da cv2.VideoWriter è MPEG-4 Part 2 (2001) e degrada
    significativamente la qualità; H.264 offre qualità nettamente superiore
    a parità di bitrate.
    """
    try:
        video_in = ffmpeg.input(video_no_audio)
        audio_in = ffmpeg.input(original_video).audio
        (
            ffmpeg.output(
                video_in,
                audio_in,
                output_path,
                vcodec="libx264",
                crf=18,
                preset="slow",
                pix_fmt="yuv420p",
                acodec="aac",
                audio_bitrate="192k",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        # Fallback: tenta senza audio
        try:
            video_in = ffmpeg.input(video_no_audio)
            (
                ffmpeg.output(
                    video_in,
                    output_path,
                    vcodec="libx264",
                    crf=18,
                    preset="slow",
                    pix_fmt="yuv420p",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            shutil.copy(video_no_audio, output_path)


def encode_without_audio(video_no_audio, output_path):
    """Codifica il video in H.264 senza audio."""
    try:
        video_in = ffmpeg.input(video_no_audio)
        (
            ffmpeg.output(
                video_in, output_path, vcodec="libx264", crf=18, preset="slow", pix_fmt="yuv420p"
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        shutil.copy(video_no_audio, output_path)


# ============================================================
# VERIFICA POST-RENDERING
# ============================================================


def run_post_render_check(anonymized_video_path, model, confidence, report_data, check_scales=None):
    """
    Secondo passaggio YOLO sul video oscurato, con multi-scale.

    Usa le stesse scale della detection originale per intercettare
    anche persone piccole che una singola scala non rileva.

    Parameters
    ----------
    anonymized_video_path : str
        Percorso video anonimizzato.
    model : YOLO
        Modello YOLO caricato.
    confidence : float
        Soglia di confidenza.
    report_data : dict
        Dati report per aggiornamento alerting.
    check_scales : list of float, optional
        Scale di verifica (default: [1.0, 2.0]).

    Returns
    -------
    list of tuple
        Ogni tupla: (frame_idx, count, nms_boxes) dove nms_boxes è la lista
        dei box [x1, y1, x2, y2, conf] rilevati nel frame.
    """
    if check_scales is None:
        check_scales = [1.0, 2.0]

    cap = cv2.VideoCapture(anonymized_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alert_frames = []
    frame_idx = 0

    pbar = tqdm(total=total, desc="Verifica", unit=" frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        all_boxes = []
        for scale in check_scales:
            if scale == 1.0:
                results = model(frame, conf=confidence, classes=[0], verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    all_boxes.append([x1, y1, x2, y2, float(box.conf[0])])
            else:
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                boxes = detect_and_rescale(model, scaled, confidence, scale)
                all_boxes.extend(boxes)

        nms_boxes = apply_nms(all_boxes, NMS_IOU_THRESHOLD)

        if len(nms_boxes) > 0:
            alert_frames.append((frame_idx, len(nms_boxes), nms_boxes))
            if frame_idx in report_data:
                report_data[frame_idx]["post_check_alerts"] = len(nms_boxes)

        frame_idx += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    return alert_frames


def filter_artifact_detections(alert_frames, annotations, iou_threshold):
    """
    Filtra rilevamenti post-render che sono artefatti della pixelazione.

    Un rilevamento che si sovrappone (IoU >= soglia) con un'area già
    anonimizzata è quasi certamente un artefatto: YOLO interpreta la
    pixelazione come una persona. I rilevamenti genuini sono quelli che
    NON si sovrappongono significativamente con aree già oscurate.

    Parameters
    ----------
    alert_frames : list of tuple
        Ogni tupla: (frame_idx, count, nms_boxes).
    annotations : dict
        Annotazioni correnti {frame_idx: {"auto": [...], "manual": [...]}}.
    iou_threshold : float
        Soglia IoU per considerare un rilevamento come artefatto.

    Returns
    -------
    genuine_alerts : list of tuple
        (frame_idx, genuine_boxes) solo frame con rilevamenti genuini.
    total_artifacts : int
        Numero totale di artefatti filtrati.
    total_genuine : int
        Numero totale di rilevamenti genuini.
    """
    genuine_alerts = []
    total_artifacts = 0
    total_genuine = 0

    for frame_idx, count, nms_boxes in alert_frames:
        ann = annotations.get(frame_idx, {"auto": [], "manual": []})
        existing_bboxes = []
        for poly in ann.get("auto", []):
            existing_bboxes.append(polygon_to_bbox(poly))
        for poly in ann.get("manual", []):
            existing_bboxes.append(polygon_to_bbox(poly))

        genuine_boxes = []
        for det_box in nms_boxes:
            det_bbox = det_box[:4]
            is_artifact = False
            for ex_bbox in existing_bboxes:
                if compute_iou_boxes(det_bbox, ex_bbox) >= iou_threshold:
                    is_artifact = True
                    break
            if is_artifact:
                total_artifacts += 1
            else:
                genuine_boxes.append(det_box)
                total_genuine += 1

        if genuine_boxes:
            genuine_alerts.append((frame_idx, genuine_boxes))

    return genuine_alerts, total_artifacts, total_genuine


# ============================================================
# BOX -> POLIGONO
# ============================================================


def box_to_polygon(x1, y1, x2, y2, padding=0, frame_w=None, frame_h=None):
    """
    Converte bounding box in poligono 4 punti con padding direzionale.

    Applica padding maggiorato quando il box è vicino ai bordi del frame,
    per coprire persone parzialmente visibili in ingresso/uscita dalla scena.
    Il clamping avviene DOPO il padding.
    """
    if frame_w and frame_h:
        edge_x = int(frame_w * EDGE_THRESHOLD)
        edge_y = int(frame_h * EDGE_THRESHOLD)
        pad_l = int(padding * EDGE_PADDING_MULTIPLIER) if x1 < edge_x else padding
        pad_t = int(padding * EDGE_PADDING_MULTIPLIER) if y1 < edge_y else padding
        pad_r = int(padding * EDGE_PADDING_MULTIPLIER) if frame_w - x2 < edge_x else padding
        pad_b = int(padding * EDGE_PADDING_MULTIPLIER) if frame_h - y2 < edge_y else padding
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


# ============================================================
# NORMALIZZAZIONE ANNOTAZIONI
# ============================================================


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
    Merge iterativo di rettangoli sovrapposti.

    Parameters
    ----------
    rects : list of tuple
        Lista di rettangoli (x, y, w, h).

    Returns
    -------
    list of tuple
        Rettangoli non sovrapposti dopo il merge.
    """
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _rects_overlap(current, merged[j]):
                    current = _merge_rects(current, merged[j])
                    used[j] = True
                    changed = True
            new_merged.append(current)
        merged = new_merged
    return merged


def normalize_annotations(annotations):
    """
    Normalizza le annotazioni: converte poligoni in rettangoli
    e unifica le aree sovrapposte.

    Parameters
    ----------
    annotations : dict
        Dizionario {frame_idx: {"auto": [...], "manual": [...], "intensities": [...]}}.

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
            if ENABLE_ADAPTIVE_INTENSITY:
                intensity = compute_adaptive_intensity(
                    h, ANONYMIZATION_INTENSITY, ADAPTIVE_REFERENCE_HEIGHT
                )
            else:
                intensity = ANONYMIZATION_INTENSITY
            new_intensities.append(intensity)

        normalized[fidx] = {"auto": new_polys, "manual": [], "intensities": new_intensities}

    return normalized, (total_before, total_after)


# ============================================================
# DETECTION COMPLETA SU UN FRAME
# ============================================================


def run_full_detection(model, frame, conf, frame_w, frame_h, motion_regions, patches):
    """
    Esegue rilevamento sliding window + multi-scale + TTA su un frame.

    NMS a due stadi: prima interna per-strategia (deduplicazione),
    poi le box vengono combinate per la NMS finale cross-strategia.

    Returns
    -------
    tuple (all_boxes, sw_hits, ms_hits)
    """
    all_boxes = []
    sw_hits = 0
    ms_hits = 0

    # Sliding window + NMS interna
    if ENABLE_SLIDING_WINDOW and patches:
        sw_boxes, sw_hits = run_sliding_window(model, frame, patches, conf, motion_regions)
        sw_boxes = apply_nms(sw_boxes, NMS_IOU_INTERNAL)
        all_boxes.extend(sw_boxes)

    # Multi-scale + TTA + NMS interna
    ms_boxes, ms_hits = run_multiscale_inference(
        model, frame, INFERENCE_SCALES, TTA_AUGMENTATIONS, conf, frame_w, frame_h
    )
    ms_boxes = apply_nms(ms_boxes, NMS_IOU_INTERNAL)
    all_boxes.extend(ms_boxes)

    return all_boxes, sw_hits, ms_hits


# ============================================================
# RENDERING VIDEO
# ============================================================


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
            if ENABLE_ADAPTIVE_INTENSITY and i < len(intensities):
                intensity = intensities[i]
            else:
                intensity = ANONYMIZATION_INTENSITY
            if method == "blur" and intensity % 2 == 0:
                intensity += 1
            render_frame = obscure_polygon(render_frame, poly, method, intensity)

        for poly in manual_polys:
            intensity = ANONYMIZATION_INTENSITY
            if method == "blur" and intensity % 2 == 0:
                intensity += 1
            render_frame = obscure_polygon(render_frame, poly, method, intensity)

        out_writer.write(render_frame)

        if debug_writer:
            debug_frame = draw_debug_polygons(frame, auto_polys, manual_polys)
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


# ============================================================
# PIPELINE PRINCIPALE
# ============================================================


def run_pipeline(args):
    """Esegue la pipeline completa di anonimizzazione."""

    input_path = args.input
    mode = args.mode or OPERATION_MODE
    method = args.method or ANONYMIZATION_METHOD
    enable_debug = not args.no_debug
    enable_report = not args.no_report
    review_json = args.review

    # --- Validazione --normalize ---
    if args.normalize and not review_json:
        print("Errore: --normalize richiede --review <json>")
        sys.exit(1)

    # --- Validazione input ---
    if not os.path.isfile(input_path):
        print(f"Errore: file non trovato: {input_path}")
        sys.exit(1)

    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Errore: formato non supportato '{ext}'.")
        print(f"  Formati supportati: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    # Verifica ffmpeg
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        print("ATTENZIONE: ffmpeg non trovato nel PATH.")
        print("  L'audio non verrà preservato nel video di output.")
        print("  Installare ffmpeg: sudo apt install ffmpeg")

    # --- Output paths ---
    input_stem = Path(input_path).stem
    input_dir = Path(input_path).parent

    if args.output:
        output_path = args.output
        output_dir = Path(output_path).parent
    else:
        output_dir = input_dir
        output_path = str(output_dir / f"{input_stem}_anonymized.mp4")

    temp_video_path = str(output_dir / f"{input_stem}_temp_noaudio.avi")
    temp_debug_path = str(output_dir / f"{input_stem}_temp_debug.avi")
    debug_path = str(output_dir / f"{input_stem}_debug.mp4")
    report_path = str(output_dir / f"{input_stem}_report.csv")
    json_path = str(output_dir / f"{input_stem}_annotations.json")

    # --- Apertura video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0
    if total_frames <= 0:
        print("Errore: impossibile determinare il numero di frame.")
        sys.exit(1)

    # Risoluzione label
    res_label = f"{frame_w}x{frame_h}"
    if frame_h >= 2160:
        res_label = "4K"
    elif frame_h >= 1080:
        res_label = "1080p"
    elif frame_h >= 720:
        res_label = "720p"
    elif frame_h >= 480:
        res_label = "480p"

    # --- Fish-eye auto-disable ---
    fisheye_enabled = ENABLE_FISHEYE_CORRECTION
    if fisheye_enabled and (CAMERA_MATRIX is None or DIST_COEFFICIENTS is None):
        fisheye_enabled = False

    # --- Interpolazione auto-disable ---
    do_interpolation = ENABLE_SUBFRAME_INTERPOLATION and should_interpolate(
        fps, INTERPOLATION_FPS_THRESHOLD
    )
    if do_interpolation and fps < 3:
        print(
            "ATTENZIONE: fps molto basso (< 3). L'interpolazione potrebbe "
            "produrre artefatti. Considera di disabilitarla."
        )

    # --- Intensità label ---
    if ENABLE_ADAPTIVE_INTENSITY:
        method_label = (
            f"{method} adattivo "
            f"(base: {ANONYMIZATION_INTENSITY}px, "
            f"ref: {ADAPTIVE_REFERENCE_HEIGHT}px)"
        )
    else:
        method_label = f"{method} ({ANONYMIZATION_INTENSITY}px)"

    # --- Header console ---
    print(f"\nPerson Anonymizer v{VERSION}")
    print("-" * 40)
    print(
        f"Input:          {Path(input_path).name}  "
        f"({total_frames} frame, {fps:.0f}fps, {res_label})"
    )
    print(f"Output:         {Path(output_path).name}")
    print(f"Modalita':      {mode}")
    print(f"Metodo:         {method_label}")
    print(f"Modello:        {YOLO_MODEL}  |  Confidenza: {DETECTION_CONFIDENCE}")
    print(
        f"Scale:          [{', '.join(f'{s}x' for s in INFERENCE_SCALES)}]"
        f" + {', '.join(TTA_AUGMENTATIONS)}"
    )
    sw_status = (
        f"griglia {SLIDING_WINDOW_GRID}x{SLIDING_WINDOW_GRID}, "
        f"overlap {int(SLIDING_WINDOW_OVERLAP * 100)}%"
        if ENABLE_SLIDING_WINDOW
        else "disabilitato"
    )
    print(f"Sliding window: {sw_status}")
    print(f"Fish-eye:       {'abilitato' if fisheye_enabled else 'disabilitato'}")
    print(
        f"Tracking:       ByteTrack (max_age: {TRACK_MAX_AGE})"
        if ENABLE_TRACKING
        else "Tracking:       disabilitato"
    )
    print(
        f"Smoothing:      EMA (alpha: {SMOOTHING_ALPHA})"
        if ENABLE_TEMPORAL_SMOOTHING
        else "Smoothing:      disabilitato"
    )
    interp_status = (
        "disabilitata (fps >= " f"{INTERPOLATION_FPS_THRESHOLD})"
        if not do_interpolation
        else f"abilitata ({fps:.0f}fps < {INTERPOLATION_FPS_THRESHOLD}fps)"
    )
    print(f"Interpolazione: {interp_status}")
    print("-" * 40)

    # --- Caricamento modello YOLO ---
    print(f"\nCaricamento modello {YOLO_MODEL}...")
    model = YOLO(YOLO_MODEL)

    # --- Preparazione componenti ---
    # Fish-eye maps
    undist_map1, undist_map2 = None, None
    if fisheye_enabled:
        undist_map1, undist_map2 = build_undistortion_maps(
            CAMERA_MATRIX, DIST_COEFFICIENTS, frame_w, frame_h
        )

    # CLAHE object (creato una sola volta, riutilizzato per ogni frame)
    clahe_obj = cv2.createCLAHE(clipLimit=QUALITY_CLAHE_CLIP, tileGridSize=QUALITY_CLAHE_GRID)

    # Motion detector
    motion_detector = None
    if ENABLE_MOTION_DETECTION:
        motion_detector = MotionDetector(MOTION_THRESHOLD, MOTION_MIN_AREA, MOTION_PADDING)

    # Sliding window patches
    patches = []
    if ENABLE_SLIDING_WINDOW:
        patches = get_window_patches(frame_w, frame_h, SLIDING_WINDOW_GRID, SLIDING_WINDOW_OVERLAP)

    # Tracker
    tracker = None
    if ENABLE_TRACKING:
        tracker = create_tracker(fps)

    # Smoother (EMA + ghost boxes)
    smoother = None
    if ENABLE_TEMPORAL_SMOOTHING:
        smoother = TemporalSmoother(SMOOTHING_ALPHA, GHOST_FRAMES, GHOST_EXPANSION)

    # Struttura annotazioni
    annotations = {}
    report_data = {}

    # ============================================
    # FASE 1 — RILEVAMENTO AUTOMATICO (o caricamento JSON)
    # ============================================
    start_time = time.time()

    if review_json:
        # --- Caricamento da JSON esistente ---
        if not os.path.isfile(review_json):
            print(f"Errore: file JSON non trovato: {review_json}")
            sys.exit(1)

        print(f"\n[FASE 1/5] Caricamento annotazioni da {Path(review_json).name}...")
        with open(review_json) as f:
            json_data = json.load(f)

        for fidx_str, frame_ann in json_data.get("frames", {}).items():
            fidx = int(fidx_str)
            auto_polys = [[tuple(pt) for pt in poly] for poly in frame_ann.get("auto", [])]
            manual_polys = [[tuple(pt) for pt in poly] for poly in frame_ann.get("manual", [])]
            # Ricalcola intensità per i poligoni auto
            intensities = []
            for poly in auto_polys:
                ys = [pt[1] for pt in poly]
                box_h = max(ys) - min(ys) if ys else 0
                if ENABLE_ADAPTIVE_INTENSITY:
                    intensities.append(
                        compute_adaptive_intensity(
                            box_h, ANONYMIZATION_INTENSITY, ADAPTIVE_REFERENCE_HEIGHT
                        )
                    )
                else:
                    intensities.append(ANONYMIZATION_INTENSITY)
            annotations[fidx] = {
                "auto": auto_polys,
                "manual": manual_polys,
                "intensities": intensities,
            }

        total_polys = sum(
            len(a.get("auto", [])) + len(a.get("manual", [])) for a in annotations.values()
        )
        print(
            f"\n  Annotazioni caricate: {len(annotations)} frame, " f"{total_polys} poligoni totali"
        )

        if args.normalize:
            # Normalizzazione: converti poligoni in rettangoli e unifica
            print(f"\n  Normalizzazione annotazioni...")
            annotations, (n_before, n_after) = normalize_annotations(annotations)
            print(f"  Poligoni prima:  {n_before}")
            print(f"  Rettangoli dopo: {n_after}  " f"(riduzione: {n_before - n_after})")
            # Salta la revisione manuale, vai direttamente al rendering
            mode = "auto"
        else:
            # Forza modalità manual per la revisione
            mode = "manual"
        cap.release()

    else:
        # --- Detection automatica ---
        # Statistiche
        unique_track_ids = set()
        total_instances = 0
        frames_zero_det = 0
        all_confidences = []

        print(f"\n[FASE 1/5] Rilevamento automatico...")

        prev_frame_for_interp = None

        pbar = tqdm(total=total_frames, desc="Elaborazione", unit=" frame")
        frame_idx = 0

        corrupted_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                # Distingui fine video da frame corrotto
                if frame_idx < total_frames - 1:
                    corrupted_frames.append(frame_idx)
                    frame_idx += 1
                    pbar.update(1)
                    continue
                break

            original_frame = frame.copy()

            # [A] Correzione fish-eye
            if fisheye_enabled:
                frame = undistort_frame(frame, undist_map1, undist_map2)

            # [B] Miglioramento qualità (CLAHE condizionale, no sharpening)
            enhanced = enhance_frame(frame, clahe_obj, QUALITY_DARKNESS_THRESHOLD)

            # [C] Frame differencing
            motion_regions = None
            if motion_detector:
                motion_regions = motion_detector.get_motion_regions(enhanced)

            # Interpolazione sub-frame
            all_boxes = []
            sw_hits_total = 0
            ms_hits_total = 0

            if do_interpolation and prev_frame_for_interp is not None:
                n_interp = max(1, int(INTERPOLATION_FPS_THRESHOLD / fps) - 1)
                virtual_frames = interpolate_frames(prev_frame_for_interp, enhanced, n_interp)
                for vf in virtual_frames:
                    vf_boxes, _, _ = run_full_detection(
                        model, vf, DETECTION_CONFIDENCE, frame_w, frame_h, motion_regions, patches
                    )
                    all_boxes.extend(vf_boxes)

            prev_frame_for_interp = enhanced.copy()

            # [D] Rilevamento multi-strategia sul frame reale
            real_boxes, sw_hits, ms_hits = run_full_detection(
                model, enhanced, DETECTION_CONFIDENCE, frame_w, frame_h, motion_regions, patches
            )
            all_boxes.extend(real_boxes)
            sw_hits_total += sw_hits
            ms_hits_total += ms_hits

            # [E] NMS
            nms_boxes = apply_nms(all_boxes, NMS_IOU_THRESHOLD)

            # [F] Tracking
            tracked = []
            if ENABLE_TRACKING and tracker is not None:
                tracked = update_tracker(tracker, nms_boxes, (frame_h, frame_w, 3))
            else:
                for i, b in enumerate(nms_boxes):
                    tracked.append((i, b[0], b[1], b[2], b[3], b[4]))

            # [G] Temporal Smoothing + [H] Intensità adattiva
            frame_polygons = []
            frame_intensities = []
            active_ids = set()

            for tid, x1, y1, x2, y2, conf in tracked:
                active_ids.add(tid)
                unique_track_ids.add(tid)

                if ENABLE_TEMPORAL_SMOOTHING and smoother is not None:
                    x1, y1, x2, y2 = smoother.smooth(tid, x1, y1, x2, y2)

                # Clamp to frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                box_h = y2 - y1
                if ENABLE_ADAPTIVE_INTENSITY:
                    intensity = compute_adaptive_intensity(
                        box_h, ANONYMIZATION_INTENSITY, ADAPTIVE_REFERENCE_HEIGHT
                    )
                else:
                    intensity = ANONYMIZATION_INTENSITY

                polygon = box_to_polygon(x1, y1, x2, y2, PERSON_PADDING, frame_w, frame_h)
                frame_polygons.append(polygon)
                frame_intensities.append(intensity)

                all_confidences.append(conf)

            if smoother:
                smoother.clear_stale(active_ids)

                # Ghost boxes: oscura posizioni previste per track persi
                for gtid, gx1, gy1, gx2, gy2 in smoother.get_ghost_boxes():
                    gx1 = max(0, gx1)
                    gy1 = max(0, gy1)
                    gx2 = min(frame_w, gx2)
                    gy2 = min(frame_h, gy2)
                    ghost_h = gy2 - gy1
                    if ghost_h <= 0:
                        continue
                    if ENABLE_ADAPTIVE_INTENSITY:
                        g_intensity = compute_adaptive_intensity(
                            ghost_h, ANONYMIZATION_INTENSITY, ADAPTIVE_REFERENCE_HEIGHT
                        )
                    else:
                        g_intensity = ANONYMIZATION_INTENSITY
                    ghost_poly = box_to_polygon(
                        gx1, gy1, gx2, gy2, PERSON_PADDING, frame_w, frame_h
                    )
                    frame_polygons.append(ghost_poly)
                    frame_intensities.append(g_intensity)

            # [I] Salvataggio annotazioni
            annotations[frame_idx] = {
                "auto": frame_polygons,
                "manual": [],
                "intensities": frame_intensities,
            }

            # [J] Statistiche
            n_det = len(frame_polygons)
            total_instances += n_det
            if n_det == 0:
                frames_zero_det += 1

            confs = [t[5] for t in tracked]
            report_data[frame_idx] = {
                "frame_number": frame_idx,
                "persons_detected": n_det,
                "avg_confidence": float(np.mean(confs)) if confs else 0.0,
                "min_confidence": float(min(confs)) if confs else 0.0,
                "max_confidence": float(max(confs)) if confs else 0.0,
                "motion_zones": len(motion_regions) if motion_regions else 0,
                "sliding_window_hits": sw_hits_total,
                "multiscale_hits": ms_hits_total,
                "post_check_alerts": 0,
            }

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Statistiche Fase 1
        avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
        zero_pct = (frames_zero_det / total_frames * 100) if total_frames > 0 else 0

        print(f"\n  Persone tracciate (ID unici):   {len(unique_track_ids)}")
        print(f"  Istanze totali rilevate:        {total_instances:,}")
        print(f"  Frame con 0 rilevamenti:        {frames_zero_det}  ({zero_pct:.1f}%)")
        print(f"  Confidenza media:               {avg_conf:.2f}")
        if corrupted_frames:
            print(
                f"  ATTENZIONE: {len(corrupted_frames)} frame corrotti "
                f"(saltati): {corrupted_frames[:10]}"
                f"{'...' if len(corrupted_frames) > 10 else ''}"
            )

    # ============================================
    # FASE 2 — AUTO REFINEMENT LOOP
    # ============================================
    actual_refinement_passes = 0
    refinement_annotations_added = 0
    review_stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}

    if ENABLE_POST_RENDER_CHECK:
        for pass_num in range(1, MAX_REFINEMENT_PASSES + 1):
            actual_refinement_passes = pass_num
            pass_label = f"pass {pass_num}/{MAX_REFINEMENT_PASSES}"

            # --- Rendering (sempre dal video originale, senza debug) ---
            print(f"\n[FASE 2/5] Auto refinement — rendering ({pass_label})...")
            render_video(
                input_path,
                temp_video_path,
                annotations,
                fps,
                frame_w,
                frame_h,
                method,
                fisheye_enabled,
                undist_map1,
                undist_map2,
                desc=f"Rendering ({pass_label})",
            )

            # --- Verifica post-rendering ---
            print(f"\n  Verifica post-rendering ({pass_label})...")
            alert_frames = run_post_render_check(
                temp_video_path, model, POST_RENDER_CHECK_CONFIDENCE, report_data
            )

            if not alert_frames:
                print(f"\n  Nessun rilevamento residuo — rendering OK.")
                break

            # --- Filtraggio artefatti ---
            genuine_alerts, n_artifacts, n_genuine = filter_artifact_detections(
                alert_frames, annotations, REFINEMENT_OVERLAP_THRESHOLD
            )

            print(f"\n  Rilevamenti post-render: {n_artifacts + n_genuine}")
            print(
                f"  Artefatti filtrati (IoU >= {REFINEMENT_OVERLAP_THRESHOLD}): " f"{n_artifacts}"
            )
            print(f"  Residui genuini: {n_genuine}")

            if not genuine_alerts:
                print(f"  Tutti i rilevamenti sono artefatti della pixelazione " f"— rendering OK.")
                break

            if pass_num == MAX_REFINEMENT_PASSES:
                print(f"\n  Raggiunto limite di {MAX_REFINEMENT_PASSES} pass.")
                print(f"  Residui genuini rimasti in {len(genuine_alerts)} frame:")
                for fidx, boxes in genuine_alerts:
                    print(f"    Frame {fidx}: {len(boxes)} persona/e")
                if mode == "manual":
                    print("  -> Verranno mostrati nella revisione manuale.")
                else:
                    print("  -> Rieseguire con --mode manual per correzione.")
                break

            # --- Aggiunta residui genuini alle annotazioni ---
            added_this_pass = 0
            for fidx, boxes in genuine_alerts:
                if fidx not in annotations:
                    annotations[fidx] = {"auto": [], "manual": [], "intensities": []}
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    poly = box_to_polygon(
                        x1, y1, x2, y2, padding=PERSON_PADDING, frame_w=frame_w, frame_h=frame_h
                    )
                    annotations[fidx]["auto"].append(poly)
                    if ENABLE_ADAPTIVE_INTENSITY:
                        box_h = y2 - y1
                        inten = compute_adaptive_intensity(
                            box_h, ANONYMIZATION_INTENSITY, ADAPTIVE_REFERENCE_HEIGHT
                        )
                        annotations[fidx]["intensities"].append(inten)
                    added_this_pass += 1

            refinement_annotations_added += added_this_pass
            print(f"\n  Aggiunte {added_this_pass} annotazioni — " f"ri-rendering in corso...")
    else:
        print(f"\n[FASE 2/5] Auto refinement — saltato (verifica disabilitata)")

    # ============================================
    # FASE 3 — REVISIONE MANUALE
    # ============================================
    if mode == "manual":
        web_review_state = getattr(args, "_review_state", None)
        if web_review_state is not None:
            # --- REVIEW VIA WEB ---
            print(f"\n[FASE 3/5] Revisione manuale — in attesa di conferma dal browser...")

            web_review_state.setup(
                input_path,
                annotations,
                total_frames,
                frame_w,
                frame_h,
                fps,
                fisheye_enabled,
                undist_map1,
                undist_map2,
            )

            sse_mgr = getattr(args, "_sse_manager")
            web_job_id = getattr(args, "_job_id")
            sse_mgr.emit(
                web_job_id,
                "review_ready",
                {
                    "total_frames": total_frames,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "fps": fps,
                },
            )

            # Blocca fino a conferma utente
            original_annotations = {
                fidx: {
                    "auto": list(fdata.get("auto", [])),
                    "manual": list(fdata.get("manual", [])),
                }
                for fidx, fdata in annotations.items()
            }
            annotations = web_review_state.wait_for_completion()
            review_stats = _compute_review_stats(original_annotations, annotations, total_frames)

            print(f"\n  Revisione completata:")
            print(f"  Poligoni aggiunti:     {review_stats['added']}")
            print(f"  Poligoni rimossi:      {review_stats['removed']}")
            print(
                f"  Frame modificati:      {review_stats['frames_modified']}  "
                f"({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
            )
        else:
            # --- REVIEW CLI (OpenCV nativo) ---
            print(f"\n[FASE 3/5] Revisione manuale — apertura interfaccia...")
            print(
                "  -> Usa Spazio per navigare, Click per disegnare, "
                "D per eliminare, Q per confermare."
            )

            from manual_reviewer import run_manual_review

            config = {
                "auto_color": REVIEW_AUTO_COLOR,
                "manual_color": REVIEW_MANUAL_COLOR,
                "drawing_color": REVIEW_DRAWING_COLOR,
                "fill_alpha": REVIEW_FILL_ALPHA,
                "max_width": REVIEW_WINDOW_MAX_WIDTH,
            }
            annotations, review_stats = run_manual_review(
                input_path, annotations, config, fisheye_enabled, undist_map1, undist_map2
            )

            print(f"\n  Revisione completata:")
            print(
                f"  Frame revisionati:     " f"{review_stats['frames_reviewed']} / {total_frames}"
            )
            print(f"  Poligoni aggiunti:     {review_stats['added']}")
            print(f"  Poligoni rimossi:      {review_stats['removed']}")
            print(
                f"  Frame modificati:      {review_stats['frames_modified']}  "
                f"({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
            )
    else:
        print(f"\n[FASE 3/5] Revisione manuale — saltata (modalita' auto)")

    # ============================================
    # FASE 4 — RENDERING FINALE
    # ============================================
    print(f"\n[FASE 4/5] Rendering finale...")
    render_video(
        input_path,
        temp_video_path,
        annotations,
        fps,
        frame_w,
        frame_h,
        method,
        fisheye_enabled,
        undist_map1,
        undist_map2,
        debug_path=temp_debug_path if enable_debug else None,
        desc="Rendering finale",
    )

    # ============================================
    # FASE 5 — POST-PROCESSING
    # ============================================
    print(f"\n[FASE 5/5] Post-processing...")

    try:
        # Encoding H.264 + reintegro audio
        if ffmpeg_available:
            encode_with_audio(temp_video_path, input_path, output_path)
            # Encoding debug video in H.264
            if enable_debug and os.path.exists(temp_debug_path):
                encode_without_audio(temp_debug_path, debug_path)
        else:
            shutil.copy(temp_video_path, output_path)
            if enable_debug and os.path.exists(temp_debug_path):
                shutil.copy(temp_debug_path, debug_path)

        # Report CSV
        if enable_report:
            with open(report_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_number",
                        "persons_detected",
                        "avg_confidence",
                        "min_confidence",
                        "max_confidence",
                        "motion_zones",
                        "sliding_window_hits",
                        "multiscale_hits",
                        "post_check_alerts",
                    ],
                )
                writer.writeheader()
                for fidx in sorted(report_data.keys()):
                    writer.writerow(report_data[fidx])

        # JSON annotazioni (schema arricchito per riproducibilità)
        if mode == "manual" or args.normalize:
            json_annotations = {
                "schema_version": "2.0",
                "tool_version": VERSION,
                "video": {
                    "filename": Path(input_path).name,
                    "total_frames": total_frames,
                    "fps": fps,
                    "resolution": [frame_w, frame_h],
                },
                "pipeline_config": {
                    "yolo_model": YOLO_MODEL,
                    "detection_confidence": DETECTION_CONFIDENCE,
                    "nms_iou_threshold": NMS_IOU_THRESHOLD,
                    "nms_iou_internal": NMS_IOU_INTERNAL,
                    "inference_scales": INFERENCE_SCALES,
                    "sliding_window_grid": SLIDING_WINDOW_GRID,
                    "padding": PERSON_PADDING,
                    "anonymization_method": method,
                    "base_intensity": ANONYMIZATION_INTENSITY,
                    "adaptive_reference_height": ADAPTIVE_REFERENCE_HEIGHT,
                    "smoothing_alpha": SMOOTHING_ALPHA,
                    "ghost_frames": GHOST_FRAMES,
                },
                "refinement": {
                    "max_passes": MAX_REFINEMENT_PASSES,
                    "actual_passes": actual_refinement_passes,
                    "annotations_added": refinement_annotations_added,
                    "overlap_threshold": REFINEMENT_OVERLAP_THRESHOLD,
                },
                "mode": mode,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "review_stats": review_stats,
                "frames": {},
            }
            for fidx, ann in annotations.items():
                auto_list = [[list(pt) for pt in poly] for poly in ann.get("auto", [])]
                manual_list = [[list(pt) for pt in poly] for poly in ann.get("manual", [])]
                frame_data = {"auto": auto_list, "manual": manual_list}
                # Salva anche le intensità per evitare ricalcolo
                intensities = ann.get("intensities", [])
                if intensities:
                    frame_data["intensities"] = intensities
                json_annotations["frames"][str(fidx)] = frame_data
            with open(json_path, "w") as f:
                json.dump(json_annotations, f, indent=2)

    finally:
        # Cleanup temp files (FFV1 lossless, possono essere grandi)
        for tmp in (temp_video_path, temp_debug_path):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError as e:
                print(f"  ATTENZIONE: impossibile rimuovere {tmp}: {e}")

    # --- Riepilogo finale ---
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print("-" * 40)
    print(f"Completato  —  Tempo totale: {minutes}m {seconds}s")
    print(f"  Output:       {Path(output_path).name}")
    if enable_debug:
        print(f"  Debug:        {Path(debug_path).name}")
    if enable_report:
        print(f"  Report:       {Path(report_path).name}")
    if mode == "manual" or args.normalize:
        print(f"  Annotazioni:  {Path(json_path).name}")
    print()


# ============================================================
# CLI
# ============================================================


def parse_args():
    """Parser argomenti CLI."""
    parser = argparse.ArgumentParser(
        description=f"Person Anonymizer v{VERSION} — "
        "Oscuramento automatico persone in video di sorveglianza",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Percorso del video da elaborare")
    parser.add_argument(
        "-M",
        "--mode",
        choices=["manual", "auto"],
        default=None,
        help="Modalita' operativa (default: da config)",
    )
    parser.add_argument("-o", "--output", default=None, help="Percorso file di output")
    parser.add_argument(
        "-m",
        "--method",
        choices=["pixelation", "blur"],
        default=None,
        help="Metodo di oscuramento (default: da config)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disabilita video debug")
    parser.add_argument("--no-report", action="store_true", help="Disabilita CSV report")
    parser.add_argument(
        "--review",
        default=None,
        help="Ricarica annotazioni da JSON esistente, "
        "salta la detection e apre solo la revisione manuale",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalizza i poligoni in rettangoli e unifica "
        "le aree sovrapposte. Richiede --review.",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente (Ctrl+C).")
        sys.exit(1)


if __name__ == "__main__":
    main()
