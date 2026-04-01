"""
Modulo di pre-processing frame per la pipeline di anonimizzazione.

Contiene le funzioni di correzione fish-eye, miglioramento qualità CLAHE,
rilevamento del movimento tramite frame differencing, e utilità per
l'interpolazione sub-frame.
"""

import cv2
import numpy as np

# ============================================================
# UTILITÀ PRE-PROCESSING
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
