"""
Modulo tracking per Person Anonymizer.

Contiene ByteTracker wrapper, TemporalSmoother
con EMA + ghost boxes per occlusioni temporanee.
"""

import logging

import numpy as np

from config import PipelineConfig

_log = logging.getLogger(__name__)


# ============================================================
# TRACKING BYTETRACK
# ============================================================


def create_tracker(fps, config: PipelineConfig):
    """
    Crea un'istanza di ByteTracker.

    Parameters
    ----------
    fps : float
        Frame rate del video sorgente.
    config : PipelineConfig
        Configurazione della pipeline; usa ``track_max_age`` e
        ``track_match_thresh``.

    Returns
    -------
    BYTETracker
        Istanza configurata del tracker.
    """
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
        track_buffer=max(config.track_max_age, int(fps * 2)),
        match_thresh=config.track_match_thresh,
    )
    return BYTETracker(tracker_args, frame_rate=int(fps))


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
    except Exception as e:
        _log.warning("Tracker update failed: %s", e, exc_info=True)
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
        except Exception as e:
            _log.debug("Failed to extract track data: %s", e, exc_info=True)
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
