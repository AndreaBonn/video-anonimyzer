"""
Modulo tracking per Person Anonymizer.

Contiene ByteTracker wrapper, TemporalSmoother
con EMA + ghost boxes per occlusioni temporanee.
"""

import itertools
import logging

import numpy as np

from .config import PipelineConfig
from .detection import compute_iou_boxes

__all__ = ["PersonTracker", "create_tracker", "update_tracker", "TemporalSmoother"]

_log = logging.getLogger(__name__)

# Minimum IoU to consider a detection the same person as a track or as a
# previous-frame detection. Consecutive frames of the same person overlap far
# more than this; two adjacent people rarely do.
ASSOCIATION_IOU = 0.3

# Column of the track id in the rows returned by BYTETracker.update
# ([x1, y1, x2, y2, track_id, score, cls, idx]).
_TRACK_ID_COLUMN = 4


class PersonTracker:
    """
    ByteTrack wrapper that never drops a detection.

    ByteTrack returns only confirmed tracks: a person seen for the first time,
    or detected below its new-track threshold, would be missing from its
    output and therefore left unobscured. This wrapper returns every input
    detection with its original coordinates and uses ByteTrack only to assign
    identities. Detections no track claims get a negative fallback id,
    carried over from the overlapping detection of the previous frame.

    Parameters
    ----------
    byte_tracker : BYTETracker
        Configured ultralytics tracker.
    boxes_cls : type
        ``ultralytics.engine.results.Boxes``, the input type BYTETracker reads.
    """

    def __init__(self, byte_tracker, boxes_cls):
        self.byte_tracker = byte_tracker
        self._boxes_cls = boxes_cls
        self._fallback_ids = itertools.count(start=-1, step=-1)
        self._prev_fallback = []
        # ByteTrack id -> fallback id the person had before being confirmed
        self._aliases = {}
        self._known_track_ids = set()
        self._failure_logged = False

    def update(self, nms_boxes, frame_shape):
        """
        Assign a track id to each detection.

        Parameters
        ----------
        nms_boxes : list of [x1, y1, x2, y2, conf]
            Detections after NMS.
        frame_shape : tuple
            Frame shape ``(h, w, ...)``.

        Returns
        -------
        list of (track_id, x1, y1, x2, y2, conf)
            One entry per detection, in input order.
        """
        tracks = self._run_byte_tracker(nms_boxes, frame_shape)
        track_ids = _match_ids(
            nms_boxes, [t[:4] for t in tracks], [int(t[_TRACK_ID_COLUMN]) for t in tracks]
        )
        # Detections without a known identity: untracked ones and those of a
        # track ByteTrack has just confirmed. Both inherit the fallback id of
        # the overlapping untracked detection of the previous frame.
        pending = [
            i for i, tid in enumerate(track_ids) if tid is None or tid not in self._known_track_ids
        ]
        prev_ids = _match_ids(
            [nms_boxes[i] for i in pending],
            [box for _, box in self._prev_fallback],
            [fid for fid, _ in self._prev_fallback],
        )
        ids = list(track_ids)
        for i, prev_id in zip(pending, prev_ids):
            if track_ids[i] is not None:
                if prev_id is not None:
                    self._aliases[track_ids[i]] = prev_id
                self._known_track_ids.add(track_ids[i])
            elif prev_id is not None:
                ids[i] = prev_id
            else:
                ids[i] = next(self._fallback_ids)
        ids = [self._aliases.get(tid, tid) for tid in ids]
        self._prev_fallback = [
            (ids[i], nms_boxes[i][:4]) for i, tid in enumerate(track_ids) if tid is None
        ]
        return [(ids[i], b[0], b[1], b[2], b[3], b[4]) for i, b in enumerate(nms_boxes)]

    def _run_byte_tracker(self, nms_boxes, frame_shape):
        det = np.array(
            [[b[0], b[1], b[2], b[3], b[4], 0] for b in nms_boxes], dtype=np.float32
        ).reshape(-1, 6)
        try:
            return self.byte_tracker.update(self._boxes_cls(det, frame_shape[:2]))
        except Exception as e:
            # Every frame may fail the same way: one traceback is enough
            _log.warning("Tracker update failed: %s", e, exc_info=not self._failure_logged)
            self._failure_logged = True
            return []


def _match_ids(boxes, ref_boxes, ref_ids):
    """
    Greedy one-to-one association of boxes to reference boxes by IoU.

    Parameters
    ----------
    boxes : list
        Boxes ``[x1, y1, x2, y2, ...]`` to label.
    ref_boxes : list
        Reference boxes ``[x1, y1, x2, y2]``.
    ref_ids : list of int
        Id of each reference box.

    Returns
    -------
    list of int or None
        For each box the id of the best overlapping free reference box with
        IoU >= ``ASSOCIATION_IOU``, or None.
    """
    pairs = sorted(
        (
            (compute_iou_boxes(box[:4], ref[:4]), i, j)
            for i, box in enumerate(boxes)
            for j, ref in enumerate(ref_boxes)
        ),
        reverse=True,
    )
    ids = [None] * len(boxes)
    used_refs = set()
    for iou, i, j in pairs:
        if iou < ASSOCIATION_IOU:
            break
        if ids[i] is None and j not in used_refs:
            ids[i] = ref_ids[j]
            used_refs.add(j)
    return ids


def create_tracker(fps, config: PipelineConfig):
    """
    Crea un'istanza di PersonTracker basata su ByteTrack.

    Parameters
    ----------
    fps : float
        Frame rate del video sorgente.
    config : PipelineConfig
        Configurazione della pipeline; usa ``track_max_age`` e
        ``track_match_thresh``.

    Returns
    -------
    PersonTracker
        Tracker configurato.
    """
    from ultralytics.engine.results import Boxes

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
        fuse_score=True,
    )
    return PersonTracker(BYTETracker(tracker_args, frame_rate=int(fps)), Boxes)


def update_tracker(tracker, nms_boxes, frame_shape):
    """
    Aggiorna il tracker con i box NMS.

    Returns
    -------
    list of (track_id, x1, y1, x2, y2, conf)
        Una voce per ogni box in ingresso, nello stesso ordine.
    """
    return tracker.update(nms_boxes, frame_shape)


class TemporalSmoother:
    """
    Exponential Moving Average con ghost boxes per occlusioni temporanee.

    A differenza della media mobile semplice (lag di N/2 frame), l'EMA
    dà peso maggiore ai frame recenti. Quando una persona è temporaneamente
    occlusa, mantiene un "ghost box" leggermente espanso per continuare
    l'oscuramento, evitando gap nell'anonimizzazione.
    """

    def __init__(self, alpha, ghost_frames=10, ghost_expansion=1.15):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha deve essere tra 0 (escl.) e 1.0, ricevuto {alpha}")
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
        """Avvia ghost countdown per track non più attivi.

        ATTENZIONE: deve essere chiamato PRIMA di get_ghost_boxes()
        per ogni frame. Invertire l'ordine produce ghost erronei.
        """
        for tid in list(self.state):
            if tid not in active_ids and tid not in self.ghost_countdown:
                self.ghost_countdown[tid] = self.ghost_frames
