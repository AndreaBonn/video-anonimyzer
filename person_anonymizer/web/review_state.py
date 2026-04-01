"""
Stato condiviso thread-safe per la revisione manuale via web.
Bridge tra il thread della pipeline e i thread Flask.
"""

import copy
import threading

import cv2

__all__ = ["ReviewState"]


class ReviewState:
    """Gestisce lo stato della review manuale tra pipeline thread e Flask."""

    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._active = False

        self._video_path = None
        self._annotations = None
        self._total_frames = 0
        self._frame_w = 0
        self._frame_h = 0
        self._fps = 0.0
        self._fisheye_enabled = False
        self._undist_map1 = None
        self._undist_map2 = None
        self._cap = None

    @property
    def is_active(self):
        with self._lock:
            return self._active

    def setup(
        self,
        video_path,
        annotations,
        total_frames,
        frame_w,
        frame_h,
        fps,
        fisheye_enabled=False,
        undist_map1=None,
        undist_map2=None,
    ):
        """Chiamato dal pipeline thread quando le annotazioni sono pronte.

        Parameters
        ----------
        video_path : str
            Percorso del video sorgente.
        annotations : dict
            Annotazioni auto-rilevate {frame_idx: {auto, manual, intensities}}.
        total_frames : int
            Numero totale di frame nel video.
        frame_w : int
            Larghezza frame in pixel.
        frame_h : int
            Altezza frame in pixel.
        fps : float
            Frame per secondo del video.
        fisheye_enabled : bool
            Se la correzione fisheye è attiva.
        undist_map1 : numpy.ndarray or None
            Mappa di undistortion 1.
        undist_map2 : numpy.ndarray or None
            Mappa di undistortion 2.
        """
        with self._lock:
            self._video_path = video_path
            self._annotations = copy.deepcopy(annotations)
            self._total_frames = total_frames
            self._frame_w = frame_w
            self._frame_h = frame_h
            self._fps = fps
            self._fisheye_enabled = fisheye_enabled
            self._undist_map1 = undist_map1
            self._undist_map2 = undist_map2
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            try:
                self._cap = cv2.VideoCapture(video_path)
                self._active = True
            except Exception:
                self._cap = None
                raise
            self._event.clear()

    def wait_for_completion(self):
        """Blocca il pipeline thread fino a conferma dell'utente.

        Returns
        -------
        dict
            Le annotazioni riviste dall'utente.
        """
        self._event.wait()
        with self._lock:
            result = copy.deepcopy(self._annotations)
            self._active = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return result

    def complete(self, annotations):
        """Chiamato da Flask quando l'utente conferma la review.

        Parameters
        ----------
        annotations : dict
            Le annotazioni finali dopo la revisione dell'utente.
        """
        with self._lock:
            self._annotations = copy.deepcopy(annotations)
        self._event.set()

    def get_frame_jpeg(self, frame_idx, max_width=1280):
        """Estrae un frame dal video e lo restituisce come JPEG.

        Parameters
        ----------
        frame_idx : int
            Indice del frame da estrarre.
        max_width : int
            Larghezza massima per il ridimensionamento.

        Returns
        -------
        tuple of (bytes, float)
            (jpeg_bytes, scale_factor) dove scale_factor è il rapporto
            tra la dimensione visualizzata e quella originale.
        """
        # cap.set/cap.read e la lettura di fisheye/map1/map2 avvengono
        # interamente dentro il lock: nessuna richiesta concorrente può
        # intercalare un secondo cap.set tra set e read di questa chiamata.
        # remap/resize/imencode operano su copie locali e non richiedono il lock.
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None, 1.0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret:
                return None, 1.0
            fisheye = self._fisheye_enabled
            map1 = self._undist_map1
            map2 = self._undist_map2

        if fisheye and map1 is not None and map2 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        scale = 1.0
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes(), scale

    def get_annotations(self):
        """Restituisce una copia delle annotazioni correnti.

        Returns
        -------
        dict
            Copia profonda delle annotazioni.
        """
        with self._lock:
            return copy.deepcopy(self._annotations)

    def update_annotations(self, frame_idx, frame_data):
        """Aggiorna le annotazioni per un singolo frame.

        Parameters
        ----------
        frame_idx : int
            Indice del frame.
        frame_data : dict
            Dati annotazione del frame {auto, manual, intensities}.
        """
        with self._lock:
            self._annotations[frame_idx] = copy.deepcopy(frame_data)

    def get_metadata(self):
        """Restituisce i metadati del video.

        Returns
        -------
        dict
            Contiene total_frames, frame_w, frame_h, fps.
        """
        with self._lock:
            return {
                "total_frames": self._total_frames,
                "frame_w": self._frame_w,
                "frame_h": self._frame_h,
                "fps": self._fps,
            }
