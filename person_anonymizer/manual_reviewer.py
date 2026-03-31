"""
Modulo di revisione manuale con interfaccia OpenCV.

Gestisce la finestra interattiva per aggiungere/rimuovere poligoni
di oscuramento su ogni frame del video.
"""

import cv2
import numpy as np
from copy import deepcopy

# Key codes OpenCV
KEY_RIGHT = 83
KEY_LEFT = 81
KEY_SPACE = 32
KEY_ENTER = 13
KEY_CTRL_Z = 26
KEY_ESC = 27


class ManualReviewer:
    """Interfaccia interattiva OpenCV per revisione poligoni di oscuramento."""

    def __init__(
        self,
        video_path,
        auto_annotations,
        config,
        fisheye_enabled=False,
        undist_map1=None,
        undist_map2=None,
    ):
        # Copia annotazioni
        self.annotations = deepcopy(auto_annotations)

        # Configurazione colori e display
        self.auto_color = config.get("auto_color", (0, 255, 0))
        self.manual_color = config.get("manual_color", (0, 120, 255))
        self.drawing_color = config.get("drawing_color", (255, 255, 0))
        self.fill_alpha = config.get("fill_alpha", 0.35)
        self.max_width = config.get("max_width", 1280)

        # Video
        self.cap = cv2.VideoCapture(video_path)
        self.fisheye_enabled = fisheye_enabled
        self.undist_map1 = undist_map1
        self.undist_map2 = undist_map2

        # Metadata video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Scaling
        self.scale = 1.0
        self.display_w = self.frame_w
        self.display_h = self.frame_h
        if self.frame_w > self.max_width:
            self.scale = self.max_width / self.frame_w
            self.display_w = self.max_width
            self.display_h = int(self.frame_h * self.scale)

        # Stato interazione
        self.current_frame_idx = 0
        self.current_polygon_points = []
        self.delete_mode = False
        self.mouse_pos = (0, 0)

        # Cache frame
        self._cached_frame = None
        self._cached_frame_idx = -1

        # Statistiche
        self.stats = {
            "added": 0,
            "removed": 0,
            "frames_modified": set(),
            "frames_reviewed": set(),
        }

    def _get_frame(self, idx):
        """
        Legge un frame dal video con cache sull'indice corrente.

        Parameters
        ----------
        idx : int
            Indice del frame da leggere.

        Returns
        -------
        numpy.ndarray
            Frame BGR pronto per il rendering.
        """
        if idx == self._cached_frame_idx and self._cached_frame is not None:
            return self._cached_frame.copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            # Frame corrotto: restituisci frame nero
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Frame {idx}: non leggibile",
                (50, self.frame_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if self.fisheye_enabled and self.undist_map1 is not None:
            frame = cv2.remap(frame, self.undist_map1, self.undist_map2, cv2.INTER_LINEAR)

        self._cached_frame = frame.copy()
        self._cached_frame_idx = idx
        return frame.copy()

    def _display_to_original(self, x, y):
        """
        Converte coordinate display in coordinate originali del frame.

        Parameters
        ----------
        x, y : int
            Coordinate nello spazio display (scalato).

        Returns
        -------
        tuple[int, int]
            Coordinate nello spazio originale del frame.
        """
        return int(x / self.scale), int(y / self.scale)

    def _original_to_display(self, x, y):
        """
        Converte coordinate originali del frame in coordinate display.

        Parameters
        ----------
        x, y : int
            Coordinate nello spazio originale del frame.

        Returns
        -------
        tuple[int, int]
            Coordinate nello spazio display (scalato).
        """
        return int(x * self.scale), int(y * self.scale)

    def _point_in_polygon(self, px, py, polygon):
        """
        Verifica se un punto è dentro un poligono tramite cv2.pointPolygonTest.

        Parameters
        ----------
        px, py : float
            Coordinate del punto da testare.
        polygon : list
            Lista di punti (x, y) che definiscono il poligono.

        Returns
        -------
        bool
            True se il punto è dentro o sul bordo del poligono.
        """
        pts = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, (px, py), False)
        return result >= 0

    def _render_display(self):
        """
        Renderizza il frame corrente con overlay di poligoni e UI.

        Returns
        -------
        numpy.ndarray
            Frame BGR pronto per cv2.imshow.
        """
        frame = self._get_frame(self.current_frame_idx)
        overlay = frame.copy()

        ann = self.annotations.get(
            self.current_frame_idx, {"auto": [], "manual": [], "intensities": []}
        )
        auto_polys = ann.get("auto", [])
        manual_polys = ann.get("manual", [])

        # Poligoni automatici (verde)
        for poly in auto_polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], self.auto_color)
            cv2.polylines(frame, [pts], True, self.auto_color, 2)
            for pt in poly:
                cv2.circle(frame, tuple(pt), 4, (255, 255, 255), -1)

        # Poligoni manuali (arancione)
        for poly in manual_polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], self.manual_color)
            cv2.polylines(frame, [pts], True, self.manual_color, 2)
            for pt in poly:
                cv2.circle(frame, tuple(pt), 4, (255, 255, 255), -1)

        # Blend overlay
        cv2.addWeighted(overlay, self.fill_alpha, frame, 1 - self.fill_alpha, 0, frame)

        # Poligono in corso di disegno
        if self.current_polygon_points:
            for i, pt in enumerate(self.current_polygon_points):
                cv2.circle(frame, tuple(pt), 5, (255, 255, 255), -1)
                if i > 0:
                    cv2.line(
                        frame,
                        tuple(self.current_polygon_points[i - 1]),
                        tuple(pt),
                        self.drawing_color,
                        2,
                    )

            # Linea dall'ultimo punto al cursore
            last_pt = self.current_polygon_points[-1]
            mx, my = self._display_to_original(*self.mouse_pos)
            cv2.line(frame, tuple(last_pt), (mx, my), self.drawing_color, 1, cv2.LINE_AA)

        # Resize per display
        if self.scale != 1.0:
            frame = cv2.resize(frame, (self.display_w, self.display_h))

        # Barra info in alto
        n_auto = len(auto_polys)
        n_manual = len(manual_polys)
        mode_label = "[DEL]" if self.delete_mode else "[NORM]"
        info_text = (
            f"  Frame {self.current_frame_idx + 1} / {self.total_frames}  |  "
            f"Poligoni: {n_auto + n_manual} "
            f"({n_auto} auto, {n_manual} man)  |  {mode_label}"
        )

        # Sfondo barra info
        bar_h = 30
        cv2.rectangle(frame, (0, 0), (self.display_w, bar_h), (40, 40, 40), -1)
        cv2.putText(frame, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Barra istruzioni in basso
        help_h = 45
        y_start = self.display_h - help_h
        cv2.rectangle(frame, (0, y_start), (self.display_w, self.display_h), (40, 40, 40), -1)
        help_line1 = "  <-/-> Naviga  |  Spazio Avanti  |  " "Invio Chiudi poligono  |  Q Esci"
        help_line2 = (
            "  Click Aggiungi punto  |  Ctrl+Z Annulla punto  |  "
            "D Elimina  |  Esc Annulla poligono"
        )
        cv2.putText(
            frame, help_line1, (5, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1
        )
        cv2.putText(
            frame, help_line2, (5, y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1
        )

        return frame

    def _on_mouse(self, event, x, y, flags, param):
        """
        Callback mouse per aggiunta punti e cancellazione poligoni.

        Parameters
        ----------
        event : int
            Tipo di evento OpenCV (cv2.EVENT_*).
        x, y : int
            Coordinate del cursore nello spazio display.
        flags, param :
            Parametri aggiuntivi OpenCV (non usati).
        """
        self.mouse_pos = (x, y)
        orig_x, orig_y = self._display_to_original(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.delete_mode:
                self._delete_polygon_at(orig_x, orig_y)
            else:
                self.current_polygon_points.append((orig_x, orig_y))

    def _delete_polygon_at(self, orig_x, orig_y):
        """
        Cerca e rimuove il primo poligono che contiene il punto dato.

        Controlla prima i poligoni manuali, poi quelli automatici.

        Parameters
        ----------
        orig_x, orig_y : int
            Coordinate nello spazio originale del frame.
        """
        ann = self.annotations.get(
            self.current_frame_idx, {"auto": [], "manual": [], "intensities": []}
        )

        # Cerca prima nei manuali
        for i, poly in enumerate(ann.get("manual", [])):
            if self._point_in_polygon(orig_x, orig_y, poly):
                ann["manual"].pop(i)
                self.stats["removed"] += 1
                self.stats["frames_modified"].add(self.current_frame_idx)
                return

        # Poi negli automatici
        for i, poly in enumerate(ann.get("auto", [])):
            if self._point_in_polygon(orig_x, orig_y, poly):
                ann["auto"].pop(i)
                # Rimuovi anche intensità corrispondente
                intensities = ann.get("intensities", [])
                if i < len(intensities):
                    intensities.pop(i)
                self.stats["removed"] += 1
                self.stats["frames_modified"].add(self.current_frame_idx)
                return

    def _handle_key(self, key):
        """
        Gestisce la pressione di un tasto e aggiorna lo stato interno.

        Parameters
        ----------
        key : int
            Codice tasto restituito da cv2.waitKey.

        Returns
        -------
        bool
            False se il loop deve terminare (tasto Q), True altrimenti.
        """
        # Q - Conferma e esci
        if key == ord("q") or key == ord("Q"):
            return False

        # Freccia destra o Spazio — Avanti
        if key in (KEY_RIGHT, KEY_SPACE):
            self.current_polygon_points = []
            self.delete_mode = False
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1

        # Freccia sinistra — Indietro
        elif key == KEY_LEFT:
            self.current_polygon_points = []
            self.delete_mode = False
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1

        # Invio — Chiudi poligono
        elif key == KEY_ENTER:
            if len(self.current_polygon_points) >= 3:
                ann = self.annotations.get(
                    self.current_frame_idx, {"auto": [], "manual": [], "intensities": []}
                )
                ann["manual"].append(list(self.current_polygon_points))
                self.annotations[self.current_frame_idx] = ann
                self.stats["added"] += 1
                self.stats["frames_modified"].add(self.current_frame_idx)
                self.current_polygon_points = []

        # Ctrl+Z — Annulla ultimo punto
        elif key == KEY_CTRL_Z:
            if self.current_polygon_points:
                self.current_polygon_points.pop()

        # D — Toggle modalità delete
        elif key == ord("d") or key == ord("D"):
            self.delete_mode = not self.delete_mode
            self.current_polygon_points = []

        # Esc — Annulla poligono in corso
        elif key == KEY_ESC:
            self.current_polygon_points = []

        return True

    def _get_final_stats(self):
        """
        Converte i set interni in conteggi per l'output finale.

        Returns
        -------
        dict
            Statistiche con chiavi added, removed, frames_modified,
            frames_reviewed come interi.
        """
        return {
            "added": self.stats["added"],
            "removed": self.stats["removed"],
            "frames_modified": len(self.stats["frames_modified"]),
            "frames_reviewed": len(self.stats["frames_reviewed"]),
        }

    def run(self):
        """
        Avvia il loop principale della revisione manuale.

        Returns
        -------
        tuple (annotations, final_stats)
            Annotazioni aggiornate e statistiche della sessione.
        """
        window_name = "Person Anonymizer - Revisione Manuale"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._on_mouse)

        running = True
        while running:
            self.stats["frames_reviewed"].add(self.current_frame_idx)

            # Assicurati che il frame abbia entry nelle annotazioni
            if self.current_frame_idx not in self.annotations:
                self.annotations[self.current_frame_idx] = {
                    "auto": [],
                    "manual": [],
                    "intensities": [],
                }

            display = self._render_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                # Nessun tasto premuto
                continue

            running = self._handle_key(key)

        cv2.destroyAllWindows()
        self.cap.release()

        return self.annotations, self._get_final_stats()


def run_manual_review(
    video_path, auto_annotations, config, fisheye_enabled=False, undist_map1=None, undist_map2=None
):
    """
    Backward-compatible wrapper per ManualReviewer.

    Parameters
    ----------
    video_path : str
        Percorso del video originale.
    auto_annotations : dict
        Annotazioni dalla pipeline automatica.
    config : dict
        Colori, dimensioni finestra, parametri UI.
    fisheye_enabled : bool
        Se True, applica undistortion ai frame.
    undist_map1, undist_map2 : ndarray or None
        Mappe di undistortion.

    Returns
    -------
    tuple (annotations, stats)
        Annotazioni finali e statistiche revisione.
    """
    reviewer = ManualReviewer(
        video_path,
        auto_annotations,
        config,
        fisheye_enabled,
        undist_map1,
        undist_map2,
    )
    if not reviewer.cap.isOpened():
        print("Errore: impossibile aprire il video per la revisione.")
        stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
        return reviewer.annotations, stats
    return reviewer.run()
