"""
Modulo di revisione manuale con interfaccia OpenCV.

Gestisce la finestra interattiva per aggiungere/rimuovere poligoni
di oscuramento su ogni frame del video.
"""

import cv2
import numpy as np
from copy import deepcopy


def run_manual_review(video_path, auto_annotations, config,
                      fisheye_enabled=False, undist_map1=None,
                      undist_map2=None):
    """
    Apre la finestra OpenCV di revisione manuale.

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
    annotations = deepcopy(auto_annotations)

    auto_color = config.get("auto_color", (0, 255, 0))
    manual_color = config.get("manual_color", (0, 120, 255))
    drawing_color = config.get("drawing_color", (255, 255, 0))
    fill_alpha = config.get("fill_alpha", 0.35)
    max_width = config.get("max_width", 1280)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video per la revisione.")
        stats = {"added": 0, "removed": 0, "frames_modified": 0,
                 "frames_reviewed": 0}
        return annotations, stats

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calcolo scaling per display
    scale = 1.0
    display_w, display_h = frame_w, frame_h
    if frame_w > max_width:
        scale = max_width / frame_w
        display_w = max_width
        display_h = int(frame_h * scale)

    # Stato
    current_frame_idx = 0
    current_polygon_points = []  # Punti del poligono in corso (coordinate originali)
    delete_mode = False
    mouse_pos = (0, 0)  # Posizione cursore (coordinate display)

    # Statistiche
    stats = {"added": 0, "removed": 0, "frames_modified": set(),
             "frames_reviewed": set()}

    # Cache frame corrente
    cached_frame = None
    cached_frame_idx = -1

    def get_frame(idx):
        """Legge un frame dal video, con cache."""
        nonlocal cached_frame, cached_frame_idx
        if idx == cached_frame_idx and cached_frame is not None:
            return cached_frame.copy()

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Frame corrotto: restituisci frame nero
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {idx}: non leggibile",
                       (50, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)

        if fisheye_enabled and undist_map1 is not None:
            frame = cv2.remap(frame, undist_map1, undist_map2,
                             cv2.INTER_LINEAR)

        cached_frame = frame.copy()
        cached_frame_idx = idx
        return frame.copy()

    def display_to_original(x, y):
        """Converte coordinate display → originali."""
        return int(x / scale), int(y / scale)

    def original_to_display(x, y):
        """Converte coordinate originali → display."""
        return int(x * scale), int(y * scale)

    def point_in_polygon(px, py, polygon):
        """Verifica se un punto è dentro un poligono."""
        pts = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, (px, py), False)
        return result >= 0

    def render_display():
        """Renderizza il frame con overlay poligoni per il display."""
        frame = get_frame(current_frame_idx)
        overlay = frame.copy()

        ann = annotations.get(current_frame_idx,
                              {"auto": [], "manual": [], "intensities": []})
        auto_polys = ann.get("auto", [])
        manual_polys = ann.get("manual", [])

        # Poligoni automatici (verde)
        for poly in auto_polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], auto_color)
            cv2.polylines(frame, [pts], True, auto_color, 2)
            for pt in poly:
                cv2.circle(frame, tuple(pt), 4, (255, 255, 255), -1)

        # Poligoni manuali (arancione)
        for poly in manual_polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], manual_color)
            cv2.polylines(frame, [pts], True, manual_color, 2)
            for pt in poly:
                cv2.circle(frame, tuple(pt), 4, (255, 255, 255), -1)

        # Blend overlay
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # Poligono in corso di disegno
        if current_polygon_points:
            for i, pt in enumerate(current_polygon_points):
                cv2.circle(frame, tuple(pt), 5, (255, 255, 255), -1)
                if i > 0:
                    cv2.line(frame, tuple(current_polygon_points[i - 1]),
                            tuple(pt), drawing_color, 2)

            # Linea dal ultimo punto al cursore
            last_pt = current_polygon_points[-1]
            mx, my = display_to_original(*mouse_pos)
            cv2.line(frame, tuple(last_pt), (mx, my), drawing_color, 1,
                    cv2.LINE_AA)

        # Resize per display
        if scale != 1.0:
            frame = cv2.resize(frame, (display_w, display_h))

        # Barra info in alto
        n_auto = len(auto_polys)
        n_manual = len(manual_polys)
        mode_label = "[DEL]" if delete_mode else "[NORM]"
        info_text = (f"  Frame {current_frame_idx + 1} / {total_frames}  |  "
                     f"Poligoni: {n_auto + n_manual} "
                     f"({n_auto} auto, {n_manual} man)  |  {mode_label}")

        # Sfondo barra info
        bar_h = 30
        cv2.rectangle(frame, (0, 0), (display_w, bar_h), (40, 40, 40), -1)
        cv2.putText(frame, info_text, (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Barra istruzioni in basso
        help_h = 45
        y_start = display_h - help_h
        cv2.rectangle(frame, (0, y_start), (display_w, display_h),
                     (40, 40, 40), -1)
        help_line1 = ("  <-/-> Naviga  |  Spazio Avanti  |  "
                      "Invio Chiudi poligono  |  Q Esci")
        help_line2 = ("  Click Aggiungi punto  |  Ctrl+Z Annulla punto  |  "
                      "D Elimina  |  Esc Annulla poligono")
        cv2.putText(frame, help_line1, (5, y_start + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, help_line2, (5, y_start + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return frame

    def mouse_callback(event, x, y, flags, param):
        """Gestisce eventi mouse."""
        nonlocal current_polygon_points, delete_mode, mouse_pos

        mouse_pos = (x, y)
        orig_x, orig_y = display_to_original(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if delete_mode:
                # Cerca e elimina poligono sotto il click
                ann = annotations.get(current_frame_idx,
                                      {"auto": [], "manual": [],
                                       "intensities": []})

                # Cerca prima nei manuali
                for i, poly in enumerate(ann.get("manual", [])):
                    if point_in_polygon(orig_x, orig_y, poly):
                        ann["manual"].pop(i)
                        stats["removed"] += 1
                        stats["frames_modified"].add(current_frame_idx)
                        return

                # Poi negli automatici
                for i, poly in enumerate(ann.get("auto", [])):
                    if point_in_polygon(orig_x, orig_y, poly):
                        ann["auto"].pop(i)
                        # Rimuovi anche intensità corrispondente
                        intensities = ann.get("intensities", [])
                        if i < len(intensities):
                            intensities.pop(i)
                        stats["removed"] += 1
                        stats["frames_modified"].add(current_frame_idx)
                        return
            else:
                # Aggiungi punto al poligono in corso
                current_polygon_points.append((orig_x, orig_y))

    # Creazione finestra
    window_name = "Person Anonymizer - Revisione Manuale"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    running = True
    while running:
        stats["frames_reviewed"].add(current_frame_idx)

        # Assicurati che il frame abbia entry nelle annotazioni
        if current_frame_idx not in annotations:
            annotations[current_frame_idx] = {
                "auto": [], "manual": [], "intensities": []
            }

        display = render_display()
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            # Nessun tasto premuto
            continue

        # Q - Conferma e esci
        if key == ord('q') or key == ord('Q'):
            running = False
            continue

        # Freccia destra (83) o Spazio (32) — Avanti
        if key == 83 or key == 32:
            current_polygon_points = []
            delete_mode = False
            if current_frame_idx < total_frames - 1:
                current_frame_idx += 1

        # Freccia sinistra (81) — Indietro
        elif key == 81:
            current_polygon_points = []
            delete_mode = False
            if current_frame_idx > 0:
                current_frame_idx -= 1

        # Invio (13) — Chiudi poligono
        elif key == 13:
            if len(current_polygon_points) >= 3:
                ann = annotations.get(current_frame_idx,
                                      {"auto": [], "manual": [],
                                       "intensities": []})
                ann["manual"].append(list(current_polygon_points))
                annotations[current_frame_idx] = ann
                stats["added"] += 1
                stats["frames_modified"].add(current_frame_idx)
                current_polygon_points = []

        # Ctrl+Z (26) — Annulla ultimo punto
        elif key == 26:
            if current_polygon_points:
                current_polygon_points.pop()

        # D — Toggle modalità delete
        elif key == ord('d') or key == ord('D'):
            delete_mode = not delete_mode
            current_polygon_points = []

        # Esc (27) — Annulla poligono in corso
        elif key == 27:
            current_polygon_points = []

    cv2.destroyAllWindows()
    cap.release()

    # Converti set in conteggi per le statistiche
    final_stats = {
        "added": stats["added"],
        "removed": stats["removed"],
        "frames_modified": len(stats["frames_modified"]),
        "frames_reviewed": len(stats["frames_reviewed"])
    }

    return annotations, final_stats
