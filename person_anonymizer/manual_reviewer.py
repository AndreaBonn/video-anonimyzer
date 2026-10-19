"""
Modulo di revisione manuale con interfaccia OpenCV.

Shell OpenCV della revisione: finestra, input e rendering. Lo stato e le
regole di modifica stanno in `review_editor`, il disegno in `review_drawing`.
"""

import cv2
import numpy as np

from .models import FisheyeContext
from .review_drawing import (
    OverlayStyle,
    draw_annotations,
    draw_pending_polygon,
    draw_status_bars,
    format_info_text,
)
from .review_editor import ReviewEditor

__all__ = ["ManualReviewer", "run_manual_review"]

# Raw key codes as returned by cv2.waitKeyEx. Arrows are X11 keysyms on the
# Qt/GTK backends, virtual-key codes shifted by 16 on Windows and NSEvent
# function keys on macOS. Masking with 0xFF (cv2.waitKey) would turn the left
# arrow into "Q" (quit) and the right arrow into "S".
KEY_LEFT = 0xFF51
KEY_RIGHT = 0xFF53
KEY_LEFT_CODES = frozenset({KEY_LEFT, 0x250000, 0xF702})
KEY_RIGHT_CODES = frozenset({KEY_RIGHT, 0x270000, 0xF703})
KEY_SPACE = 32
KEY_ENTER = 13
KEY_CTRL_Z = 26
KEY_ESC = 27
# The Qt backend swallows every Ctrl+key, so Backspace is the undo key that
# works everywhere.
KEY_BACKSPACE_CODES = frozenset({8, 0xFF08})
KEY_NONE = -1
# GTK reports modifier state (NumLock, Shift...) above the low 16 bits.
_KEYSYM_MASK = 0xFFFF

WINDOW_NAME = "Person Anonymizer - Revisione Manuale"
WAIT_KEY_DELAY_MS = 30
UNREADABLE_TEXT_ORIGIN_X = 50
UNREADABLE_FONT_SCALE = 1
UNREADABLE_COLOR = (0, 0, 255)
UNREADABLE_THICKNESS = 2


def display_scale(frame_w: int, frame_h: int, max_width: int) -> tuple[float, int, int]:
    """
    Compute the scale that fits a frame into the review window.

    Returns
    -------
    tuple[float, int, int]
        Scale factor, display width and display height. Frames not wider than
        `max_width` are shown unscaled.
    """
    if frame_w <= max_width:
        return 1.0, frame_w, frame_h
    scale = max_width / frame_w
    return scale, max_width, int(frame_h * scale)


def decode_key(raw: int) -> int:
    """
    Normalize a raw cv2.waitKeyEx code.

    Arrow codes are kept whole because masking turns them into letters;
    every other code loses the modifier bits GTK reports above bit 16.
    """
    if raw in KEY_LEFT_CODES or raw in KEY_RIGHT_CODES:
        return raw
    return raw & _KEYSYM_MASK


class ManualReviewer:
    """OpenCV shell of the manual review: video access, rendering and input decoding."""

    def __init__(self, video_path, auto_annotations, config, fisheye=None):
        self.style = OverlayStyle.from_config(config)
        self.max_width = config.review_window_max_width

        self.cap = cv2.VideoCapture(video_path)
        self.fisheye = fisheye or FisheyeContext()

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale, self.display_w, self.display_h = display_scale(
            self.frame_w, self.frame_h, self.max_width
        )

        self.editor = ReviewEditor(auto_annotations, self.total_frames)
        self.mouse_pos = (0, 0)

        self._cached_frame = None
        self._cached_frame_idx = -1

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
            frame = self._unreadable_frame(idx)

        frame = self.fisheye.undistort(frame)

        self._cached_frame = frame.copy()
        self._cached_frame_idx = idx
        return frame.copy()

    def _unreadable_frame(self, idx):
        """Return a black frame labelled as unreadable, for corrupted video frames."""
        frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {idx}: non leggibile",
            (UNREADABLE_TEXT_ORIGIN_X, self.frame_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            UNREADABLE_FONT_SCALE,
            UNREADABLE_COLOR,
            UNREADABLE_THICKNESS,
        )
        return frame

    def _display_to_original(self, x, y):
        """Convert display (scaled) coordinates to original frame coordinates."""
        return int(x / self.scale), int(y / self.scale)

    def _original_to_display(self, x, y):
        """Convert original frame coordinates to display (scaled) coordinates."""
        return int(x * self.scale), int(y * self.scale)

    def _render_display(self):
        """
        Compose the current frame with overlays, scaling and status bars.

        Returns
        -------
        numpy.ndarray
            Frame BGR pronto per cv2.imshow.
        """
        editor = self.editor
        annotation = editor.current_annotation()
        frame = draw_annotations(
            self._get_frame(editor.current_frame_idx), annotation=annotation, style=self.style
        )
        frame = draw_pending_polygon(
            frame,
            points=editor.pending_points,
            cursor=self._display_to_original(*self.mouse_pos),
            color=self.style.drawing_color,
        )
        if self.scale != 1.0:
            frame = cv2.resize(frame, (self.display_w, self.display_h))
        info_text = format_info_text(
            editor.current_frame_idx, editor.total_frames, annotation, editor.delete_mode
        )
        return draw_status_bars(frame, info_text=info_text)

    def _on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback: track the cursor and forward left clicks to the editor.

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
        if event == cv2.EVENT_LBUTTONDOWN:
            self.editor.click(self._display_to_original(x, y))

    def _handle_key(self, key):
        """
        Map a raw key press (cv2.waitKeyEx code) to an editor action.

        Returns
        -------
        bool
            False when the loop must end (Q key), True otherwise.
        """
        key = decode_key(key)
        editor = self.editor
        if key in (ord("q"), ord("Q")):
            return False
        if key in KEY_RIGHT_CODES or key == KEY_SPACE:
            editor.go_next()
        elif key in KEY_LEFT_CODES:
            editor.go_previous()
        elif key == KEY_ENTER:
            editor.close_polygon()
        elif key == KEY_CTRL_Z or key in KEY_BACKSPACE_CODES:
            editor.undo_point()
        elif key in (ord("d"), ord("D")):
            editor.toggle_delete_mode()
        elif key == KEY_ESC:
            editor.cancel_polygon()
        return True

    def run(self):
        """
        Avvia il loop principale della revisione manuale.

        Returns
        -------
        tuple (annotations, final_stats)
            Annotazioni aggiornate e statistiche della sessione.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        running = True
        while running:
            self.editor.visit_current()
            cv2.imshow(WINDOW_NAME, self._render_display())

            key = cv2.waitKeyEx(WAIT_KEY_DELAY_MS)
            if key == KEY_NONE:
                continue
            running = self._handle_key(key)

        cv2.destroyAllWindows()
        self.cap.release()

        return self.editor.annotations, self.editor.final_stats()


def run_manual_review(video_path, auto_annotations, config, fisheye=None):
    """
    Wrapper per ManualReviewer.

    Parameters
    ----------
    video_path : str
        Percorso del video originale.
    auto_annotations : dict
        Annotazioni dalla pipeline automatica.
    config : PipelineConfig
        Configurazione della pipeline.
    fisheye : FisheyeContext or None
        Contesto di correzione fish-eye.

    Returns
    -------
    tuple (annotations, stats)
        Annotazioni finali e statistiche revisione.
    """
    reviewer = ManualReviewer(video_path, auto_annotations, config, fisheye)
    if not reviewer.cap.isOpened():
        print("Errore: impossibile aprire il video per la revisione.")
        stats = {"added": 0, "removed": 0, "frames_modified": 0, "frames_reviewed": 0}
        return reviewer.editor.annotations, stats
    return reviewer.run()
