"""Backend SAM3 (Segment Anything Model 3) per segmentazione pixel-precise.

Fornisce wrapper thin per SAM3 image e video predictor.
L'import di SAM3 è lazy: il modulo è utilizzabile solo se sam3 è installato
e Python >= 3.12.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "check_sam3_available",
    "mask_to_polygons",
    "Sam3ImageRefiner",
    "Sam3VideoDetector",
]

_SAM3_MIN_PYTHON = (3, 12)


def check_sam3_available() -> bool:
    """Verifica se SAM3 è installabile e il runtime è compatibile.

    Returns
    -------
    bool
        True se sam3 è importabile e Python >= 3.12.
    """
    if sys.version_info < _SAM3_MIN_PYTHON:
        return False
    return importlib.util.find_spec("sam3") is not None


def _require_sam3():
    """Solleva ImportError con messaggio chiaro se SAM3 non è disponibile."""
    if sys.version_info < _SAM3_MIN_PYTHON:
        major, minor = _SAM3_MIN_PYTHON
        raise ImportError(
            f"SAM3 richiede Python >= {major}.{minor}. "
            f"Versione corrente: {sys.version_info.major}.{sys.version_info.minor}. "
            "Aggiorna Python per usare il backend SAM3."
        )
    if importlib.util.find_spec("sam3") is None:
        raise ImportError(
            "SAM3 non è installato. Installa con:\n"
            "  pip install -r requirements-sam3.txt\n"
            "oppure:\n"
            "  pip install 'person-anonymizer[sam3]'"
        )


def mask_to_polygons(
    binary_mask: np.ndarray,
    epsilon_ratio: float = 0.005,
    min_area: int = 100,
) -> list[list[tuple[int, int]]]:
    """Converte una maschera binaria in una lista di poligoni semplificati.

    Parameters
    ----------
    binary_mask : np.ndarray
        Maschera binaria (H, W) con valori 0/255 o 0/1.
    epsilon_ratio : float
        Rapporto epsilon per ``cv2.approxPolyDP`` rispetto al perimetro.
        Valori più bassi = più fedeltà alla maschera originale.
    min_area : int
        Area minima in pixel per filtrare contorni rumorosi.

    Returns
    -------
    list[list[tuple[int, int]]]
        Lista di poligoni, ciascuno come lista di punti (x, y).
    """
    mask = binary_mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) >= 3:
            poly = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
            polygons.append(poly)

    return polygons


@dataclass
class Sam3ImageRefiner:
    """Wrapper per SAM3 image predictor — affina box YOLO in maschere precise.

    Attributes
    ----------
    model_path : str
        Percorso al checkpoint SAM3.
    epsilon_ratio : float
        Rapporto semplificazione per ``mask_to_polygons``.
    min_mask_area : int
        Area minima maschera.
    device : str
        Device PyTorch (``"cuda"`` o ``"cpu"``).
    """

    model_path: str = "sam3_hiera_large.pt"
    epsilon_ratio: float = 0.005
    min_mask_area: int = 100
    device: str = "cuda"
    _predictor: object = field(default=None, repr=False)

    def _load_predictor(self):
        """Caricamento lazy del predictor SAM3."""
        _require_sam3()
        import torch
        from sam3 import SAM3ImagePredictor, build_sam3

        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"

        model = build_sam3(self.model_path)
        model = model.to(self.device)
        model.eval()
        self._predictor = SAM3ImagePredictor(model)

    def refine_boxes(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Affina bounding box in poligoni pixel-precisi via SAM3.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR (H, W, 3).
        boxes : list[tuple[int, int, int, int]]
            Lista di bounding box (x1, y1, x2, y2).

        Returns
        -------
        list[list[tuple[int, int]]]
            Lista di poligoni, uno per ciascun box input.
            Se una maschera non produce contorni validi, viene usato
            il rettangolo originale come fallback.
        """
        import torch

        if self._predictor is None:
            self._load_predictor()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(frame_rgb)

        polygons = []
        for x1, y1, x2, y2 in boxes:
            box_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)
            masks, _, _ = self._predictor.predict(box=box_tensor, multimask_output=False)
            mask = masks[0].cpu().numpy().astype(np.uint8)

            polys = mask_to_polygons(mask, self.epsilon_ratio, self.min_mask_area)
            if polys:
                polygons.append(polys[0])
            else:
                polygons.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        return polygons

    def release(self):
        """Libera memoria GPU."""
        if self._predictor is not None:
            import torch

            del self._predictor
            self._predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@dataclass
class Sam3VideoDetector:
    """Wrapper per SAM3 video predictor — detection + tracking integrato.

    Attributes
    ----------
    model_path : str
        Percorso al checkpoint SAM3.
    text_prompt : str
        Prompt testuale per la detection (es. ``"person"``).
    epsilon_ratio : float
        Rapporto semplificazione per ``mask_to_polygons``.
    min_mask_area : int
        Area minima maschera.
    device : str
        Device PyTorch (``"cuda"`` o ``"cpu"``).
    """

    model_path: str = "sam3_hiera_large.pt"
    text_prompt: str = "person"
    epsilon_ratio: float = 0.005
    min_mask_area: int = 100
    device: str = "cuda"

    def detect_video(
        self,
        video_path: str,
        config,
        stop_event=None,
    ) -> tuple[dict, dict]:
        """Esegue detection + tracking SAM3 su un intero video.

        Parameters
        ----------
        video_path : str
            Percorso del video.
        config : PipelineConfig
            Configurazione pipeline.
        stop_event : threading.Event | None
            Evento di stop per interruzione asincrona.

        Returns
        -------
        tuple[dict, dict]
            (annotations, report_data) nello stesso formato di
            ``run_detection_loop``.
        """
        _require_sam3()
        import torch
        from sam3 import SAM3VideoPredictor, build_sam3
        from tqdm import tqdm

        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"

        model = build_sam3(self.model_path)
        model = model.to(self.device)
        model.eval()
        predictor = SAM3VideoPredictor(model)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        state = predictor.init_state(video_path=video_path)
        predictor.add_new_text_prompt(state, text=self.text_prompt)

        annotations = {}
        report_data = {}

        print("\n[FASE 1/5] Rilevamento SAM3 video...")
        pbar = tqdm(total=total_frames, desc="SAM3 detection", unit=" frame")

        try:
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                if stop_event is not None and stop_event.is_set():
                    print("\n  Pipeline interrotta dall'utente.")
                    break

                frame_polygons = []
                for mask in masks:
                    mask_np = mask.cpu().numpy().squeeze().astype(np.uint8)
                    polys = mask_to_polygons(mask_np, self.epsilon_ratio, self.min_mask_area)
                    frame_polygons.extend(polys)

                from .anonymization import resolve_intensity

                frame_intensities = []
                for poly in frame_polygons:
                    ys = [p[1] for p in poly]
                    box_h = max(ys) - min(ys) if ys else 0
                    frame_intensities.append(resolve_intensity(config, box_h))

                annotations[frame_idx] = {
                    "auto": frame_polygons,
                    "manual": [],
                    "intensities": frame_intensities,
                }
                report_data[frame_idx] = {
                    "frame_number": frame_idx,
                    "persons_detected": len(frame_polygons),
                    "avg_confidence": 1.0,
                    "min_confidence": 1.0,
                    "max_confidence": 1.0,
                    "motion_zones": 0,
                    "sliding_window_hits": 0,
                    "multiscale_hits": 0,
                    "post_check_alerts": 0,
                }
                pbar.update(1)
        finally:
            pbar.close()
            cap.release()

        predictor.reset_state(state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_persons = sum(len(a["auto"]) for a in annotations.values())
        n_empty = sum(1 for a in annotations.values() if not a["auto"])
        print(f"\n  Istanze totali rilevate:        {n_persons:,}")
        print(f"  Frame con 0 rilevamenti:        {n_empty}")

        return annotations, report_data
