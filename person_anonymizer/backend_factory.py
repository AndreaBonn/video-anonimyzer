"""Factory per il backend di detection — crea il modello giusto in base alla config."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig

__all__ = ["DetectionBackend", "load_detection_backend"]


@dataclass
class DetectionBackend:
    """Container per modello YOLO + eventuale SAM3 refiner/detector.

    Attributes
    ----------
    yolo_model : object
        Modello YOLO (sempre caricato, serve anche per post-render check).
    sam3_refiner : object | None
        ``Sam3ImageRefiner`` per modalità ``yolo+sam3``.
    sam3_video_detector : object | None
        ``Sam3VideoDetector`` per modalità ``sam3``.
    backend_name : str
        Nome del backend attivo.
    """

    yolo_model: object
    sam3_refiner: object = None
    sam3_video_detector: object = None
    backend_name: str = "yolo"


def load_detection_backend(config: PipelineConfig) -> DetectionBackend:
    """Crea il backend di detection in base a ``config.detection_backend``.

    Parameters
    ----------
    config : PipelineConfig
        Configurazione pipeline.

    Returns
    -------
    DetectionBackend
        Backend con modelli caricati.

    Raises
    ------
    ImportError
        Se SAM3 è richiesto ma non disponibile.
    """
    from ultralytics import YOLO

    backend = config.detection_backend

    print(f"\nCaricamento modello {config.yolo_model}...")
    yolo_model = YOLO(config.yolo_model)

    sam3_refiner = None
    sam3_video_detector = None

    if backend in ("yolo+sam3", "sam3"):
        from .sam3_backend import Sam3ImageRefiner, Sam3VideoDetector, check_sam3_available

        if not check_sam3_available():
            from .sam3_backend import _require_sam3

            _require_sam3()

        if backend == "yolo+sam3":
            print(f"Caricamento SAM3 refiner ({config.sam3_model})...")
            sam3_refiner = Sam3ImageRefiner(
                model_path=config.sam3_model,
                epsilon_ratio=config.sam3_mask_simplify_epsilon,
                min_mask_area=config.sam3_min_mask_area,
            )
        else:
            print(f"Caricamento SAM3 video detector ({config.sam3_model})...")
            sam3_video_detector = Sam3VideoDetector(
                model_path=config.sam3_model,
                text_prompt=config.sam3_text_prompt,
                epsilon_ratio=config.sam3_mask_simplify_epsilon,
                min_mask_area=config.sam3_min_mask_area,
            )

    return DetectionBackend(
        yolo_model=yolo_model,
        sam3_refiner=sam3_refiner,
        sam3_video_detector=sam3_video_detector,
        backend_name=backend,
    )
