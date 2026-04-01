"""
Thread wrapper per eseguire person_anonymizer.run_pipeline() dal web.
Crea PipelineConfig dai parametri web, cattura stdout/tqdm, invia eventi SSE.
"""

import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from web.sse_manager import SSEManager
from web.review_state import ReviewState
from config import PipelineConfig

_CONFIG_VALIDATORS = {
    "operation_mode": lambda v: isinstance(v, str) and v in ("auto", "manual"),
    "anonymization_method": lambda v: isinstance(v, str) and v in ("pixelation", "blur"),
    "anonymization_intensity": lambda v: isinstance(v, (int, float)) and 1 <= v <= 100,
    "person_padding": lambda v: isinstance(v, (int, float)) and 0 <= v <= 200,
    "detection_confidence": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.99,
    "nms_iou_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "nms_iou_internal": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "yolo_model": lambda v: isinstance(v, str) and v in ("yolov8x.pt", "yolov8n.pt"),
    "sliding_window_grid": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "sliding_window_overlap": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 0.9,
    "max_refinement_passes": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "smoothing_alpha": lambda v: isinstance(v, (int, float)) and 0.0 < v <= 1.0,
    "ghost_frames": lambda v: isinstance(v, int) and 0 <= v <= 120,
    "ghost_expansion": lambda v: isinstance(v, (int, float)) and 1.0 <= v <= 2.0,
    "track_max_age": lambda v: isinstance(v, int) and 1 <= v <= 300,
    "track_match_thresh": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "adaptive_reference_height": lambda v: isinstance(v, int) and 10 <= v <= 500,
    "post_render_check_confidence": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.99,
    "refinement_overlap_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
    "edge_padding_multiplier": lambda v: isinstance(v, (int, float)) and 1.0 <= v <= 5.0,
    "edge_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 0.5,
    "motion_threshold": lambda v: isinstance(v, int) and 1 <= v <= 255,
    "motion_min_area": lambda v: isinstance(v, int) and 1 <= v <= 100000,
    "motion_padding": lambda v: isinstance(v, int) and 0 <= v <= 500,
    "quality_clahe_clip": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 10.0,
    "quality_darkness_threshold": lambda v: isinstance(v, int) and 0 <= v <= 255,
    "interpolation_fps_threshold": lambda v: isinstance(v, int) and 1 <= v <= 120,
}

# Campi booleani: accettano solo bool
_BOOL_FIELDS = {
    "enable_fisheye_correction",
    "enable_motion_detection",
    "enable_sliding_window",
    "enable_tracking",
    "enable_temporal_smoothing",
    "enable_adaptive_intensity",
    "enable_subframe_interpolation",
    "enable_post_render_check",
    "enable_debug_video",
    "enable_confidence_report",
}


def validate_config_params(web_config: dict) -> tuple[bool, str]:
    """
    Valida i parametri di configurazione prima di applicarli.

    Parameters
    ----------
    web_config : dict
        Parametri dalla form web.

    Returns
    -------
    tuple[bool, str]
        (True, "") se valido, (False, messaggio_errore) altrimenti.
    """
    for key, value in web_config.items():
        if key in _BOOL_FIELDS:
            if not isinstance(value, bool):
                return False, f"Parametro '{key}' deve essere booleano"
        elif key in _CONFIG_VALIDATORS:
            if not _CONFIG_VALIDATORS[key](value):
                return False, f"Parametro '{key}' non valido: {value!r}"
    return True, ""


def _build_config(web_config: dict) -> PipelineConfig:
    """Crea PipelineConfig dai parametri dell'interfaccia web.

    Parameters
    ----------
    web_config : dict
        Dizionario di parametri provenienti dalla form web.

    Returns
    -------
    PipelineConfig
        Istanza configurata con i valori ricevuti; i campi non presenti
        in web_config mantengono il valore di default di PipelineConfig.

    Raises
    ------
    ValueError
        Se un parametro non supera la validazione.
    """
    valid, msg = validate_config_params(web_config)
    if not valid:
        raise ValueError(msg)

    field_map = {
        "operation_mode": "operation_mode",
        "anonymization_method": "anonymization_method",
        "anonymization_intensity": "anonymization_intensity",
        "person_padding": "person_padding",
        "detection_confidence": "detection_confidence",
        "nms_iou_threshold": "nms_iou_threshold",
        "yolo_model": "yolo_model",
        "enable_fisheye_correction": "enable_fisheye_correction",
        "enable_motion_detection": "enable_motion_detection",
        "motion_threshold": "motion_threshold",
        "motion_min_area": "motion_min_area",
        "motion_padding": "motion_padding",
        "enable_sliding_window": "enable_sliding_window",
        "sliding_window_grid": "sliding_window_grid",
        "sliding_window_overlap": "sliding_window_overlap",
        "inference_scales": "inference_scales",
        "tta_augmentations": "tta_augmentations",
        "quality_clahe_clip": "quality_clahe_clip",
        "quality_clahe_grid": "quality_clahe_grid",
        "quality_darkness_threshold": "quality_darkness_threshold",
        "enable_tracking": "enable_tracking",
        "track_max_age": "track_max_age",
        "track_match_thresh": "track_match_thresh",
        "enable_temporal_smoothing": "enable_temporal_smoothing",
        "smoothing_alpha": "smoothing_alpha",
        "ghost_frames": "ghost_frames",
        "ghost_expansion": "ghost_expansion",
        "enable_adaptive_intensity": "enable_adaptive_intensity",
        "adaptive_reference_height": "adaptive_reference_height",
        "enable_subframe_interpolation": "enable_subframe_interpolation",
        "interpolation_fps_threshold": "interpolation_fps_threshold",
        "enable_post_render_check": "enable_post_render_check",
        "post_render_check_confidence": "post_render_check_confidence",
        "max_refinement_passes": "max_refinement_passes",
        "refinement_overlap_threshold": "refinement_overlap_threshold",
        "enable_debug_video": "enable_debug_video",
        "enable_confidence_report": "enable_confidence_report",
        "edge_padding_multiplier": "edge_padding_multiplier",
        "edge_threshold": "edge_threshold",
        "nms_iou_internal": "nms_iou_internal",
    }
    kwargs = {}
    for web_key, config_key in field_map.items():
        if web_key in web_config:
            val = web_config[web_key]
            if config_key == "quality_clahe_grid" and isinstance(val, list):
                val = tuple(val)
            kwargs[config_key] = val
    return PipelineConfig(**kwargs)


class TqdmCapture:
    """Monkey-patch tqdm per catturare il progresso ed emetterlo via SSE."""

    def __init__(self, sse: SSEManager, job_id: str):
        self._sse = sse
        self._job_id = job_id
        self._original_tqdm = None

    def install(self):
        """Installa il patch su tqdm (sia nel modulo tqdm che in person_anonymizer)."""
        import tqdm as tqdm_module
        import person_anonymizer as pa

        self._original_tqdm = tqdm_module.tqdm

        sse = self._sse
        job_id = self._job_id

        class PatchedTqdm(self._original_tqdm):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._sse = sse
                self._job_id = job_id
                self._last_emit = 0
                # Emetti evento inizio fase
                desc = self.desc or ""
                sse.emit(
                    job_id,
                    "phase",
                    {
                        "description": desc,
                        "total": self.total or 0,
                    },
                )

            def update(self, n=1):
                super().update(n)
                now = time.monotonic()
                # Rate-limit: emetti max ogni 0.25s
                if now - self._last_emit >= 0.25:
                    self._last_emit = now
                    rate = self.format_dict.get("rate", 0) or 0
                    elapsed = self.format_dict.get("elapsed", 0) or 0
                    self._sse.emit(
                        self._job_id,
                        "progress",
                        {
                            "current": self.n,
                            "total": self.total or 0,
                            "description": self.desc or "",
                            "rate": round(rate, 2),
                            "elapsed": round(elapsed, 1),
                        },
                    )

            def close(self):
                # Emetti progresso finale
                self._sse.emit(
                    self._job_id,
                    "progress",
                    {
                        "current": self.total or self.n,
                        "total": self.total or self.n,
                        "description": self.desc or "",
                        "rate": 0,
                        "elapsed": 0,
                    },
                )
                super().close()

        # Patcha sia il modulo tqdm che il riferimento in person_anonymizer
        tqdm_module.tqdm = PatchedTqdm
        pa.tqdm = PatchedTqdm

    def uninstall(self):
        """Ripristina tqdm originale."""
        if self._original_tqdm:
            import tqdm as tqdm_module
            import person_anonymizer as pa

            tqdm_module.tqdm = self._original_tqdm
            pa.tqdm = self._original_tqdm


class StdoutCapture:
    """Cattura stdout e invia le righe come eventi SSE 'log'."""

    _PATH_RE = re.compile(r"/[^\s]*/(uploads|outputs)/[^\s]+")

    def __init__(self, sse: SSEManager, job_id: str):
        self._sse = sse
        self._job_id = job_id
        self._original = None
        self._buffer = ""

    @classmethod
    def _sanitize_message(cls, msg: str) -> str:
        """Rimuove path assoluti dai messaggi di log inviati al client."""
        return cls._PATH_RE.sub("[FILE]", msg)

    def install(self):
        self._original = sys.stdout
        sys.stdout = self

    def uninstall(self):
        if self._original:
            sys.stdout = self._original

    def write(self, text):
        if self._original:
            self._original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                sanitized = self._sanitize_message(line)
                # Detecta fasi dalla stampa
                phase_match = re.match(r"\[FASE (\d)/5\]", sanitized)
                if phase_match:
                    self._sse.emit(
                        self._job_id,
                        "phase_label",
                        {
                            "phase": int(phase_match.group(1)),
                            "label": sanitized,
                        },
                    )
                self._sse.emit(self._job_id, "log", {"message": sanitized})

    def flush(self):
        if self._original:
            self._original.flush()


class PipelineRunner:
    """Gestisce l'esecuzione della pipeline in un thread separato."""

    def __init__(self, sse: SSEManager, output_dir: Path):
        self._sse = sse
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._current_job_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self.review_state = ReviewState()

    def start(
        self, job_id: str, video_path: str, config_dict: dict, review_json: str | None = None
    ) -> tuple[bool, str]:
        """Avvia la pipeline. Restituisce (success, message)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False, "Una pipeline è già in esecuzione"
            self._current_job_id = job_id
            self._stop_requested = False

        self._thread = threading.Thread(
            target=self._run,
            args=(job_id, video_path, config_dict, review_json),
            daemon=True,
        )
        self._thread.start()
        return True, "Pipeline avviata"

    def stop(self, job_id: str | None = None) -> bool:
        """Richiede l'interruzione della pipeline."""
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return False
            if job_id and job_id != self._current_job_id:
                return False
            self._stop_requested = True
        return True

    def get_status(self) -> dict:
        """Restituisce lo stato corrente."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "running": running,
                "job_id": self._current_job_id if running else None,
            }

    def _run(self, job_id: str, video_path: str, config_dict: dict, review_json: str | None):
        """Esegue la pipeline nel thread. Crea PipelineConfig, cattura output."""
        import logging

        import person_anonymizer as pa

        _log = logging.getLogger(__name__)

        # --- Crea config dalla web form (con validazione) ---
        try:
            config = _build_config(config_dict)
        except ValueError as e:
            self._sse.emit(
                job_id,
                "error",
                {
                    "job_id": job_id,
                    "message": f"Configurazione non valida: {e}",
                },
            )
            self._sse.close(job_id)
            with self._lock:
                self._current_job_id = None
            return

        # --- Assicura path assoluto per il modello YOLO ---
        pa_dir = Path(pa.__file__).resolve().parent
        yolo_path = pa_dir / config.yolo_model
        if yolo_path.exists():
            config.yolo_model = str(yolo_path)

        # --- Prepara output dir per questo job ---
        job_output = self._output_dir / job_id
        job_output.mkdir(exist_ok=True)

        input_stem = Path(video_path).stem
        output_path = str(job_output / f"{input_stem}_anonymized.mp4")

        # --- Costruisci args namespace ---
        mode = config.operation_mode

        args = SimpleNamespace(
            input=video_path,
            mode=mode,
            method=config_dict.get("anonymization_method"),
            no_debug=not config_dict.get("enable_debug_video", True),
            no_report=not config_dict.get("enable_confidence_report", True),
            review=review_json,
            output=output_path,
            normalize=config_dict.get("normalize", False),
        )

        # In modalità manual da web, passa lo stato review e il manager SSE
        if mode == "manual":
            args._review_state = self.review_state
            args._sse_manager = self._sse
            args._job_id = job_id

        # --- Installa cattura ---
        tqdm_capture = TqdmCapture(self._sse, job_id)
        stdout_capture = StdoutCapture(self._sse, job_id)

        tqdm_capture.install()
        stdout_capture.install()

        self._sse.emit(job_id, "started", {"job_id": job_id})

        try:
            pa.run_pipeline(args, config=config)

            # Successo: elenca file di output
            outputs = []
            for f in sorted(job_output.iterdir()):
                if f.is_file():
                    outputs.append(
                        {
                            "name": f.name,
                            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                        }
                    )

            self._sse.emit(
                job_id,
                "completed",
                {
                    "job_id": job_id,
                    "outputs": outputs,
                },
            )

        except SystemExit as e:
            # run_pipeline chiama sys.exit(1) su errori
            exit_code = e.code if isinstance(e.code, int) else 1
            self._sse.emit(
                job_id,
                "error",
                {
                    "job_id": job_id,
                    "message": f"Pipeline terminata con codice {exit_code}",
                },
            )

        except Exception:
            _log.exception("Pipeline error for job %s", job_id)
            self._sse.emit(
                job_id,
                "error",
                {
                    "job_id": job_id,
                    "message": "Errore durante l'elaborazione della pipeline",
                },
            )

        finally:
            # --- Ripristina cattura output ---
            stdout_capture.uninstall()
            tqdm_capture.uninstall()

            self._sse.close(job_id)

            with self._lock:
                self._current_job_id = None
