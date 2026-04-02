"""
Thread wrapper per eseguire person_anonymizer.run_pipeline() dal web.
Crea PipelineConfig dai parametri web, cattura stdout/tqdm, invia eventi SSE.
"""

import threading
from pathlib import Path

from person_anonymizer.web.sse_manager import SSEManager
from person_anonymizer.web.review_state import ReviewState
from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import PipelineContext, PipelineError
from person_anonymizer.web.config_validator import (
    _CONFIG_VALIDATORS,  # noqa: F401
    _BOOL_FIELDS,  # noqa: F401
    _ALLOWED_FIELDS,
    validate_config_params,
)
from person_anonymizer.web.output_capture import TqdmCapture, StdoutCapture

__all__ = ["PipelineRunner", "validate_config_params"]


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

    kwargs = {}
    for key, val in web_config.items():
        if key in _ALLOWED_FIELDS:
            if key == "quality_clahe_grid" and isinstance(val, list):
                val = tuple(val)
            kwargs[key] = val
    return PipelineConfig(**kwargs)


class PipelineRunner:
    """Gestisce l'esecuzione della pipeline in un thread separato."""

    def __init__(self, sse: SSEManager, output_dir: Path):
        self._sse = sse
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._current_job_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.review_state = ReviewState()

    def start(
        self, job_id: str, video_path: str, config_dict: dict, review_json: str | None = None
    ) -> tuple[bool, str]:
        """Avvia la pipeline. Restituisce (success, message)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False, "Una pipeline è già in esecuzione"
            self._current_job_id = job_id
            self._stop_event.clear()

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
            self._stop_event.set()
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
        from person_anonymizer.pipeline import run_pipeline

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
        yolo_resolved = yolo_path.resolve()
        if not str(yolo_resolved).startswith(str(pa_dir.resolve())):
            self._sse.emit(
                job_id,
                "error",
                {"job_id": job_id, "message": "Percorso modello YOLO non autorizzato"},
            )
            self._sse.close(job_id)
            with self._lock:
                self._current_job_id = None
            return
        if yolo_resolved.exists():
            config.yolo_model = str(yolo_resolved)

        # --- Prepara output dir per questo job ---
        job_output = self._output_dir / job_id
        job_output.mkdir(exist_ok=True)

        input_stem = Path(video_path).stem
        output_path = str(job_output / f"{input_stem}_anonymized.mp4")

        # --- Costruisci contesto pipeline ---
        mode = config.operation_mode

        ctx = PipelineContext(
            input=video_path,
            mode=mode,
            method=config_dict.get("anonymization_method"),
            output=output_path,
            no_debug=not config_dict.get("enable_debug_video", True),
            no_report=not config_dict.get("enable_confidence_report", True),
            review=review_json,
            normalize=config_dict.get("normalize", False),
            stop_event=self._stop_event,
        )

        # In modalità manual da web, passa lo stato review e il manager SSE
        if mode == "manual":
            ctx.review_state = self.review_state
            ctx.sse_manager = self._sse
            ctx.job_id = job_id

        # --- Installa cattura ---
        tqdm_capture = TqdmCapture(self._sse, job_id)
        stdout_capture = StdoutCapture(self._sse, job_id)

        tqdm_capture.install()
        stdout_capture.install()

        self._sse.emit(job_id, "started", {"job_id": job_id})

        try:
            run_pipeline(ctx, config=config)

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

        except PipelineError as e:
            self._sse.emit(
                job_id,
                "error",
                {
                    "job_id": job_id,
                    "message": str(e),
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

            if self._stop_event.is_set():
                self._sse.emit(
                    job_id,
                    "stopped",
                    {"job_id": job_id, "message": "Pipeline interrotta dall'utente"},
                )

            self._sse.close(job_id)

            with self._lock:
                self._current_job_id = None
