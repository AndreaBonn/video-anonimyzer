"""
Thread wrapper per eseguire person_anonymizer.run_pipeline() dal web.
Crea PipelineConfig dai parametri web, cattura stdout/tqdm, invia eventi SSE.
"""

import threading
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from person_anonymizer.config import PipelineConfig
from person_anonymizer.models import PipelineContext, PipelineError, PipelineStoppedError
from person_anonymizer.web.config_validator import (
    _ALLOWED_FIELDS,
    _BOOL_FIELDS,  # noqa: F401
    _CONFIG_VALIDATORS,  # noqa: F401
    validate_config_params,
)
from person_anonymizer.web.output_capture import StdoutCapture, TqdmCapture
from person_anonymizer.web.review_state import ReviewState
from person_anonymizer.web.sse_manager import SSEManager

__all__ = ["PipelineRunner", "validate_config_params"]

BYTES_PER_MB = 1024 * 1024
SIZE_MB_DECIMALS = 2


class _JobRequest(NamedTuple):
    """Inputs of one pipeline job, as received from the web layer."""

    job_id: str
    video_path: str
    config_dict: dict
    review_json: str | None


def resolve_yolo_model(model_name: str, package_dir: Path) -> str | None:
    """Resolve a YOLO model name against the package directory.

    Parameters
    ----------
    model_name : str
        Model file name or relative path, as set in the config.
    package_dir : Path
        Directory the model must stay inside.

    Returns
    -------
    str | None
        None if the resolved path escapes ``package_dir``; the absolute path if
        the file exists there; otherwise ``model_name`` unchanged, so that
        ultralytics can download it.
    """
    candidate = (package_dir / model_name).resolve()
    if not candidate.is_relative_to(package_dir.resolve()):
        return None
    return str(candidate) if candidate.exists() else model_name


def list_job_outputs(job_dir: Path) -> list[dict]:
    """List the regular files of a job directory, sorted by name.

    Parameters
    ----------
    job_dir : Path
        Output directory of the job; subdirectories are ignored.

    Returns
    -------
    list[dict]
        One ``{"name", "size_mb"}`` entry per file, size rounded to 2 decimals.
    """
    return [
        {
            "name": f.name,
            "size_mb": round(f.stat().st_size / BYTES_PER_MB, SIZE_MB_DECIMALS),
        }
        for f in sorted(job_dir.iterdir())
        if f.is_file()
    ]


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
        # Il thread può essere fermo in attesa della revisione manuale: va sbloccato
        self.review_state.cancel()
        return True

    def get_status(self) -> dict:
        """Restituisce lo stato corrente."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "running": running,
                "job_id": self._current_job_id if running else None,
            }

    def _emit_error(self, job_id: str, message: str) -> None:
        self._sse.emit(job_id, "error", {"job_id": job_id, "message": message})

    def _fail(self, job_id: str, message: str) -> None:
        """Abort a job that failed before the pipeline started."""
        self._emit_error(job_id, message)
        self._sse.close(job_id)
        with self._lock:
            self._current_job_id = None

    def _build_context(
        self, job: _JobRequest, config: PipelineConfig, output_path: str
    ) -> PipelineContext:
        """Build the pipeline context; manual mode also gets the review state and SSE."""
        ctx = PipelineContext(
            input=job.video_path,
            mode=config.operation_mode,
            method=job.config_dict.get("anonymization_method"),
            output=output_path,
            no_debug=not job.config_dict.get("enable_debug_video", True),
            no_report=not job.config_dict.get("enable_confidence_report", True),
            review=job.review_json,
            normalize=job.config_dict.get("normalize", False),
            stop_event=self._stop_event,
        )
        if config.operation_mode == "manual":
            ctx.review_state = self.review_state
            ctx.sse_manager = self._sse
            ctx.job_id = job.job_id
        return ctx

    def _run(self, job_id: str, video_path: str, config_dict: dict, review_json: str | None):
        """Run the pipeline in the worker thread, capturing its output as SSE events.

        The capture install, the try/except/finally around ``run_pipeline`` and the
        final job-id reset form one lifecycle transaction, so they stay together.
        """
        import logging

        import person_anonymizer as pa
        from person_anonymizer.pipeline import run_pipeline

        log = logging.getLogger(__name__)
        job = _JobRequest(
            job_id=job_id, video_path=video_path, config_dict=config_dict, review_json=review_json
        )

        try:
            config = _build_config(config_dict)
        except ValueError as e:
            self._fail(job_id, f"Configurazione non valida: {e}")
            return

        yolo_model = resolve_yolo_model(
            model_name=config.yolo_model, package_dir=Path(pa.__file__).resolve().parent
        )
        if yolo_model is None:
            self._fail(job_id, "Percorso modello YOLO non autorizzato")
            return
        config.yolo_model = yolo_model

        job_output = self._output_dir / job_id
        job_output.mkdir(exist_ok=True)
        output_path = str(job_output / f"{Path(video_path).stem}_anonymized.mp4")
        ctx = self._build_context(job, config, output_path)

        tqdm_capture = TqdmCapture(self._sse, job_id)
        stdout_capture = StdoutCapture(self._sse, job_id)
        tqdm_capture.install()
        stdout_capture.install()

        self._sse.emit(job_id, "started", {"job_id": job_id})
        completed = False

        try:
            post_check = run_pipeline(ctx, config=config)
            self._sse.emit(
                job_id,
                "completed",
                {
                    "job_id": job_id,
                    "outputs": list_job_outputs(job_output),
                    "post_check": asdict(post_check),
                },
            )
            completed = True

        except PipelineStoppedError:
            # The "stopped" event is emitted once, in the finally block below
            log.info("Pipeline stopped by user for job %s", job_id)

        except PipelineError as e:
            self._emit_error(job_id, str(e))

        except Exception:
            log.exception("Pipeline error for job %s", job_id)
            self._emit_error(job_id, "Errore durante l'elaborazione della pipeline")

        finally:
            stdout_capture.uninstall()
            tqdm_capture.uninstall()

            # A stop that arrives after the last check must not follow "completed"
            if self._stop_event.is_set() and not completed:
                self._sse.emit(
                    job_id,
                    "stopped",
                    {"job_id": job_id, "message": "Pipeline interrotta dall'utente"},
                )

            self._sse.close(job_id)

            with self._lock:
                self._current_job_id = None
