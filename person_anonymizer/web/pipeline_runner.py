"""
Thread wrapper per eseguire person_anonymizer.run_pipeline() dal web.
Patcha i globals del modulo, cattura stdout/tqdm, invia eventi SSE.
"""

import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from web.sse_manager import SSEManager


# Mappa config web -> variabile globale del modulo
GLOBALS_MAP = {
    "operation_mode": "OPERATION_MODE",
    "anonymization_method": "ANONYMIZATION_METHOD",
    "anonymization_intensity": "ANONYMIZATION_INTENSITY",
    "person_padding": "PERSON_PADDING",
    "edge_padding_multiplier": "EDGE_PADDING_MULTIPLIER",
    "edge_threshold": "EDGE_THRESHOLD",
    "detection_confidence": "DETECTION_CONFIDENCE",
    "nms_iou_internal": "NMS_IOU_INTERNAL",
    "nms_iou_threshold": "NMS_IOU_THRESHOLD",
    "yolo_model": "YOLO_MODEL",
    "enable_fisheye_correction": "ENABLE_FISHEYE_CORRECTION",
    "enable_motion_detection": "ENABLE_MOTION_DETECTION",
    "motion_threshold": "MOTION_THRESHOLD",
    "motion_min_area": "MOTION_MIN_AREA",
    "motion_padding": "MOTION_PADDING",
    "enable_sliding_window": "ENABLE_SLIDING_WINDOW",
    "sliding_window_grid": "SLIDING_WINDOW_GRID",
    "sliding_window_overlap": "SLIDING_WINDOW_OVERLAP",
    "inference_scales": "INFERENCE_SCALES",
    "tta_augmentations": "TTA_AUGMENTATIONS",
    "quality_clahe_clip": "QUALITY_CLAHE_CLIP",
    "quality_clahe_grid": "QUALITY_CLAHE_GRID",
    "quality_darkness_threshold": "QUALITY_DARKNESS_THRESHOLD",
    "enable_tracking": "ENABLE_TRACKING",
    "track_max_age": "TRACK_MAX_AGE",
    "track_match_thresh": "TRACK_MATCH_THRESH",
    "enable_temporal_smoothing": "ENABLE_TEMPORAL_SMOOTHING",
    "smoothing_alpha": "SMOOTHING_ALPHA",
    "ghost_frames": "GHOST_FRAMES",
    "ghost_expansion": "GHOST_EXPANSION",
    "enable_adaptive_intensity": "ENABLE_ADAPTIVE_INTENSITY",
    "adaptive_reference_height": "ADAPTIVE_REFERENCE_HEIGHT",
    "enable_subframe_interpolation": "ENABLE_SUBFRAME_INTERPOLATION",
    "interpolation_fps_threshold": "INTERPOLATION_FPS_THRESHOLD",
    "enable_post_render_check": "ENABLE_POST_RENDER_CHECK",
    "post_render_check_confidence": "POST_RENDER_CHECK_CONFIDENCE",
    "max_refinement_passes": "MAX_REFINEMENT_PASSES",
    "refinement_overlap_threshold": "REFINEMENT_OVERLAP_THRESHOLD",
    "enable_debug_video": "ENABLE_DEBUG_VIDEO",
    "enable_confidence_report": "ENABLE_CONFIDENCE_REPORT",
}


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
                sse.emit(job_id, "phase", {
                    "description": desc,
                    "total": self.total or 0,
                })

            def update(self, n=1):
                super().update(n)
                now = time.monotonic()
                # Rate-limit: emetti max ogni 0.25s
                if now - self._last_emit >= 0.25:
                    self._last_emit = now
                    rate = self.format_dict.get("rate", 0) or 0
                    elapsed = self.format_dict.get("elapsed", 0) or 0
                    self._sse.emit(self._job_id, "progress", {
                        "current": self.n,
                        "total": self.total or 0,
                        "description": self.desc or "",
                        "rate": round(rate, 2),
                        "elapsed": round(elapsed, 1),
                    })

            def close(self):
                # Emetti progresso finale
                self._sse.emit(self._job_id, "progress", {
                    "current": self.total or self.n,
                    "total": self.total or self.n,
                    "description": self.desc or "",
                    "rate": 0,
                    "elapsed": 0,
                })
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

    def __init__(self, sse: SSEManager, job_id: str):
        self._sse = sse
        self._job_id = job_id
        self._original = None
        self._buffer = ""

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
                # Detecta fasi dalla stampa
                phase_match = re.match(r"\[FASE (\d)/5\]", line)
                if phase_match:
                    self._sse.emit(self._job_id, "phase_label", {
                        "phase": int(phase_match.group(1)),
                        "label": line,
                    })
                self._sse.emit(self._job_id, "log", {"message": line})

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

    def start(self, job_id: str, video_path: str, config: dict,
              review_json: str | None = None) -> tuple[bool, str]:
        """Avvia la pipeline. Restituisce (success, message)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False, "Una pipeline è già in esecuzione"
            self._current_job_id = job_id
            self._stop_requested = False

        self._thread = threading.Thread(
            target=self._run,
            args=(job_id, video_path, config, review_json),
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

    def _run(self, job_id: str, video_path: str, config: dict,
             review_json: str | None):
        """Esegue la pipeline nel thread. Patcha globals, cattura output."""

        import person_anonymizer as pa

        # --- Assicura che il cwd sia la dir di person_anonymizer (per modelli YOLO) ---
        original_cwd = os.getcwd()
        pa_dir = str(Path(pa.__file__).resolve().parent)
        os.chdir(pa_dir)

        # --- Salva globals originali ---
        saved_globals = {}
        for web_key, mod_key in GLOBALS_MAP.items():
            saved_globals[mod_key] = getattr(pa, mod_key)

        # --- Applica config dal web ---
        for web_key, mod_key in GLOBALS_MAP.items():
            if web_key in config:
                val = config[web_key]
                # Conversione tipi speciali
                if mod_key == "QUALITY_CLAHE_GRID" and isinstance(val, list):
                    val = tuple(val)
                setattr(pa, mod_key, val)

        # --- Prepara output dir per questo job ---
        job_output = self._output_dir / job_id
        job_output.mkdir(exist_ok=True)

        input_stem = Path(video_path).stem
        output_path = str(job_output / f"{input_stem}_anonymized.mp4")

        # --- Costruisci args namespace ---
        mode = config.get("operation_mode") or pa.OPERATION_MODE
        # In modalità web, forza "auto" per evitare la finestra OpenCV
        if mode == "manual":
            # Esegue detection ma salta la review interattiva
            pass

        args = SimpleNamespace(
            input=video_path,
            mode="auto",  # Sempre auto da web, la review manuale è separata
            method=config.get("anonymization_method"),
            no_debug=not config.get("enable_debug_video", True),
            no_report=not config.get("enable_confidence_report", True),
            review=review_json,
            output=output_path,
            normalize=config.get("normalize", False),
        )

        # --- Installa cattura ---
        tqdm_capture = TqdmCapture(self._sse, job_id)
        stdout_capture = StdoutCapture(self._sse, job_id)

        tqdm_capture.install()
        stdout_capture.install()

        self._sse.emit(job_id, "started", {"job_id": job_id})

        try:
            pa.run_pipeline(args)

            # Successo: elenca file di output
            outputs = []
            for f in sorted(job_output.iterdir()):
                if f.is_file():
                    outputs.append({
                        "name": f.name,
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    })

            self._sse.emit(job_id, "completed", {
                "job_id": job_id,
                "outputs": outputs,
            })

        except SystemExit as e:
            # run_pipeline chiama sys.exit(1) su errori
            self._sse.emit(job_id, "error", {
                "job_id": job_id,
                "message": f"Pipeline terminata con codice {e.code}",
            })

        except Exception as e:
            self._sse.emit(job_id, "error", {
                "job_id": job_id,
                "message": str(e),
            })

        finally:
            # --- Ripristina tutto ---
            stdout_capture.uninstall()
            tqdm_capture.uninstall()

            for mod_key, original_val in saved_globals.items():
                setattr(pa, mod_key, original_val)

            os.chdir(original_cwd)

            self._sse.close(job_id)

            with self._lock:
                self._current_job_id = None
