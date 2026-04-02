# Person Anonymizer — Istruzioni per Claude Code

## Panoramica
Tool di anonimizzazione persone in video di sorveglianza.
Pipeline YOLO v8 multi-scala con tracking ByteTrack, temporal smoothing
e interfaccia web Flask.

## Struttura
- `person_anonymizer/` — Package Python installabile (`pip install -e .`)
  - `__init__.py` — Package init, esporta VERSION
  - `config.py` — PipelineConfig dataclass con validazione `__post_init__`, VERSION single source
  - `models.py` — Dataclass: PipelineContext, OutputPaths, VideoMeta, PipelineResult, FrameProcessors, FrameDetectionResult, FisheyeContext, eccezioni
  - `pipeline.py` — Orchestratore pipeline `run_pipeline()`
  - `pipeline_stages.py` — Re-export delle 3 fasi per backward compatibility
  - `stage_detection.py` — Detection loop (YOLO + tracking + smoothing)
  - `stage_refinement.py` — Auto-refinement loop
  - `stage_review.py` — Revisione manuale (web/CLI)
  - `output.py` — Salvataggio output e caricamento annotazioni JSON
  - `cli.py` — CLI entry point
  - `detection.py` — YOLO multi-scala + NMS
  - `tracking.py` — ByteTrack + TemporalSmoother
  - `anonymization.py` — Oscuramento + box/poligono + resolve_intensity
  - `preprocessing.py` — CLAHE, fisheye, motion detection
  - `postprocessing.py` — Encoding, post-render check
  - `normalization.py` — Merge poligoni sovrapposti, normalizzazione annotazioni
  - `rendering.py` — Rendering video + review stats
  - `manual_reviewer.py` — GUI OpenCV per revisione manuale
  - `camera_calibration.py` — Utility calibrazione camera
  - `web/` — Interfaccia web Flask con Blueprint
    - `app.py` — App factory, upload, pipeline, SSE
    - `extensions.py` — Limiter e helper condivisi
    - `middleware.py` — CSRF, security headers, request ID
    - `routes_review.py` — Blueprint endpoint review
    - `routes_output.py` — Blueprint endpoint download/config
    - `config_validator.py` — Validazione parametri config web
    - `output_capture.py` — TqdmCapture + StdoutCapture per SSE
    - `pipeline_runner.py` — Thread wrapper per run_pipeline
    - `sse_manager.py` — Server-Sent Events manager
    - `review_state.py` — Stato review thread-safe
- `tests/` — Test suite (pytest, 241 test)
- `reports/` — Report di audit (code roast, security, architecture)
- `requirements.txt` — Dipendenze produzione pinnate
- `requirements-dev.txt` — Dipendenze sviluppo (pytest, ruff)
- `pyproject.toml` — Build system, CLI entry point, ruff, pytest
- `.env.example` — Variabili d'ambiente di esempio
- `SECURITY.md` — Documentazione sicurezza

## Comandi
```bash
# Attiva venv
source person_anonymizer/.venv/bin/activate

# CLI
python -m person_anonymizer.cli video.mp4

# Web
python -m person_anonymizer.web.app

# Test
pytest tests/ -v
```

## Convenzioni
- Python: snake_case, docstring NumPy, import assoluti (`from person_anonymizer.xxx import`)
- Installazione: `pip install -e .` (package installabile via pyproject.toml)
- Configurazione: tutto in PipelineConfig (config.py) con validazione __post_init__
- Dipendenze: requirements.txt (prod) + requirements-dev.txt (dev), versioni pinnate
- Test: pytest, pattern AAA, import con prefisso `person_anonymizer.`
- Pipeline interface: PipelineContext, FisheyeContext, FrameDetectionResult dataclass
- Web: Flask Blueprint pattern, middleware separato, extensions condivise
- Security: CSRF via X-Requested-With, rate limiting con default, cleanup automatico, secret key da env
