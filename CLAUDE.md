# Person Anonymizer — Istruzioni per Claude Code

## Panoramica
Tool di anonimizzazione persone in video di sorveglianza.
Pipeline YOLO v8 multi-scala con tracking ByteTrack, temporal smoothing
e interfaccia web Flask.

## Struttura
- `person_anonymizer/` — Package Python (import relativi)
  - `__init__.py` — Package init, esporta VERSION
  - `config.py` — PipelineConfig dataclass con validazione `__post_init__`
  - `models.py` — Dataclass: PipelineContext, OutputPaths, VideoMeta, PipelineResult, FrameProcessors, eccezioni
  - `pipeline.py` — Orchestratore pipeline `run_pipeline()`
  - `pipeline_stages.py` — Fasi: detection loop, refinement loop, manual review
  - `output.py` — Salvataggio output e caricamento annotazioni JSON
  - `cli.py` — CLI entry point
  - `person_anonymizer.py` — Facade backward-compat (re-export)
  - `detection.py` — YOLO multi-scala + NMS
  - `tracking.py` — ByteTrack + TemporalSmoother
  - `anonymization.py` — Oscuramento + box/poligono + resolve_intensity
  - `preprocessing.py` — CLAHE, fisheye, motion detection
  - `postprocessing.py` — Encoding, post-render check, normalizzazione
  - `rendering.py` — Rendering video + review stats
  - `manual_reviewer.py` — GUI OpenCV per revisione manuale
  - `camera_calibration.py` — Utility calibrazione camera
  - `web/` — Interfaccia web Flask con SSE
- `tests/` — Test suite (pytest, 218 test)
- `reports/` — Report di audit (code roast, security, architecture, test quality)
- `requirements.txt` — Dipendenze pinnate (root del progetto)
- `pyproject.toml` — Configurazione ruff e pytest
- `.env.example` — Variabili d'ambiente di esempio

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
- Python: snake_case, docstring NumPy, import relativi dentro il package
- Configurazione: tutto in PipelineConfig (config.py) con validazione __post_init__
- Dipendenze: requirements.txt alla root con versioni pinnate
- Test: pytest, pattern AAA, import con prefisso `person_anonymizer.`
- Pipeline interface: PipelineContext dataclass (non SimpleNamespace)
- Security: CSRF check, rate limiting, cleanup automatico upload, secret key da env
