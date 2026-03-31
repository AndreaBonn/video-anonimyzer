# Person Anonymizer — Istruzioni per Claude Code

## Panoramica
Tool di anonimizzazione persone in video di sorveglianza.
Pipeline YOLO v8 multi-scala con tracking ByteTrack, temporal smoothing
e interfaccia web Flask.

## Struttura
- `person_anonymizer/` — Codice sorgente Python
  - `config.py` — PipelineConfig dataclass con tutti i parametri
  - `person_anonymizer.py` — Pipeline principale e CLI
  - `web/` — Interfaccia web Flask con SSE
- `tests/` — Test suite (pytest)

## Comandi
```bash
# Attiva venv
source .venv/bin/activate

# CLI
python person_anonymizer/person_anonymizer.py video.mp4

# Web
python person_anonymizer/web/app.py

# Test
pytest tests/ -v
```

## Convenzioni
- Python: snake_case, docstring NumPy
- Configurazione: tutto in PipelineConfig (config.py), mai globals
- Dipendenze: requirements.txt con versioni pinnate
- Test: pytest, pattern AAA, solo funzioni pure (no cv2/YOLO)
