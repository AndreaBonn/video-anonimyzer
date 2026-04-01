<div align="center">

# Person Anonymizer

**Automatic person anonymization in surveillance videos using YOLO v8 multi-scale detection, ByteTrack tracking, and temporal smoothing.**

Created by <a href="https://andreabonn.github.io/" target="_blank">Andrea Bonacci</a>

[English](#english) | [Italiano](#italiano)

</div>

---

<a id="english"></a>
## English

[Passa all'italiano](#italiano)

### What it does

CLI and web tool for automatic person anonymization in surveillance videos. Designed for fixed cameras with wide-angle lenses where people may appear small (30–100 px).

### 🚀 Features

- **Multi-scale YOLO v8 detection** — inference at 4 scales (1.0x–2.5x) + 3x3 sliding window + Test-Time Augmentation
- **ByteTrack tracking** — persistent person IDs across consecutive frames
- **Temporal smoothing EMA** — stabilizes bounding boxes with moving average; ghost boxes handle temporary occlusions
- **Auto-refinement** — re-analyzes the rendered video and adds missed detections (up to 3 passes)
- **Manual review** — interactive OpenCV (CLI) or browser (web) interface to add/edit/delete polygons
- **Two anonymization methods** — pixelation (default) or Gaussian blur
- **Adaptive intensity** — obscuring strength proportional to person size
- **Post-render verification** — second YOLO pass on the anonymized video to flag residual detections
- **Optional fish-eye correction** — optical undistortion via camera calibration
- **Complete output set** — anonymized H.264 video, debug video, CSV report, reusable JSON annotations

### 📦 Requirements

- Python 3.10+
- ffmpeg (for H.264 encoding and audio preservation)
- ~150 MB disk space for YOLO models (downloaded automatically on first run)
- CUDA GPU recommended; CPU also supported

**Install ffmpeg:**

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 🔧 Installation

```bash
git clone https://github.com/AndreaBonn/PRIVATE__video-anonimyzer.git
cd PRIVATE__video-anonimyzer
python -m venv person_anonymizer/.venv
source person_anonymizer/.venv/bin/activate
# Windows: person_anonymizer\.venv\Scripts\activate
pip install -r requirements.txt
```

### 🎯 CLI Usage

```bash
# Standard — automatic detection + manual review (recommended)
python -m person_anonymizer.cli video.mp4

# Fully automatic (no manual review)
python -m person_anonymizer.cli video.mp4 -M auto

# Specify output and method
python -m person_anonymizer.cli video.mp4 -o output.mp4 -m blur

# Disable debug video and CSV report
python -m person_anonymizer.cli video.mp4 --no-debug --no-report

# Reload JSON annotations and reopen review
python -m person_anonymizer.cli video.mp4 --review annotations.json

# Normalize annotations (merge overlapping polygons)
python -m person_anonymizer.cli video.mp4 --review annotations.json --normalize
```

**CLI options:**

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Path to input video | (required) |
| `-M, --mode` | `manual` (with review) or `auto` | `manual` |
| `-o, --output` | Output file path | `<input>_anonymized.mp4` |
| `-m, --method` | `pixelation` or `blur` | `pixelation` |
| `--no-debug` | Disable debug video | `False` |
| `--no-report` | Disable CSV report | `False` |
| `--review` | Reload annotations from JSON | `None` |
| `--normalize` | Normalize polygons (requires --review) | `False` |

### 🌐 Web Interface

```bash
python -m person_anonymizer.web.app
# Open http://127.0.0.1:5000
```

The web GUI allows you to:
- Upload videos via drag & drop
- Configure all pipeline parameters
- Monitor progress in real time (SSE)
- Review annotations frame by frame in the browser
- Download all outputs (video, debug, report, JSON)

**Environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | Random (generated at startup) |
| `FLASK_HOST` | Web server host | `127.0.0.1` |
| `FLASK_PORT` | Web server port | `5000` |

### ⚙️ Pipeline (5 stages)

1. **Detection** — YOLO v8 multi-scale + sliding window + TTA, with optional motion detection
2. **Auto-refinement** — Re-render + second YOLO pass, up to 3 iterations
3. **Manual review** — Interactive interface (OpenCV or web) for corrections
4. **Rendering** — Apply anonymization to the original video (FFV1 lossless intermediate)
5. **Post-processing** — H.264 encoding with ffmpeg, audio preservation, report saving

### 📁 Output Files

| File | Description |
|------|-------------|
| `*_anonymized.mp4` | Video with persons obscured (H.264) |
| `*_debug.mp4` | Video with colored detection overlays |
| `*_report.csv` | Per-frame report (confidence, detections, motion) |
| `*_annotations.json` | Full annotations (reusable with --review) |

### 🗂 Supported Formats

`.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`

### 🔬 Advanced Configuration

All 40+ parameters are configurable via `PipelineConfig` or the web GUI. Key parameters:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `detection_confidence` | YOLO confidence threshold | 0.20 | 0.01–0.99 |
| `anonymization_intensity` | Obscuring strength | 10 | 1–100 |
| `person_padding` | Padding around person (px) | 15 | 0–200 |
| `yolo_model` | YOLO model | `yolov8x.pt` | `yolov8x.pt`, `yolov8n.pt` |
| `enable_sliding_window` | 3x3 sliding window grid | `True` | |
| `enable_tracking` | ByteTrack tracking | `True` | |
| `enable_temporal_smoothing` | EMA + ghost boxes | `True` | |
| `smoothing_alpha` | EMA weight (1 = no smoothing) | 0.35 | 0.01–1.0 |
| `ghost_frames` | Ghost frames for occlusions | 10 | 0–120 |
| `enable_adaptive_intensity` | Intensity proportional to size | `True` | |
| `max_refinement_passes` | Auto-refinement iterations | 3 | 1–10 |

### 🗃 Project Structure

```
person_anonymizer/
├── config.py            # PipelineConfig with validation
├── models.py            # Dataclasses (PipelineContext, OutputPaths, etc.)
├── pipeline.py          # Pipeline orchestrator
├── pipeline_stages.py   # Stages: detection, refinement, review
├── output.py            # Output saving and JSON loading
├── cli.py               # CLI entry point
├── detection.py         # YOLO multi-scale + NMS
├── tracking.py          # ByteTrack + TemporalSmoother
├── anonymization.py     # Obscuring + polygon geometry
├── preprocessing.py     # CLAHE, fisheye, motion detection
├── postprocessing.py    # H.264 encoding, post-render check
├── rendering.py         # Video rendering + review stats
├── manual_reviewer.py   # OpenCV manual review GUI
├── camera_calibration.py# Camera calibration utility
└── web/                 # Flask web interface
    ├── app.py           # Flask routes + SSE + security
    ├── pipeline_runner.py
    ├── sse_manager.py
    └── review_state.py
tests/                   # 218 tests (pytest)
reports/                 # Audit reports
```

### 🛠 Development

```bash
source person_anonymizer/.venv/bin/activate
pytest tests/ -v
ruff check person_anonymizer/
```

### 🔒 Security

See [SECURITY.md](SECURITY.md) for full details on implemented protections.

### 🧰 Technologies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
- [OpenCV](https://opencv.org/) — Video processing
- [Flask](https://flask.palletsprojects.com/) — Web interface
- [ffmpeg](https://ffmpeg.org/) — Video encoding

### 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

> **Note**: This project depends on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) which is licensed under AGPL-3.0. If you use this software as a network service, the AGPL requires that the complete source code be made available. Since this project is already open source, there is no practical conflict. For commercial/proprietary use of YOLO, see [Ultralytics Licensing](https://www.ultralytics.com/license).

---

<a id="italiano"></a>
## Italiano

[Switch to English](#english)

### Cosa fa

Tool CLI e web per l'anonimizzazione automatica di persone in video di sorveglianza. Progettato per telecamere fisse con lenti grandangolari, dove le persone possono apparire di piccole dimensioni (30–100 px).

### 🚀 Funzionalità

- **Rilevamento YOLO v8 multi-scala** — inferenza a 4 scale (1.0x–2.5x) + sliding window 3x3 + Test-Time Augmentation
- **Tracking ByteTrack** — ID persona persistenti tra frame consecutivi
- **Temporal smoothing EMA** — stabilizza i bounding box con media mobile; ghost box per gestire occlusioni temporanee
- **Auto-refinement** — ri-analizza il video renderizzato e aggiunge detection mancanti (fino a 3 iterazioni)
- **Revisione manuale** — interfaccia interattiva OpenCV (CLI) o browser (web) per aggiungere/modificare/eliminare poligoni
- **Due metodi di oscuramento** — pixelation (default) o blur gaussiano
- **Intensità adattiva** — forza dell'oscuramento proporzionale alla dimensione della persona
- **Verifica post-rendering** — secondo passaggio YOLO sul video anonimizzato per segnalare detection residue
- **Correzione fish-eye opzionale** — undistortion ottica tramite calibrazione camera
- **Output completo** — video H.264 anonimizzato, video debug, report CSV, annotazioni JSON riutilizzabili

### 📦 Requisiti

- Python 3.10+
- ffmpeg (per encoding H.264 e preservazione audio)
- ~150 MB di spazio disco per i modelli YOLO (scaricati automaticamente al primo avvio)
- GPU CUDA raccomandata; funziona anche su CPU

**Installare ffmpeg:**

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 🔧 Installazione

```bash
git clone https://github.com/AndreaBonn/PRIVATE__video-anonimyzer.git
cd PRIVATE__video-anonimyzer
python -m venv person_anonymizer/.venv
source person_anonymizer/.venv/bin/activate
# Windows: person_anonymizer\.venv\Scripts\activate
pip install -r requirements.txt
```

### 🎯 Utilizzo CLI

```bash
# Standard — detection automatica + revisione manuale (consigliato)
python -m person_anonymizer.cli video.mp4

# Completamente automatico (senza revisione)
python -m person_anonymizer.cli video.mp4 -M auto

# Specificare output e metodo
python -m person_anonymizer.cli video.mp4 -o output.mp4 -m blur

# Disabilitare video debug e report CSV
python -m person_anonymizer.cli video.mp4 --no-debug --no-report

# Ricaricare annotazioni JSON e riaprire la revisione
python -m person_anonymizer.cli video.mp4 --review annotazioni.json

# Normalizzare annotazioni (merge poligoni sovrapposti)
python -m person_anonymizer.cli video.mp4 --review annotazioni.json --normalize
```

**Opzioni CLI:**

| Opzione | Descrizione | Default |
|---------|-------------|---------|
| `input` | Percorso video da elaborare | (obbligatorio) |
| `-M, --mode` | `manual` (con revisione) o `auto` | `manual` |
| `-o, --output` | Percorso file di output | `<input>_anonymized.mp4` |
| `-m, --method` | `pixelation` o `blur` | `pixelation` |
| `--no-debug` | Disabilita video debug | `False` |
| `--no-report` | Disabilita CSV report | `False` |
| `--review` | Ricarica annotazioni da JSON | `None` |
| `--normalize` | Normalizza poligoni (richiede --review) | `False` |

### 🌐 Interfaccia Web

```bash
python -m person_anonymizer.web.app
# Apri http://127.0.0.1:5000 nel browser
```

La web GUI permette di:
- Caricare video tramite drag & drop
- Configurare tutti i parametri della pipeline
- Monitorare il progresso in tempo reale (SSE)
- Revisionare le annotazioni frame per frame nel browser
- Scaricare tutti gli output (video, debug, report, JSON)

**Variabili d'ambiente:**

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `FLASK_SECRET_KEY` | Chiave segreta per sessioni Flask | Random (generata all'avvio) |
| `FLASK_HOST` | Host del server web | `127.0.0.1` |
| `FLASK_PORT` | Porta del server web | `5000` |

### ⚙️ Pipeline (5 fasi)

1. **Detection** — YOLO v8 multi-scala + sliding window + TTA, con motion detection opzionale
2. **Auto-refinement** — Re-rendering + secondo passaggio YOLO, fino a 3 iterazioni
3. **Revisione manuale** — Interfaccia interattiva (OpenCV o web) per correzioni
4. **Rendering** — Applicazione oscuramento al video originale (intermedio FFV1 lossless)
5. **Post-processing** — Encoding H.264 con ffmpeg, preservazione audio, salvataggio report

### 📁 File di Output

| File | Descrizione |
|------|-------------|
| `*_anonymized.mp4` | Video con persone oscurate (H.264) |
| `*_debug.mp4` | Video con overlay colorati delle detection |
| `*_report.csv` | Report per-frame (confidenza, detection, motion) |
| `*_annotations.json` | Annotazioni complete (riutilizzabili con --review) |

### 🗂 Formati Supportati

`.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`

### 🔬 Configurazione Avanzata

Tutti i 40+ parametri sono configurabili tramite `PipelineConfig` o la web GUI. I principali:

| Parametro | Descrizione | Default | Range |
|-----------|-------------|---------|-------|
| `detection_confidence` | Soglia confidenza YOLO | 0.20 | 0.01–0.99 |
| `anonymization_intensity` | Intensità oscuramento | 10 | 1–100 |
| `person_padding` | Padding intorno alla persona (px) | 15 | 0–200 |
| `yolo_model` | Modello YOLO | `yolov8x.pt` | `yolov8x.pt`, `yolov8n.pt` |
| `enable_sliding_window` | Griglia sliding window 3x3 | `True` | |
| `enable_tracking` | ByteTrack tracking | `True` | |
| `enable_temporal_smoothing` | EMA + ghost box | `True` | |
| `smoothing_alpha` | Peso EMA (1 = nessuno smoothing) | 0.35 | 0.01–1.0 |
| `ghost_frames` | Frame ghost per occlusioni | 10 | 0–120 |
| `enable_adaptive_intensity` | Intensità proporzionale alla dimensione | `True` | |
| `max_refinement_passes` | Iterazioni auto-refinement | 3 | 1–10 |

### 🗃 Struttura del Progetto

```
person_anonymizer/
├── config.py            # PipelineConfig con validazione
├── models.py            # Dataclass (PipelineContext, OutputPaths, ecc.)
├── pipeline.py          # Orchestratore pipeline
├── pipeline_stages.py   # Fasi: detection, refinement, review
├── output.py            # Salvataggio output e caricamento JSON
├── cli.py               # CLI entry point
├── detection.py         # YOLO multi-scala + NMS
├── tracking.py          # ByteTrack + TemporalSmoother
├── anonymization.py     # Oscuramento + geometria poligoni
├── preprocessing.py     # CLAHE, fisheye, motion detection
├── postprocessing.py    # Encoding H.264, verifica post-render
├── rendering.py         # Rendering video + statistiche review
├── manual_reviewer.py   # GUI OpenCV per revisione manuale
├── camera_calibration.py# Utility calibrazione camera
└── web/                 # Interfaccia web Flask
    ├── app.py           # Flask routes + SSE + security
    ├── pipeline_runner.py
    ├── sse_manager.py
    └── review_state.py
tests/                   # 218 test (pytest)
reports/                 # Report di audit
```

### 🛠 Sviluppo

```bash
source person_anonymizer/.venv/bin/activate
pytest tests/ -v
ruff check person_anonymizer/
```

### 🔒 Sicurezza

Vedi [SECURITY.md](SECURITY.md) per i dettagli completi sulle protezioni implementate.

### 🧰 Tecnologie

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
- [OpenCV](https://opencv.org/) — Video processing
- [Flask](https://flask.palletsprojects.com/) — Interfaccia web
- [ffmpeg](https://ffmpeg.org/) — Video encoding

### 📄 Licenza

Questo progetto è rilasciato sotto [Apache License 2.0](LICENSE).

> **Nota**: Questo progetto dipende da [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), rilasciato sotto AGPL-3.0. Se si utilizza questo software come servizio di rete, l'AGPL richiede che il codice sorgente completo sia reso disponibile. Essendo questo progetto già open source, non c'è conflitto pratico. Per uso commerciale/proprietario di YOLO, vedere [Ultralytics Licensing](https://www.ultralytics.com/license).
