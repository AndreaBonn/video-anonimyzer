# Code Roast Report - Person Anonymizer v7.1

**Data**: 2026-04-01
**Analista**: Claude Opus 4.6
**Scope**: Tutti i file Python in `person_anonymizer/`, `tests/`, root

---

## Sommario Esecutivo

Il progetto ha una buona architettura modulare e una separazione delle responsabilita ragionevole. La test suite copre le funzioni pure ed e ben strutturata. Tuttavia, ci sono problemi significativi nell'area della gestione delle risorse, type safety, accoppiamento tra componenti, e mancanza di un `__init__.py` che rende il package non importabile come tale. Di seguito tutti i problemi trovati, ordinati per severita.

---

## CRITICAL (rischio di crash, data loss, o vulnerabilita di sicurezza)

### C-01 | `person_anonymizer.py` -- God File da 995 righe
**File**: `person_anonymizer/person_anonymizer.py`
**Linee**: 1-995

Il file principale supera le 300 righe (limite del progetto) di oltre il triplo. Contiene 5 dataclass, 8 funzioni private di pipeline, il CLI parser e il main. Questo viola direttamente il principio SRP e rende il file difficile da navigare, testare e mantenere.

**Fix**: Estrarre in moduli separati:
- `pipeline_types.py` per le dataclass (`OutputPaths`, `VideoMeta`, `PipelineResult`, `FrameProcessors`)
- `pipeline_phases.py` per `_run_detection_loop`, `_run_refinement_loop`, `_run_manual_review`
- `pipeline_io.py` per `_save_outputs`, `_load_annotations_from_json`
- `cli.py` per `parse_args` e `main`

---

### C-02 | `manual_reviewer.py` -- 477 righe, God Class
**File**: `person_anonymizer/manual_reviewer.py`
**Linee**: 1-477

`ManualReviewer` ha 477 righe e gestisce: rendering, input mouse, input tastiera, cache frame, coordinate scaling, statistiche, interazione con OpenCV. Supera ampiamente il limite di 200 righe per classe e 300 per file.

**Fix**: Estrarre almeno `FrameRenderer`, `InputHandler` e `ReviewStatistics` come classi separate.

---

### C-03 | Resource leak: `cv2.VideoCapture` senza context manager ne try/finally
**File**: `person_anonymizer/postprocessing.py`, linee 122-161
**File**: `person_anonymizer/rendering.py`, linee 67-144

In `run_post_render_check`, se un'eccezione viene lanciata durante il loop (es. errore YOLO, OOM), `cap.release()` non viene mai chiamato. Il `VideoCapture` resta aperto, consumando file descriptor e potenzialmente bloccando il file.

```python
# postprocessing.py:122 — cap mai rilasciato in caso di eccezione
cap = cv2.VideoCapture(anonymized_video_path)
# ... 40 righe di loop senza try/finally ...
cap.release()  # mai raggiunto se c'e un'eccezione
```

Stesso problema in `render_video`: se `obscure_polygon` o `draw_debug_polygons` lanciano un'eccezione, cap/writers restano aperti.

**Fix**: Wrappare in `try/finally` o creare un context manager per `VideoCapture`/`VideoWriter`.

---

### C-04 | `pipeline_runner.py` -- Monkey-patch globale di `tqdm` non thread-safe
**File**: `person_anonymizer/web/pipeline_runner.py`, linee 171-253

`TqdmCapture.install()` sostituisce `tqdm.tqdm` a livello di modulo globale. Se due richieste web arrivano in rapida successione (prima che il lock in `start()` fermi la seconda), il monkey-patch della prima pipeline puo essere sovrascritto dalla seconda, corrompendo gli eventi SSE.

Inoltre, `pa.tqdm = PatchedTqdm` modifica un attributo del modulo importato, il che e fragile e puo avere side effect su altri import.

**Fix**: Usare dependency injection per il progress callback invece di monkey-patching globale. Passare un callable `on_progress` alla pipeline.

---

### C-05 | `pipeline_runner.py` -- `StdoutCapture` ridirige `sys.stdout` globalmente
**File**: `person_anonymizer/web/pipeline_runner.py`, linee 255-309

Ridirezione globale di `sys.stdout` in un thread. Se Flask logga qualcosa da un altro thread mentre la pipeline gira, quei messaggi finiscono nel capture SSE, potenzialmente esponendo dati interni al client web. La regex `_PATH_RE` mitiga solo i path, non altri dati sensibili.

**Fix**: Usare `logging` con handler dedicato invece di catturare stdout. Eliminare tutti i `print()` dalla pipeline in favore di logging strutturato.

---

### C-06 | Nessun `__init__.py` nella directory `person_anonymizer/`
**File**: assente in `person_anonymizer/`

La directory `person_anonymizer` non ha un `__init__.py`, il che significa che non e un package Python valido. Funziona solo perche `conftest.py` e `app.py` aggiungono manualmente il path a `sys.path`. Questo e fragile e impedisce import standard come `from person_anonymizer.config import PipelineConfig`.

**Fix**: Aggiungere `person_anonymizer/__init__.py` e convertire tutti gli import in import assoluti dal package.

---

### C-07 | Import relativi impliciti fragili
**File**: Tutti i moduli in `person_anonymizer/`

Tutti gli import tra moduli usano import "bare" (`from config import PipelineConfig`, `from detection import apply_nms`, etc.) che funzionano solo perche la directory e nel `sys.path`. Questi non sono import relativi validi (manca il `.`) ne import assoluti dal package.

```python
# anonymization.py:12 — import implicito
from config import PipelineConfig
# Dovrebbe essere:
from person_anonymizer.config import PipelineConfig
# oppure:
from .config import PipelineConfig
```

**Fix**: Aggiungere `__init__.py` e usare import relativi espliciti (`.config`, `.detection`, etc.) o import assoluti dal package.

---

## MAJOR (bug potenziali, design issues, manutenibilita)

### M-01 | `PipelineConfig` non valida i propri parametri
**File**: `person_anonymizer/config.py`

`PipelineConfig` e una dataclass senza alcuna validazione. Si puo creare `PipelineConfig(detection_confidence=999, smoothing_alpha=-5)` senza errori. La validazione esiste solo nel web layer (`pipeline_runner.py`), ma il CLI non la usa.

**Fix**: Aggiungere `__post_init__` con validazione dei range. Centralizzare la logica di validazione che oggi e duplicata in `_CONFIG_VALIDATORS`.

---

### M-02 | `PipelineConfig` usa `np.ndarray` come default per un campo `dataclass`
**File**: `person_anonymizer/config.py`, linee 41-42

```python
camera_matrix: np.ndarray | None = None
dist_coefficients: np.ndarray | None = None
```

Sebbene `None` come default sia sicuro, il type hint `np.ndarray | None` rende la dataclass non serializzabile con `dataclasses.asdict()` se i valori vengono settati (numpy array non e JSON-serializable). Inoltre, `config_defaults` in `app.py` chiama `asdict(cfg)` e si romperebbe se questi campi avessero valori non-None.

**Fix**: Usare `field(default=None)` esplicitamente e documentare che la serializzazione richiede conversione manuale.

---

### M-03 | Type hints mancanti su `list`, `tuple` nei campi di `PipelineConfig`
**File**: `person_anonymizer/config.py`, linee 56-57, 61-62, 89-93

```python
inference_scales: list = field(...)  # Manca list[float]
tta_augmentations: list = field(...)  # Manca list[str]
quality_clahe_grid: tuple = (8, 8)  # Manca tuple[int, int]
review_auto_color: tuple = (0, 255, 0)  # Manca tuple[int, int, int]
```

Type hint generici `list` e `tuple` non danno alcuna informazione sul contenuto. Con `from __future__ import annotations` attivo, si possono usare i generici built-in.

**Fix**: `inference_scales: list[float]`, `quality_clahe_grid: tuple[int, int]`, etc.

---

### M-04 | `run_pipeline` usa `args` come bag generico con attributi privati
**File**: `person_anonymizer/person_anonymizer.py`, linee 515-516, 530-532, 676-682

La funzione `run_pipeline(args, config)` accede a `args.input`, `args.mode`, `args.method` (argparse) ma anche a `args._review_state`, `args._sse_manager`, `args._job_id`, `args._stop_event` (attributi privati iniettati dal web runner). Questo accoppiamento implicito tra CLI e web e fragile e non documentato.

**Fix**: Definire una dataclass `PipelineInput` con tutti i parametri espliciti, oppure separare i parametri web in un argomento dedicato `web_context: WebContext | None = None`.

---

### M-05 | `_process_single_frame` restituisce una tupla di 8 elementi
**File**: `person_anonymizer/person_anonymizer.py`, linee 193-292

La funzione restituisce `(frame_polygons, frame_intensities, tracked, sw_hits, ms_hits, active_ids, new_prev_interp, motion_count)`. Otto valori posizionali sono impossibili da ricordare e facilissimi da confondere.

**Fix**: Creare una dataclass `FrameResult` con campi nominati.

---

### M-06 | Duplicazione della logica di intensita adattiva
**File**: `person_anonymizer/person_anonymizer.py`, linee 250-256, 269-275, 484-491
**File**: `person_anonymizer/postprocessing.py`, linee 366-372
**File**: `person_anonymizer/person_anonymizer.py`, linee 137-148

Il pattern `if config.enable_adaptive_intensity: compute_adaptive_intensity(...) else config.anonymization_intensity` e ripetuto 5 volte in 3 file. Ogni ripetizione ha lo stesso boilerplate.

**Fix**: Estrarre una funzione `resolve_intensity(config, box_height) -> int` che incapsula la scelta.

---

### M-07 | `rendering.py` importa `logging` dentro il corpo di una funzione
**File**: `person_anonymizer/rendering.py`, linee 134-136

```python
if corrupted_frames > 0:
    import logging  # import condizionale dentro funzione
    logging.getLogger(__name__).warning(...)
```

Import lazy dentro una funzione viola la convenzione Python (import all'inizio del file) e rende il logger non configurabile in anticipo.

**Fix**: Spostare `import logging` all'inizio del file e creare `_log = logging.getLogger(__name__)` a livello di modulo, come fatto correttamente in `tracking.py` e `postprocessing.py`.

---

### M-08 | `ManualReviewer.__init__` accetta un `dict` per config, non `PipelineConfig`
**File**: `person_anonymizer/manual_reviewer.py`, linee 37-41

```python
self.auto_color = config.get("auto_color", (0, 255, 0))
```

Il costruttore usa `config.get()`, quindi accetta un `dict`. Ma il resto del codebase usa `PipelineConfig` (dataclass). Questo crea un'incoerenza: `run_pipeline` deve costruire manualmente un dict di review prima di passarlo al reviewer (linee 556-562 di `person_anonymizer.py`).

**Fix**: `ManualReviewer` dovrebbe accettare `PipelineConfig` direttamente e leggere i campi come attributi.

---

### M-09 | Nessuna pulizia dei file di upload nel web
**File**: `person_anonymizer/web/app.py`

I file caricati in `UPLOAD_DIR/<job_id>/` non vengono mai eliminati. Se il server gira per settimane, lo spazio disco si esaurisce progressivamente con video di sorveglianza multi-GB.

**Fix**: Implementare un job di cleanup periodico (es. cron o thread background) che elimina le directory di upload piu vecchie di N ore.

---

### M-10 | `app.py` -- `SECRET_KEY` mancante per Flask
**File**: `person_anonymizer/web/app.py`

Flask viene creato senza `SECRET_KEY`. Se in futuro si aggiungono sessioni, flash messages o CSRF protection, tutto sara insicuro con la chiave di default. Best practice: settare sempre una `SECRET_KEY` anche se non si usano sessioni.

**Fix**: `app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))` e documentarlo in `.env.example`.

---

### M-11 | `app.py` -- `debug=False` hardcoded, nessuna configurazione via env
**File**: `person_anonymizer/web/app.py`, linea 464

```python
app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
```

Host, porta e debug sono hardcoded. Non c'e supporto per variabili d'ambiente. In sviluppo, bisogna modificare il codice per cambiare porta o abilitare il debug.

**Fix**: Leggere da `os.environ` con fallback ai default.

---

### M-12 | `app.py` -- `MAX_CONTENT_LENGTH` di 10 GB e eccessivo
**File**: `person_anonymizer/web/app.py`, linea 26

```python
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB
```

10 GB come limite di upload e molto alto. Un video di sorveglianza di 1 ora in 1080p e tipicamente 2-4 GB. 10 GB facilita DoS via upload di file enormi.

**Fix**: Ridurre a 4 GB o renderlo configurabile.

---

### M-13 | `review_state.py` -- `get_frame_jpeg` usa `cap.set(CAP_PROP_POS_FRAMES)` ripetutamente
**File**: `person_anonymizer/web/review_state.py`, linee 121-164

Ogni richiesta di frame dal browser fa `cap.set(CAP_PROP_POS_FRAMES, frame_idx)` + `cap.read()`. Per video con codec che non supportano il seeking efficiente (es. alcuni H.264), questo puo essere estremamente lento (O(n) per frame, dove n e la distanza dal frame corrente).

**Fix**: Documentare la limitazione, e considerare una cache LRU di frame decodificati per i video lunghi.

---

### M-14 | Manca `.env.example`
**File**: assente

Il progetto non ha un `.env.example` come richiesto dalle convenzioni. Il `.gitignore` menziona `.env` ma non c'e un esempio di configurazione.

**Fix**: Creare `.env.example` con `FLASK_SECRET_KEY=`, `FLASK_HOST=127.0.0.1`, `FLASK_PORT=5000`, etc.

---

## MINOR (code smell, stile, manutenibilita)

### m-01 | `VERSION` duplicata tra `config.py` e `pyproject.toml`
**File**: `person_anonymizer/config.py`, linea 14 (`VERSION = "7.1"`)
**File**: `pyproject.toml`, linea 3 (`version = "7.1"`)

La versione e definita in due posti. Se si aggiorna solo uno, restano disallineate.

**Fix**: Leggere la versione da `pyproject.toml` con `importlib.metadata.version("person-anonymizer")` oppure usare un singolo `__version__` in `__init__.py`.

---

### m-02 | `conftest.py` e `app.py` manipolano `sys.path` manualmente
**File**: `tests/conftest.py`, linea 5
**File**: `person_anonymizer/web/app.py`, linee 18-19

```python
# conftest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "person_anonymizer"))

# app.py
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
```

La manipolazione manuale di `sys.path` e fragile. Con `pyproject.toml` che gia dichiara `pythonpath = ["person_anonymizer"]`, il `conftest.py` potrebbe non servire.

**Fix**: Installare il package in modalita editable (`pip install -e .`) o usare import relativi coerenti.

---

### m-03 | `box_to_polygon` -- parametri di fallback hardcoded
**File**: `person_anonymizer/anonymization.py`, linee 148-149

```python
edge_threshold = config.edge_threshold if config is not None else 0.05
edge_padding_multiplier = config.edge_padding_multiplier if config is not None else 2.5
```

Valori di fallback duplicati (sono gli stessi di `PipelineConfig`). Se i default cambiano in `config.py`, questi restano disallineati.

**Fix**: Rendere `config` obbligatorio (rimuovere `None` come default) oppure usare `PipelineConfig()` come default.

---

### m-04 | `obscure_polygon` modifica `frame` in-place E lo restituisce
**File**: `person_anonymizer/anonymization.py`, linee 40-70

La funzione modifica il frame in-place (`frame[y:y+h, x:x+w] = ...`) e poi restituisce `frame`. Questo pattern e confuso: il chiamante non sa se il ritorno e una copia o lo stesso oggetto. Il valore di ritorno e ridondante.

**Fix**: O restituire solo `None` (modifica in-place esplicita) o lavorare su una copia e restituirla. Non entrambi.

---

### m-05 | `draw_debug_polygons` copia il frame 2 volte
**File**: `person_anonymizer/anonymization.py`, linee 99-100

```python
debug_frame = frame.copy()
overlay = debug_frame.copy()
```

Due copie complete del frame (potenzialmente 4K) quando ne basterebbe una. `overlay` viene poi blended su `debug_frame`.

**Fix**: `overlay = frame.copy()` e `debug_frame = frame` (o viceversa).

---

### m-06 | `run_full_detection` -- parametro `frame_w`, `frame_h` ridondante
**File**: `person_anonymizer/detection.py`, linee 155-203

La funzione riceve `frame_w` e `frame_h` come parametri separati, ma il frame stesso (`frame.shape`) contiene gia queste informazioni. Lo stesso pattern si ripete in molte funzioni.

**Fix**: Estrarre `frame_w, frame_h = frame.shape[1], frame.shape[0]` dentro la funzione.

---

### m-07 | `MotionDetector.get_motion_regions` -- padding non clampato ai bordi del frame
**File**: `person_anonymizer/preprocessing.py`, linee 102-109

```python
regions.append((
    max(0, x - self.padding),
    max(0, y - self.padding),
    x + w + self.padding,     # puo superare frame_w
    y + h + self.padding,     # puo superare frame_h
))
```

Il padding sinistro/superiore e clampato a 0, ma quello destro/inferiore puo superare le dimensioni del frame.

**Fix**: Passare `frame_w`, `frame_h` e clampare: `min(x + w + self.padding, frame_w)`.

---

### m-08 | `encode_without_audio` -- errore ffmpeg silenzioso
**File**: `person_anonymizer/postprocessing.py`, linee 74-86

```python
except ffmpeg.Error:
    shutil.copy(video_no_audio, output_path)
```

Se ffmpeg fallisce, il video AVI grezzo viene copiato silenziosamente come output. L'utente non viene avvisato che il video non e stato codificato in H.264.

**Fix**: Aggiungere `_log.warning()` come fatto in `encode_with_audio`.

---

### m-09 | `run_pipeline` -- troppi parametri locali passati tra fasi
**File**: `person_anonymizer/person_anonymizer.py`, linee 676-932

La funzione `run_pipeline` estrae variabili locali (`fps`, `frame_w`, `frame_h`, `undist_map1`, `undist_map2`, `fisheye_enabled`, etc.) e le passa a 5+ sotto-funzioni. Lo stesso set di ~12 variabili viene passato ripetutamente.

**Fix**: Raggruppare in una dataclass `PipelineContext` che contiene tutti i dati condivisi tra le fasi.

---

### m-10 | `_run_detection_loop` -- variabili accumulate in 7 liste/set locali
**File**: `person_anonymizer/person_anonymizer.py`, linea 306

```python
unique_ids, total_instances, frames_zero_det, all_confs, corrupted = set(), 0, 0, [], []
```

Cinque variabili di accumulo in una singola riga, piu `annotations` e `report_data`. Difficile da leggere e mantenere.

**Fix**: Creare una dataclass `DetectionStats` per le statistiche accumulate.

---

### m-11 | `camera_calibration.py` -- usa `print()` invece di `logging`
**File**: `person_anonymizer/camera_calibration.py`

Il modulo usa `print()` per tutti i messaggi. Incoerente con il resto del codebase che usa `logging.getLogger()`.

**Fix**: Usare logging, o almeno un logger dedicato.

---

### m-12 | `pipeline_runner.py` -- `_ALLOWED_FIELDS` e `_CONFIG_VALIDATORS` non sincronizzati
**File**: `person_anonymizer/web/pipeline_runner.py`

`_ALLOWED_FIELDS` ha 35 campi, `_CONFIG_VALIDATORS` ne ha ~30. Se si aggiunge un campo a `PipelineConfig`, bisogna ricordarsi di aggiornare entrambi. Non c'e un check automatico.

**Fix**: Generare `_ALLOWED_FIELDS` da `_CONFIG_VALIDATORS.keys() | _BOOL_FIELDS` automaticamente, oppure aggiungere un test che verifica la sincronizzazione.

---

### m-13 | `review_state.py` -- `_cap` mai rilasciato se `setup()` viene chiamato due volte senza `wait_for_completion()`
**File**: `person_anonymizer/web/review_state.py`, linee 81-90

Se il pipeline thread chiama `setup()` due volte (es. per un bug o retry), il vecchio `_cap` viene rilasciato (linea 82), ma se `setup()` fallisce dopo il release e prima del nuovo `VideoCapture`, lo stato resta inconsistente. Edge case, ma possibile.

---

### m-14 | `tests/` -- nessun test per `preprocessing.py`
**File**: `tests/`

Non esiste `test_preprocessing.py`. Le funzioni `enhance_frame`, `should_interpolate`, `interpolate_frames`, `MotionDetector` e `build_undistortion_maps` non hanno test. `should_interpolate` e `interpolate_frames` sono funzioni pure facilmente testabili.

**Fix**: Aggiungere `test_preprocessing.py` almeno per le funzioni pure.

---

### m-15 | `tests/` -- nessun test per `camera_calibration.py`
**File**: `tests/`

Nessun test per il modulo di calibrazione. Le funzioni `find_chessboard_corners` e `calibrate_camera` non sono testate.

---

### m-16 | `tests/` -- nessun test per `manual_reviewer.py`
**File**: `tests/`

Nessun test per `ManualReviewer`. La logica di coordinate (`_display_to_original`, `_original_to_display`) e le operazioni su annotazioni (`_delete_polygon_at`) sono testabili senza GUI.

---

### m-17 | `tests/` -- nessun test per `pipeline_runner.py` (non via web)
**File**: `tests/`

`TqdmCapture`, `StdoutCapture`, `_build_config`, `validate_config_params` sono testati solo indirettamente. Mancano test unitari per `PipelineRunner.start/stop/get_status`.

---

## NITPICK (stile, convenzioni, micro-ottimizzazioni)

### N-01 | Commenti separatori ASCII art
**File**: Tutti i moduli

```python
# ============================================================
# INTENSITA ADATTIVA
# ============================================================
```

Questo pattern di separazione visuale e presente in ogni file. In un codebase ben strutturato con file corti (<300 righe), questi separatori sono rumore visivo. Le funzioni e le classi si spiegano da sole con le docstring.

**Fix**: Rimuovere i separatori e affidarsi alla struttura del codice.

---

### N-02 | `VERSION = "7.1"` -- non segue SemVer
**File**: `person_anonymizer/config.py`, linea 14

La versione "7.1" non segue Semantic Versioning (MAJOR.MINOR.PATCH). Non c'e modo di distinguere tra bugfix e feature release.

**Fix**: Adottare SemVer: `"7.1.0"`.

---

### N-03 | `PipelineConfig` -- commenti in italiano misti a codice in inglese
**File**: `person_anonymizer/config.py`

I commenti di sezione sono in italiano (`# --- Modalita operativa ---`, `# --- Oscuramento ---`) ma i nomi dei campi e le docstring sono in inglese. Scegliere una lingua.

---

### N-04 | `obscure_polygon` -- `method` come stringa magica
**File**: `person_anonymizer/anonymization.py`, linea 55

```python
if method == "pixelation":
    ...
else:  # blur
```

Il metodo e una stringa. Se qualcuno passa `"PIXELATION"` o `"Blur"`, entra nel ramo sbagliato silenziosamente.

**Fix**: Usare un `Enum` per `AnonymizationMethod`.

---

### N-05 | `f-string` senza espressione in `camera_calibration.py`
**File**: `person_anonymizer/camera_calibration.py`, linee 109-113, 118, 130-135

```python
print(f"\nCalibrazione camera")  # f-string senza {}: inutile
print(f"  Immagini trovate: {len(image_paths)}")  # questa e corretta
```

Alcune f-string non contengono espressioni (es. linea 109) e sono equivalenti a stringhe normali.

---

### N-06 | Concatenazione implicita di stringhe per errore
**File**: `person_anonymizer/camera_calibration.py`, linea 118

```python
print(f"\nErrore: servono almeno 3 immagini valide, " f"trovate solo {len(obj_points)}.")
```

Due f-string concatenate implicitamente. Meglio una sola f-string.

---

### N-07 | `help_line1` in `manual_reviewer.py` usa concatenazione implicita
**File**: `person_anonymizer/manual_reviewer.py`, linee 250-254

```python
help_line1 = "  <-/-> Naviga  |  Spazio Avanti  |  " "Invio Chiudi poligono  |  Q Esci"
```

Concatenazione implicita di stringhe letterali su una riga. Poco leggibile.

---

### N-08 | `_run_manual_review` -- potenziale `ZeroDivisionError` se `total_frames == 0`
**File**: `person_anonymizer/person_anonymizer.py`, linee 547-548, 571-572

```python
f"  Frame modificati:      {review_stats['frames_modified']}  ({review_stats['frames_modified'] / total_frames * 100:.1f}%)"
```

Se `total_frames` e 0 (video vuoto, gia validato altrove ma non qui), si ha `ZeroDivisionError`.

**Fix**: Aggiungere guard `if total_frames > 0` o usare il pattern gia presente altrove.

---

### N-09 | `test_detection.py` -- import condizionale di `apply_nms`
**File**: `tests/test_detection.py`, linee 10-23

```python
try:
    import cv2 as _cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

if CV2_AVAILABLE:
    from detection import apply_nms
```

cv2 e nelle dipendenze del progetto ed e sempre installato nel venv. Il check e difensivo ma inutile in pratica.

---

### N-10 | `TemporalSmoother.__init__` -- `alpha` non validato
**File**: `person_anonymizer/tracking.py`, linea 119

Se `alpha > 1.0` o `alpha < 0.0`, l'EMA produce risultati senza senso (coordinate che divergono). Nessuna validazione.

**Fix**: `assert 0.0 < alpha <= 1.0` o `ValueError`.

---

### N-11 | `get_window_patches` -- nessun guard per `grid == 0`
**File**: `person_anonymizer/detection.py`, linee 20-34

Se `grid == 0`, si ha `ZeroDivisionError` su `int(frame_w / grid ...)`. La validazione esiste nel web layer ma non nella funzione stessa.

**Fix**: Aggiungere `if grid <= 0: raise ValueError(...)`.

---

### N-12 | `tests/__init__.py` e vuoto ma presente, `person_anonymizer/__init__.py` e assente
**File**: `tests/__init__.py` (presente, vuoto)
**File**: `person_anonymizer/__init__.py` (assente)

Incoerenza: i test sono un package, ma il codice sorgente no.

---

### N-13 | `FrameProcessors` usa `object` come type hint per 4 campi
**File**: `person_anonymizer/person_anonymizer.py`, linee 109-113

```python
clahe_obj: object
motion_detector: object  # MotionDetector | None
tracker: object  # BYTETracker | None
smoother: object  # TemporalSmoother | None
```

I commenti suggeriscono i tipi corretti, ma il type hint e `object`. Questo annulla ogni beneficio di type checking.

**Fix**: Usare `MotionDetector | None`, `TemporalSmoother | None`, etc. come type hint.

---

### N-14 | `app.py` -- `UPLOAD_DIR` e `OUTPUT_DIR` relativi al file, non configurabili
**File**: `person_anonymizer/web/app.py`, linee 31-32

```python
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
```

I percorsi sono hardcoded relativi al file sorgente. In deployment, si vorrebbe usare percorsi configurabili (es. `/tmp/anonymizer/uploads`).

**Fix**: Leggere da variabili d'ambiente con fallback al path corrente.

---

### N-15 | `app.py` -- `app.run()` non protetto da `if __name__`
**File**: `person_anonymizer/web/app.py`, linee 461-464

```python
# ---------- Main ----------

if __name__ == "__main__":
```

OK, questo e corretto. Pero il modulo `app` viene importato anche dai test, il che esegue tutto il codice a livello di modulo (creazione `UPLOAD_DIR`, `OUTPUT_DIR`, etc.). Con i test, le directory vengono create nel path del sorgente.

---

### N-16 | `postprocessing.py` -- `_merge_overlapping_rects` ha complessita O(n^2)
**File**: `person_anonymizer/postprocessing.py`, linee 300-315

Il commento dice "accettabile per n < 50". Corretto per sorveglianza tipica, ma non c'e un guard che avverta se n diventa grande.

**Fix**: Aggiungere un `_log.warning()` se `n > 100`.

---

### N-17 | `compute_review_stats` non considera modifiche di posizione dei poligoni
**File**: `person_anonymizer/rendering.py`, linee 147-192

La funzione conta solo il *numero* di poligoni per frame, non la loro posizione. Se un utente sposta un poligono (rimuove e aggiunge in posizione diversa), `frames_modified` resta 0 perche il count non cambia.

---

---

## Riepilogo Quantitativo

| Severita | Conteggio |
|----------|-----------|
| CRITICAL | 7 |
| MAJOR | 14 |
| MINOR | 17 |
| NITPICK | 17 |
| **Totale** | **55** |

## Top 5 Azioni Prioritarie

1. **Spezzare `person_anonymizer.py`** (995 righe) in 4-5 moduli piu piccoli
2. **Aggiungere `__init__.py`** e convertire tutti gli import in relativi espliciti
3. **Wrappare `VideoCapture`/`VideoWriter` in context manager** per prevenire resource leak
4. **Eliminare il monkey-patching di `tqdm`/`stdout`** in favore di dependency injection
5. **Aggiungere validazione in `PipelineConfig.__post_init__`** per centralizzare la validazione dei parametri
