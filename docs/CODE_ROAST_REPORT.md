# Code Roast Report — Person Anonymizer v7.1

## Panoramica
- **Linguaggi rilevati**: Python (backend/pipeline), JavaScript (frontend vanilla), CSS, HTML
- **File analizzati**: 12 file sorgente (~2500 LOC stimati)
- **Problemi totali**: 18 (CRITICAL 3 · MAJOR 5 · MINOR 6 · NITPICK 4)
- **Contesto rilevato**: progetto personale, nessun linter/formatter, nessun CI/CD, nessun test, nessun Dockerfile
- **Giudizio complessivo**: *Un prototipo impressionante per funzionalità, ma architettonicamente è come un palazzo di 50 piani costruito senza ascensore e con un unico pilone portante.*

---

## CRITICAL (3 problemi)

### ARCHITETTURA — Il God File da 2000 righe
**File**: `person_anonymizer/person_anonymizer.py` (righe 1-1999)
**Problema**: Un singolo file da **2000 righe** che contiene: configurazione, preprocessing, motion detection, multi-scale inference, NMS, tracking, temporal smoothing, intensità adattiva, interpolazione sub-frame, oscuramento, rendering, encoding audio, post-render check, normalizzazione annotazioni, CLI parsing e la pipeline principale. Praticamente ha più responsabilità di un CEO.
**Perché è grave**: È impossibile testare, rifattorizzare o estendere qualsiasi singolo componente senza rischiare di rompere tutto il resto. Ogni modifica è una roulette russa. Un nuovo sviluppatore apre questo file, vede 2000 righe, e chiude il laptop.
**Come fixare**: Scomponi in moduli per dominio:
```
src/
├── config.py          # Costanti e dataclass di configurazione
├── detection/
│   ├── multiscale.py  # Inferenza multi-scala + TTA
│   ├── sliding_window.py
│   └── nms.py
├── tracking/
│   ├── bytetrack.py
│   └── smoother.py
├── anonymization/
│   ├── obscure.py     # Pixelation/blur
│   └── adaptive.py    # Intensità adattiva
├── rendering/
│   ├── video.py       # Rendering + debug
│   └── audio.py       # Encoding H.264 + audio
├── review/
│   └── manual.py
├── postprocessing/
│   ├── verify.py      # Post-render check
│   └── normalize.py
└── pipeline.py        # Orchestrazione (solo colla)
```

---

### ARCHITETTURA — 40+ variabili globali come "configurazione"
**File**: `person_anonymizer/person_anonymizer.py` (righe 30-113)
**Problema**: La configurazione è un campo minato di **42 variabili globali a livello modulo**, mutate a runtime via `setattr(pa, mod_key, val)` in `pipeline_runner.py:260`. Ecco, hai appena inventato il "global injection pattern". Congratulazioni, non esisteva perché nessuno lo voleva.
**Perché è grave**: In un contesto multi-thread (Flask con `threaded=True`), due job concorrenti sovrascrivono gli stessi globals a vicenda. Il fatto che `PipelineRunner` limiti a un job alla volta è l'unica cosa che impedisce al castello di crollare. Inoltre, rende impossibile testare componenti in isolamento perché tutti dipendono da stato globale.
**Come fixare**: Crea una `PipelineConfig` dataclass:
```python
@dataclass
class PipelineConfig:
    operation_mode: str = "manual"
    anonymization_method: str = "pixelation"
    anonymization_intensity: int = 10
    # ... etc
```
Passala come parametro alle funzioni. Zero globals, zero `setattr`, zero preghiere.

---

### ARCHITETTURA — `run_pipeline()`: la God Function da 700+ righe
**File**: `person_anonymizer/person_anonymizer.py` (righe 1216-1939)
**Problema**: Una singola funzione di **723 righe** che fa *letteralmente tutto*: validazione input, apertura video, stampa header, caricamento modello, detection loop, refinement loop, revisione manuale, rendering finale, encoding audio, scrittura CSV, scrittura JSON, pulizia temp files e stampa riepilogo. Se fosse una persona, farebbe il chirurgo, l'avvocato e l'idraulico contemporaneamente.
**Perché è grave**: Impossibile da testare, impossibile da debuggare, impossibile da modificare senza leggere tutto 3 volte. Violi il principio di single responsibility al punto che il principio chiede un ordine restrittivo.
**Come fixare**: Ogni `[FASE X/5]` diventa una funzione separata. La pipeline orchestratrice diventa:
```python
def run_pipeline(config: PipelineConfig, args: PipelineArgs):
    ctx = setup_pipeline(config, args)
    ctx.annotations = run_detection(ctx) if not args.review else load_from_json(args.review)
    ctx.annotations = run_refinement(ctx)
    ctx.annotations = run_review(ctx) if ctx.mode == "manual" else ctx.annotations
    render_final(ctx)
    postprocess(ctx)
```

---

## MAJOR (5 problemi)

### TESTING — Zero test. Nessuno. Nada. Zilch.
**File**: Assente (nessuna directory `tests/`)
**Problema**: Un progetto con multi-scale inference, NMS, tracking, temporal smoothing, interpolazione, post-render verification... e **zero test**. Neanche uno. Nemmeno un test che verifica che 2+2=4 per sentirsi in compagnia.
**Perché è grave**: Ogni modifica è un atto di fede. Non sai se funziona fino a quando non elabori un video di 5 minuti e guardi il risultato con i tuoi occhi umani. Algoritmi come NMS, IoU, EMA smoothing, ghost boxes sono perfetti candidati per unit test. Il fatto che funzionino è pura fortuna.
**Come fixare**: Inizia da dove il rischio è più alto:
1. `apply_nms()` — input/output deterministico
2. `compute_adaptive_intensity()` — funzione pura
3. `compute_iou_boxes()` — matematica verificabile
4. `_merge_overlapping_rects()` — algoritmo con edge case
5. `TemporalSmoother` — stato verificabile

---

### ARCHITETTURA — `os.chdir()` in un thread Flask
**File**: `person_anonymizer/web/pipeline_runner.py` (riga 246)
**Problema**: `os.chdir(pa_dir)` è un'operazione **globale di processo**, non di thread. In un server Flask con `threaded=True`, cambiare la working directory in un thread cambia la working directory per TUTTI i thread. Il fatto che funzioni è solo perché c'è un solo job alla volta. Ma Flask continua a servire richieste HTTP durante l'elaborazione.
**Perché è grave**: Se una qualsiasi richiesta Flask dipendesse dal cwd (e `send_file` con path relativi lo farebbe), otterresti risultati imprevedibili. È una bomba a orologeria.
**Come fixare**: Usa path assoluti ovunque e elimina `os.chdir()`. Il modello YOLO può essere caricato specificando il path assoluto: `YOLO(str(Path(pa.__file__).parent / YOLO_MODEL))`.

---

### MANUTENIBILITA' — `pipeline_runner.py` patcha globali e stdout come un malware
**File**: `person_anonymizer/web/pipeline_runner.py` (righe 62-144, 146-187, 248-260)
**Problema**: Per far comunicare la pipeline con il frontend, si monkey-patcha `tqdm`, si redirecta `sys.stdout`, e si mutano i globals del modulo con `setattr`. È ingegnoso, ma anche disgustoso. È come risolvere un problema di comunicazione mettendo un microfono nascosto nella stanza.
**Perché è grave**: Qualsiasi libreria che scrive su stdout durante l'elaborazione (e OpenCV ne ha parecchie) contribuirà al flusso SSE senza che tu lo voglia. Se tqdm cambia la sua API interna, tutto esplode. E il `setattr` su globals è già stato roastato sopra.
**Come fixare**: Usa un pattern di callback o un event bus:
```python
class PipelineCallbacks:
    def on_progress(self, current, total, desc): ...
    def on_phase(self, phase_num, label): ...
    def on_log(self, message): ...
```
Passalo alla pipeline, e che tqdm viva in pace.

---

### MANUTENIBILITA' — `manual_reviewer.py`: 310 righe, una sola funzione con 8 closure
**File**: `person_anonymizer/manual_reviewer.py` (righe 13-309)
**Problema**: `run_manual_review()` è una funzione di 297 righe che contiene 8 funzioni closure annidate (`get_frame`, `display_to_original`, `original_to_display`, `point_in_polygon`, `render_display`, `mouse_callback`) tutte che accedono a variabili `nonlocal`. È un universo autocontenuto. Uno Spaghetti Monster delle closure.
**Perché è grave**: Non puoi testare nessuna delle closure individualmente. Il flow control è sparpagliato tra key codes hardcodati (83, 81, 13, 26, 27) e flag booleani.
**Come fixare**: Crea una classe `ManualReviewer` con stato, e metodi separati per ogni azione. I key codes diventano un dizionario di mapping.

---

### ERROR HANDLING — Bare `except Exception` silenzioso nel tracker
**File**: `person_anonymizer/person_anonymizer.py` (righe 449-456, 463-466)
**Problema**: Il fallback del tracker ingoia l'eccezione e restituisce box senza tracking come se niente fosse. Il `continue` nell'inner loop pure.
```python
except Exception:
    # Fallback: restituisce i box senza tracking
    results = []
    ...
```
**Perché è grave**: Se il tracker fallisce silenziosamente, non lo saprai mai. Perderai il tracking su interi video senza alcun warning. In un tool GDPR-critical, il silenzio è il peggior errore possibile.
**Come fixare**: Logga l'errore con severity, e alza un warning nell'output. L'utente deve sapere che il tracking è degradato.

---

## MINOR (6 problemi)

### MANUTENIBILITA' — `print()` come unico sistema di logging
**File**: Trovato in 3 file: `person_anonymizer.py`, `camera_calibration.py`, `manual_reviewer.py`
**Problema**: Tutto il logging è fatto con `print()`. In un contesto CLI è accettabile, ma la web app patcha `sys.stdout` per catturarlo — il che dimostra che serviva un sistema di logging vero.
**Come fixare**: Usa il modulo `logging` di Python. Una riga di setup.

---

### SICUREZZA UI — `innerHTML` con dati utente nel frontend
**File**: `person_anonymizer/web/static/js/app.js` (righe 539-544)
**Problema**: I nomi dei file di output vengono inseriti via `innerHTML` senza sanitizzazione:
```javascript
item.innerHTML = `<span class="result-name">${f.name}</span>...`;
```
Un filename malevolo potrebbe contenere tag HTML.
**Come fixare**: Usa `textContent` per i dati utente, o crea gli elementi DOM con `createElement`.

---

### PERFORMANCE — VideoCapture aperto per ogni frame in review_state
**File**: `person_anonymizer/web/review_state.py` (righe 122-129)
**Problema**: `get_frame_jpeg()` apre e chiude un `cv2.VideoCapture` per ogni singola richiesta di frame. Seeking in un video MP4 è costoso, e la creazione dell'oggetto pure.
**Come fixare**: Mantieni un `VideoCapture` aperto per la durata della review, protetto dal lock.

---

### CONFIGURAZIONE — Dati nel README non allineati con il codice
**File**: `README.md` (righe 403-404)
**Problema**: Il README dice `DETECTION_CONFIDENCE = 0.35` e `NMS_IOU_THRESHOLD = 0.45`, ma il codice ha `0.20` e `0.55`. Il README mente all'utente.
**Come fixare**: Allinea i valori, o meglio ancora, genera la documentazione dal codice.

---

### MANUTENIBILITA' — Key codes hardcodati nell'interfaccia OpenCV
**File**: `person_anonymizer/manual_reviewer.py` (righe 259, 266)
**Problema**: `83`, `81`, `32`, `13`, `26`, `27` sono codici magici senza nome. Devi aprire la tabella ASCII per capire cosa fanno.
**Come fixare**: Definisci costanti con nomi parlanti: `KEY_RIGHT = 83`, `KEY_ENTER = 13`.

---

### CONFIGURAZIONE — `.venv` dentro `person_anonymizer/`
**File**: `person_anonymizer/.venv/`
**Problema**: C'è un virtual environment dentro la sottodirectory del package. Perché? C'è già `.venv` nella root del progetto. Due venv = confusione garantita.
**Come fixare**: Elimina `person_anonymizer/.venv/` e usa solo quello nella root.

---

## NITPICK (4 problemi)

### NAMING — Mix di italiano e inglese nei commenti e nelle variabili
Trovato in tutti i file Python. I commenti sono in italiano, le variabili in inglese, le docstring in italiano. "Errore: impossibile aprire il video per la revisione" convive con `enhanced = enhance_frame()`. Scegline uno e basta.

### STILE — `MODALITA'` con apostrofo invece di accento
Trovato in `person_anonymizer.py` (righe 29, 71, 87) e nel README. `MODALITA'` e `INTENSITA'` sono il "Comic Sans" dell'accentazione italiana. Usa `MODALITÀ` e `INTENSITÀ`.

### STRUTTURA — Manca `CLAUDE.md` nella root del progetto (presente solo per il sub-progetto)
Hai un `.claude/` ma non un `CLAUDE.md` nella root come previsto dalle tue stesse convenzioni.

### JS STYLE — `var` usato in `review-editor.js` anziché `let`/`const`
Tutto il file usa `var` come se fossimo nel 2014. Il resto del JS (`app.js`) usa `let`/`const`. Inconsistenza tra i due file.

---

## Priorità di Refactoring Consigliate

1. **Scomponi `person_anonymizer.py`** — È il problema #1. Ogni altro miglioramento è impossibile senza questo. Crea moduli, elimina globals, passa configurazione come oggetto.

2. **Aggiungi test sulle funzioni pure** — `apply_nms()`, `compute_iou_boxes()`, `compute_adaptive_intensity()`, `_merge_overlapping_rects()` sono tutte funzioni pure testabili in 2 minuti. Inizia da queste.

3. **Elimina il monkey-patching nella web app** — Sostituisci stdout capture + tqdm patch con un sistema di callback pulito. Elimina `os.chdir()` e `setattr` sui globals.

4. **Fixa le vulnerabilità di sicurezza** — Path traversal, XSS, SSRF sono tutti presenti (vedi Security Audit). Non servono in un tool locale, ma se questo repo va su GitHub, qualcuno lo deployerà in rete.

5. **Allinea documentazione e codice** — Il README con valori sbagliati è peggio di nessun README.

---

## Verdict finale

*Il tuo tool fa cose impressionanti: multi-scale inference, ByteTrack, temporal smoothing con ghost boxes, auto-refinement loop, interfaccia web con SSE in tempo reale. È evidente che sai cosa stai facendo a livello algoritmico. Ma architettonicamente, hai messo un motore di Formula 1 in una Fiat Panda senza cinture di sicurezza. Il file da 2000 righe con 42 globals mutati via setattr in un thread Flask è un crimine contro l'ingegneria del software — l'unica ragione per cui funziona è che solo tu lo usi e solo un job alla volta. Scomponi, testa, e sarà un progetto di cui andare davvero fiero.*
