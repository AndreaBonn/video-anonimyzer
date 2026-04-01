# Code Roast Report — Person Anonymizer

## Panoramica
- **Linguaggi rilevati**: Python 3.10+ (typing moderna: `str | None`, `tuple[bool, str]`)
- **File analizzati**: 17 (10 sorgente Python, 7 test Python)
- **Problemi totali**: 19 (CRITICAL 1 · MAJOR 6 · MINOR 7 · NITPICK 5)
- **Contesto rilevato**: tool CLI + web Flask, nessun linter configurato, pytest presente, nessun CI/CD, nessun Docker, nessun pyproject.toml/setup.py, .gitignore completo e corretto
- **Giudizio complessivo**: Codebase ben strutturata con separazione dei moduli chiara e test significativi, ma con un bug di resource leak critico nella review state, concorrenza gestita in modo ingenuo, e una pipeline monolitica che usa `sys.exit` come meccanismo di errore rendendo il codice dal web non testabile.

---

## CRITICAL (1 problema)

### CONCORRENZA — `cv2.VideoCapture` condiviso tra thread senza isolamento

**File**: `person_anonymizer/web/review_state.py` (righe 133–156)
**Problema**: `get_frame_jpeg` acquisisce `self._lock`, legge `self._cap` e poi lo usa **fuori dal lock** (riga 144–155). L'operazione `cap.set()` + `cap.read()` non è atomica: se Flask riceve due richieste `/api/review/frame/<idx>` concorrenti, entrambi i thread vedono lo stesso `self._cap`, lo seek e la read si interleave, e il frame restituito a uno dei due client è il frame richiesto dall'altro.

```python
# il cap viene letto dentro il lock...
with self._lock:
    if self._cap is None or not self._cap.isOpened():
        return None, 1.0
    self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self._cap.read()
    fisheye = self._fisheye_enabled   # ← qui si esce dal lock
    ...
# ...ma cap.set/cap.read avvengono già fuori dal lock
```

**Perché è grave**: La review manuale è il momento in cui un operatore corregge i poligoni di anonimizzazione. Un frame sbagliato mostrato al reviewer significa che l'operatore modifica annotazioni sul frame sbagliato — l'output finale contiene persone non anonimizzate. Il bug è deterministicamente riproducibile aprendo due tab del browser sulla stessa review.

**Come fixare**: Spostare l'intero blocco `cap.set/cap.read` dentro il lock, oppure tenere il cap fuori dallo stato condiviso e aprirlo per ogni richiesta (approccio più semplice e più corretto per un uso poco frequente):

```python
with self._lock:
    if self._cap is None or not self._cap.isOpened():
        return None, 1.0
    self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self._cap.read()
    if not ret:
        return None, 1.0
    fisheye = self._fisheye_enabled
    map1 = self._undist_map1
    map2 = self._undist_map2
# Il resto (resize, encode) fuori dal lock va bene perché opera su una copia locale
```

---

## MAJOR (6 problemi)

### ARCHITETTURA — `sys.exit()` come meccanismo di errore nella pipeline CLI

**File**: `person_anonymizer/person_anonymizer.py` (righe 66, 592, 595, 599, 625, 633)
**Problema**: `run_pipeline` chiama `sys.exit(1)` su ogni condizione di errore (file non trovato, formato non supportato, video non apribile). Quando la pipeline è invocata dal web via `PipelineRunner._run`, questo viene catturato dal `except SystemExit` — ma il messaggio all'utente è generico ("Pipeline terminata con codice 1"), e non c'è modo di testare i percorsi di errore senza avviare un processo reale.

**Perché è grave**: Rende `run_pipeline` una funzione non testabile per i suoi percorsi critici (file mancante, formato non supportato). I test esistenti non coprono questi path. Viola il principio di separazione tra logica di business e gestione dell'uscita del processo: `sys.exit` appartiene solo a `main()`.

**Come fixare**: Sollevare eccezioni specifiche (`FileNotFoundError`, `ValueError`) invece di `sys.exit`. Confinare `sys.exit` in `main()` che cattura le eccezioni e le traduce in codice di uscita.

---

### ARCHITETTURA — Monkey-patching globale di `tqdm` in un thread condiviso

**File**: `person_anonymizer/web/pipeline_runner.py` (righe 179–250)
**Problema**: `TqdmCapture.install()` sostituisce `tqdm.tqdm` e `person_anonymizer.tqdm` globalmente nel processo. Se mai venissero eseguiti due job concorrentemente (oggi non accade perché `PipelineRunner` blocca il secondo), entrambi scriverebbero sullo stesso `tqdm` patchato con il `job_id` del secondo job. Il design è fragile per definizione: fa affidamento su un invariante di singolo job che è enforced da un lock separato, non dalla struttura del codice.

**Perché è grave**: Il monkey-patching di un modulo globale è un anti-pattern di testabilità e manutenibilità. Rende impossibile testare la cattura del progresso in isolamento senza effetti collaterali globali. Un futuro refactoring che aggiunga parallelismo romperà silenziosamente il progresso SSE.

**Come fixare**: Passare una callback `on_progress` opzionale alla pipeline, che `run_pipeline` chiama a ogni frame. La web invoca la pipeline con la callback; la CLI la lascia `None`. Eliminazione completa del monkey-patching.

---

### PERFORMANCE — `_merge_overlapping_rects` con complessità O(n²) per iterazione

**File**: `person_anonymizer/postprocessing.py` (righe 262–295)
**Problema**: L'algoritmo di merge usa un loop while `changed` che ri-scansione tutta la lista a ogni iterazione. Nel caso peggiore (n rettangoli a catena) esegue O(n²) confronti per passata × O(n) passate = O(n³). Su video con scene dense (molte persone, molti frame), `normalize_annotations` può richiamare questa funzione su centinaia di frame.

**Perché è grave**: Con 50 rettangoli per frame e 1000 frame il calcolo è trascurabile; con 200+ rettangoli per frame (scene molto dense) la funzione diventa il collo di bottiglia del comando `--normalize`. Il pattern del problema è già nella MEMORY come ricorrente.

**Come fixare**: Union-Find (Disjoint Set Union) riduce a O(n α(n)) ≈ O(n). In alternativa, ordinare i rettangoli per `x` e usare uno sweep line — riduce comunque a O(n log n).

---

### MANUTENIBILITÀ — `_save_outputs` con 22 parametri

**File**: `person_anonymizer/person_anonymizer.py` (righe 460–484)
**Problema**: `_save_outputs` ha 22 parametri posizionali. La firma occupa 25 righe solo per la dichiarazione, e ogni call site trasmette altrettanti argomenti in ordine rigido. Aggiungere o riordinare un parametro richiede modifiche in ogni punto di chiamata.

**Perché è grave**: Viola il limite di 4 parametri per funzione indicato dalle regole di qualità del progetto. Rende le chiamate opaque (non si capisce cosa è `enable_debug` vs `enable_report` senza contare le posizioni). Ogni nuova feature che aggiunge un file di output deve estendere ulteriormente la firma.

**Come fixare**: Introdurre un dataclass `OutputPaths` che raggruppi tutti i path e un `RenderContext` che raggruppi fps/frame_w/frame_h/etc. La firma si riduce a 4-5 parametri semanticamente chiari.

---

### BUG — `_run_refinement_loop` esegue il rendering anche se `review_json` è già fornito

**File**: `person_anonymizer/person_anonymizer.py` (righe 735–750)
**Problema**: Quando l'utente passa `--review file.json`, le annotazioni vengono caricate da JSON, la cap viene rilasciata (riga 727), e poi `_run_refinement_loop` viene chiamata incondizionatamente. Il loop di refinement apre `temp_video_path` che non esiste ancora (non è stato fatto nessun rendering precedente) e chiama `render_video` che sovrascrive i risultati. Il comportamento atteso con `--review` è saltare la detection e procedere alla revisione manuale, non rieseguire tutto il ciclo di refinement.

**Perché è grave**: Con `--review` + `--mode manual`, l'utente si aspetta di rivedere le annotazioni caricate. Invece il codice esegue un rendering intermedio inutile (lento, usa CPU/GPU) e potrebbe fallire se il file di output temp non è ancora scrivibile.

**Come fixare**: Aggiungere un guard esplicito prima di `_run_refinement_loop`:

```python
if not review_json:
    annotations, actual_refinement_passes, refinement_annotations_added = _run_refinement_loop(...)
else:
    actual_refinement_passes, refinement_annotations_added = 0, 0
```

---

### RESOURCE LEAK — `cv2.VideoCapture` non rilasciato in caso di eccezione in `setup()`

**File**: `person_anonymizer/web/review_state.py` (righe 82–86)
**Problema**: In `setup()`, se `cv2.VideoCapture(video_path)` riesce ma una delle operazioni successive lancia un'eccezione (improbabile ma possibile), il `VideoCapture` appena creato non viene mai rilasciato. Non c'è un blocco `try/finally` né un context manager.

**Perché è grave**: `cv2.VideoCapture` mantiene un file handle sul video originale. In un sistema con molti job consecutivi, i leak si accumulano. Su Linux ogni processo ha un limite di file descriptor aperti; su macchine con GPU, i video aperti competono per i buffer di decodifica.

**Come fixare**: Avvolgere il setup del cap in un try/except che garantisce il release in caso di errore, o usare un context manager custom per `VideoCapture`.

---

## MINOR (7 problemi)

### MANUTENIBILITÀ — `_field_map` in `_build_config` è una mappa identità inutile

**File**: `person_anonymizer/web/pipeline_runner.py` (righe 117–166)
**Problema**: `field_map` mappa ogni chiave a se stessa (es. `"operation_mode": "operation_mode"`). L'unica eccezione utile è la conversione `quality_clahe_grid` da lista a tupla. Il loop che la utilizza è equivalente a `kwargs = {k: v for k, v in web_config.items() if k in set(field_map.keys())}`.

**Come fixare**: Eliminare `field_map`, usare un set `_ALLOWED_FIELDS` per il whitelist, e gestire la conversione `quality_clahe_grid` esplicitamente dopo il loop.

---

### MANUTENIBILITÀ — `_build_config` non valida `adaptive_reference_height` dalla web

**File**: `person_anonymizer/web/pipeline_runner.py` (riga 34)
**Problema**: `adaptive_reference_height` è in `_CONFIG_VALIDATORS` (correttamente), ma non è presente in `field_map`. Se il frontend invia questo parametro, viene validato ma poi silenziosamente scartato — non finisce nel `kwargs` di `PipelineConfig`. Il valore default viene usato indipendentemente dall'input.

**Come fixare**: Aggiungere `"adaptive_reference_height": "adaptive_reference_height"` a `field_map` (o, dopo il fix del punto precedente, al set `_ALLOWED_FIELDS`).

---

### ERROR HANDLING — `encode_with_audio` inghiotte errori di ffmpeg senza logging

**File**: `person_anonymizer/postprocessing.py` (righe 31–66)
**Problema**: Il doppio `except ffmpeg.Error` degrada silenziosamente: prima tenta senza audio, poi fa una copia grezza dell'AVI intermedio. L'utente non riceve nessun avviso che l'output è un AVI non compresso invece di H.264. Non c'è logging nemmeno al livello WARNING.

**Come fixare**: Aggiungere `logging.warning(f"ffmpeg con audio fallito, tentativo senza audio: {e}")` e `logging.warning(f"ffmpeg completamente fallito, copia grezza AVI: {e}")`.

---

### MANUTENIBILITÀ — `render_video` non verifica che `out_writer` sia inizializzato correttamente

**File**: `person_anonymizer/rendering.py` (riga 69)
**Problema**: `cv2.VideoWriter` restituisce un oggetto anche se non riesce ad aprire il file di output (es. percorso non scrivibile, codec non disponibile). `out_writer.isOpened()` non viene mai verificato. Il loop scrive frame su un writer silenziosamente non funzionante, e l'utente scopre il problema solo alla fine quando il file di output è vuoto o corrotto.

**Come fixare**: Aggiungere subito dopo la costruzione:
```python
if not out_writer.isOpened():
    raise RuntimeError(f"Impossibile aprire VideoWriter per {output_path}")
```

---

### CONFIG — `camera_matrix: object = None` in `PipelineConfig` è tipizzato male

**File**: `person_anonymizer/config.py` (righe 38–39)
**Problema**: I campi `camera_matrix` e `dist_coefficients` sono annotati come `object`. Il tipo corretto è `np.ndarray | None`. L'annotazione `object` non fornisce nessun valore documentale o di type checking: qualunque cosa è `object` in Python.

**Come fixare**: `camera_matrix: "np.ndarray | None" = None` (o `Optional[np.ndarray]` con `from __future__ import annotations`).

---

### TESTING — `test_config.py` testa solo getter di dataclass, non contratti

**File**: `tests/test_config.py` (righe 1–172)
**Problema**: I 28 test in `TestPipelineConfigDefaults` e `TestPipelineConfigCustomValues` verificano esclusivamente che `PipelineConfig(x=val).x == val`. Questo non testa nessun contratto: se si sbagliasse il tipo di default o si introducesse una guardia nel `__post_init__`, questi test non lo catturerebbero. I test rischiano di essere tautologici nel senso che verificano il comportamento della `dataclass` di Python standard, non la logica del progetto.

**Come fixare**: Concentrare i test su invarianti osservabili: `inference_scales` non deve mai essere vuota per default, `anonymization_intensity` deve essere positivo, `quality_clahe_grid` deve essere una tupla di due interi. Aggiungere test per la serializzazione JSON (il path `config_defaults()` del web converte tuple in liste).

---

### MANUTENIBILITÀ — Duplicazione `SUPPORTED_EXTENSIONS` tra `config.py` e `web/app.py`

**File**: `person_anonymizer/config.py` (riga 13), `person_anonymizer/web/app.py` (riga 31)
**Problema**: La stessa costante `SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ...}` è definita in entrambi i file con contenuto identico. Se si aggiunge un formato (es. `.ts`), va aggiornato in due posti.

**Come fixare**: `web/app.py` deve importare `from config import SUPPORTED_EXTENSIONS` invece di ridichiarare il set.

---

## NITPICK (5 problemi)

### `manual_reviewer.py` non ha test e non è coperto dalla suite

Il modulo esegue UI OpenCV interattiva (`cv2.imshow`, `cv2.waitKey`) — correttamente esclusa dai test unitari. Non è un problema blocante ma va documentato esplicitamente (es. con un commento nel conftest) in modo che i futuri maintainer non si chiedano perché manca.

---

### `camera_calibration.py` usa `os.path` invece di `pathlib`

Il resto del codebase usa `pathlib.Path` in modo consistente. `camera_calibration.py` usa `os.path.isdir`, `os.path.join`, `os.path.basename`, `glob.glob` — stile legacy. Non causa bug ma è incoerente con il resto.

---

### `update_tracker` ricrea `_log = logging.getLogger(__name__)` a ogni chiamata

**File**: `person_anonymizer/tracking.py` (riga 86)
`logging.getLogger` è thread-safe e cachato internamente, quindi non è un bug di performance grave, ma è idiomaticamente sbagliato: il logger dovrebbe essere una costante a livello di modulo (`_LOG = logging.getLogger(__name__)` fuori dalla funzione).

---

### `StdoutCapture` non gestisce il buffer residuo alla chiusura

**File**: `person_anonymizer/web/pipeline_runner.py` (righe 253–300)
Quando la pipeline termina e `uninstall()` viene chiamato, `self._buffer` potrebbe contenere testo senza `\n` finale (es. l'ultima riga di progresso senza newline). Quel testo viene perso silenziosamente. In pratica, i messaggi importanti terminano sempre con `\n`, quindi l'impatto è nullo, ma `flush()` dovrebbe emettere il buffer residuo.

---

### `firebase-debug.log` committato nella directory sorgente

**File**: `person_anonymizer/firebase-debug.log`
Il file è presente nella directory sorgente. Il `.gitignore` lo esclude con `*.log`, ma il file è già tracciato da git (il `.gitignore` non rimuove file già in staging/committed). Bisogna rimuoverlo con `git rm --cached person_anonymizer/firebase-debug.log`. Inoltre la sua presenza suggerisce che Firebase è stato integrato o testato in questa directory — cosa non documentata in CLAUDE.md.

---

## Priorità di Refactoring Consigliate

1. **Fix il race condition su `get_frame_jpeg`** — È l'unico CRITICAL, direttamente in un path funzionale della review manuale. Un reviewer che usa due tab vede frame sbagliati e produce annotazioni errate. 30 minuti di fix.

2. **Sostituire `sys.exit` con eccezioni in `run_pipeline`** — Sblocca la testabilità dei percorsi di errore critici (file non trovato, formato non supportato) e rimuove la dipendenza dal `except SystemExit` in `PipelineRunner`. Refactoring di 1-2 ore con impatto sulla copertura dei test.

3. **Aggiungere guard per `review_json` prima di `_run_refinement_loop`** — Bug funzionale: con `--review file.json`, il refinement loop non dovrebbe girare. Fix di 5 righe.

4. **Aggiungere `adaptive_reference_height` a `field_map` in `_build_config`** — Il parametro è validato ma scartato silenziosamente. Bug subdolo: l'utente imposta il valore nel frontend e non ha effetto. Fix di 1 riga.

5. **Deduplicare `SUPPORTED_EXTENSIONS`** — `web/app.py` deve importare da `config.py`. Due file che divergono su questa lista produrrebbero un bug dove un formato accettato dall'upload viene poi rifiutato dalla pipeline CLI.

---

## Verdict finale

Il codebase mostra chiaramente l'investimento fatto sulla qualità: la decomposizione in moduli è corretta, la suite di test copre i path puri in modo solido, e la validazione degli input web è più seria di quanto si veda in molti progetti simili. Il gap principale non è lo stile ma la concorrenza: il componente più delicato del sistema — la review manuale, dove si decide cosa viene anonimizzato — ha un race condition che compromette la correttezza del prodotto finale.
