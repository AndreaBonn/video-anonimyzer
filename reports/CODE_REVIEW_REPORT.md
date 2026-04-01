# Code Review Report — Person Anonymizer v7.1

**Data:** 2026-04-01
**Reviewer:** Senior Code Reviewer (automated, 15 anni di esperienza)
**File analizzati:** 15 file Python (10 sorgenti + 1 `__init__` + 4 test selezionati)
**Tool di analisi statica:** ruff configurato in `pyproject.toml` (E, F, W, I)
**Contesto:** Progetto in sviluppo attivo, CI non rilevata, test suite presente, codice modulare maturo

---

## Panoramica

| Categoria | Conteggio |
|-----------|-----------|
| Problemi critici (bug reali / resource leak) | 3 |
| Problemi alti (logica errata, edge case non gestiti) | 5 |
| Problemi medi (manutenibilità, type safety) | 7 |
| Problemi bassi / nitpick | 5 |
| **Totale** | **20** |

**Giudizio complessivo:** Codebase di buona qualità per un tool di processing video. Architettura modulare pulita, sicurezza Flask solida, test suite presente. I problemi critici riguardano resource leak su VideoCapture e un edge case silenzioso nel detection loop — non bloccano in scenari normali ma diventano problemi reali su video corrotti o su ambienti embedded.

---

## CRITICI (3 problemi)

### [CRIT-01] Resource leak su `VideoCapture` in `run_post_render_check` — `postprocessing.py:122-161`

**File:** `person_anonymizer/postprocessing.py` righe 122-161

**Problema:**
`cap.release()` è chiamata solo alla fine del while loop normale. Se l'elaborazione lancia un'eccezione non catturata all'interno del ciclo (es. numpy shape mismatch su un frame malformato), `cap` non viene mai rilasciata.

**Codice attuale:**
```python
cap = cv2.VideoCapture(anonymized_video_path)
# ... ciclo while ...
cap.release()
return alert_frames
```

**Come correggere:**
```python
cap = cv2.VideoCapture(anonymized_video_path)
try:
    # ... ciclo while ...
finally:
    cap.release()
return alert_frames
```

**Impatto:** Su sistemi con molti video in coda (uso web), i file descriptor si esauriscono. Su Linux, il default è 1024 fd per processo — con video grandi e encoding parallelo, si raggiunge rapidamente.

---

### [CRIT-02] Resource leak su `VideoCapture` e `VideoWriter` in `render_video` — `rendering.py:67-144`

**File:** `person_anonymizer/rendering.py` righe 67-144

**Problema:**
Il loop `while True` ha un check `stop_event.is_set()` che provoca un `break`. In questo caso, `cap.release()`, `out_writer.release()` e `debug_writer.release()` vengono chiamati dal codice post-loop — SOLO se l'eccezione non interrompe il flusso. Ma soprattutto: se `obscure_polygon` o `undistort_frame` lanciano un'eccezione, i writer non vengono mai rilasciati. L'attuale struttura non usa `try/finally`.

**Come correggere:**
Wrappare l'intero blocco `cap = cv2.VideoCapture(...)` / `out_writer = ...` in un `try/finally` che chiama `.release()` su tutti i writer aperti.

**Impatto:** Identico a CRIT-01. In aggiunta, i file video temporanei rimangono locked su Windows (rilevante se il tool viene portato in futuro).

---

### [CRIT-03] Race condition su `MotionDetector.prev_gray` — `preprocessing.py:70-110`

**File:** `person_anonymizer/preprocessing.py` righe 70-110

**Problema:**
`MotionDetector` è istanziato in `_init_frame_processors` e condiviso all'interno di `_run_detection_loop`. Se in futuro il loop di detection venisse parallelizzato per frame (es. con `ThreadPoolExecutor`), `self.prev_gray` verrebbe letto e scritto da più thread contemporaneamente. Attualmente il codice è single-threaded, quindi non è un bug attivo, ma la classe non documenta il vincolo di non-thread-safety e non ha alcuna protezione.

**Come correggere — opzione minima:**
Aggiungere nel docstring di `MotionDetector` e di `get_motion_regions`:
```python
# NON thread-safe: istanza dedicata per thread.
```

**Opzione robusta:** Aggiungere `threading.Lock` attorno all'aggiornamento di `self.prev_gray`.

**Impatto:** Non causa bug oggi; diventa un data race silenzioso se il loop viene reso asincrono. Dato che il resto dell'architettura usa threading (PipelineRunner), il rischio di un futuro refactoring errato è concreto.

---

## ALTI (5 problemi)

### [ALTA-01] `TemporalSmoother.get_ghost_boxes` modifica il dizionario durante l'iterazione — `tracking.py:146-161`

**File:** `person_anonymizer/tracking.py` righe 146-161

**Problema:**
Il codice usa `for tid in list(self.ghost_countdown):` (corretto), ma poi chiama `del self.ghost_countdown[tid]` e `del self.state[tid]` all'interno del loop. Questo è safe perché si itera su una copia (`list()`). Tuttavia, nel ramo `else:` viene eseguito `del self.state[tid]` solo se `tid in self.state`, ma non viene verificato che `tid` sia ancora in `self.state` quando si tenta il delete nel ramo `if countdown > 0`. In realtà, nel ramo `if` si accede a `self.state[tid]` senza guard: se un chiamante esterno avesse rimosso il `tid` da `self.state` nel frattempo (es. via `clear_stale`), si otterrebbe `KeyError`.

**Codice critico:**
```python
if countdown > 0 and tid in self.state:
    s = self.state[tid]  # <-- safe
    ...
    self.ghost_countdown[tid] -= 1
else:
    del self.ghost_countdown[tid]
    if tid in self.state:  # <-- guard presente, corretto
        del self.state[tid]
```

In realtà questo codice è corretto — il guard `and tid in self.state` protegge il ramo `if`. Il problema reale è che `clear_stale` e `get_ghost_boxes` possono essere chiamati in ordine arbitrario, e `clear_stale` aggiunge a `ghost_countdown` senza controllare se `tid` è già presente: se un track viene perso e poi "riappare" (tracked di nuovo) nel frame successivo, `smooth()` fa `self.ghost_countdown.pop(track_id, None)` che rimuove il countdown — comportamento corretto. Ma se `get_ghost_boxes` viene chiamato PRIMA di `clear_stale` nello stesso frame (inversione nell'ordine delle chiamate), il ghost del frame precedente viene emesso erroneamente.

**Dove verificare:** `person_anonymizer.py:262-263` — l'ordine è `clear_stale` poi `get_ghost_boxes`, corretto. Ma questo ordine non è documentato come vincolo né verificato da un test.

**Come correggere:** Aggiungere un commento esplicito che documenta il vincolo di ordine:
```python
# ATTENZIONE: clear_stale() deve essere chiamato PRIMA di get_ghost_boxes()
# per ogni frame. Invertire l'ordine produce ghost erronei.
proc.smoother.clear_stale(active_ids)
for gtid, gx1, gy1, gx2, gy2 in proc.smoother.get_ghost_boxes():
```

---

### [ALTA-02] `StdoutCapture` non è thread-safe — `web/pipeline_runner.py:255-309`

**File:** `person_anonymizer/web/pipeline_runner.py` righe 255-309

**Problema:**
`StdoutCapture` sostituisce `sys.stdout` globale. Se Flask riceve due richieste concorrenti che avviano pipeline (anche se `PipelineRunner` lo impedisce per `_thread.is_alive()`), il thread Flask principale continua a scrivere su `sys.stdout` mentre il thread pipeline lo ha sostituito. Ogni print() di Flask (debug mode, log werkzeug) finisce nella queue SSE del job in corso invece che sul terminale reale.

In pratica con `debug=False` e `threaded=True`, werkzeug può loggare su stdout durante l'elaborazione. Il risultato è che messaggi di werkzeug vengono inviati come eventi SSE al frontend.

**Come correggere:**
Usare un `logging.Handler` custom invece di monkey-patching `sys.stdout`. In alternativa, fare in modo che `StdoutCapture` inoltri solo le righe che matchano il formato atteso della pipeline (`[FASE`, progress updates), ignorando il resto.

---

### [ALTA-03] `validate_job_id` accetta `None` senza crash ma con comportamento inaspettato — `web/app.py:57-61`

**File:** `person_anonymizer/web/app.py` righe 57-61

**Problema:**
```python
def validate_job_id(job_id: str) -> bool:
    if not job_id or len(job_id) != 12:
        return False
```

Se `job_id` è `None`, `not job_id` è `True` e la funzione ritorna `False` correttamente. Tuttavia, la firma dichiara `job_id: str` ma il parametro può arrivare come `None` da `request.args.get("job_id")` (senza default). Il type hint è fuorviante. Nei test, `test_empty_job_id_returns_404` verifica che `/api/outputs/` dia 404 (route non matchata), ma non testa il caso in cui `job_id` sia `None` nel body JSON di `/api/stop`.

**Come correggere:**
```python
def validate_job_id(job_id: str | None) -> bool:
```

---

### [ALTA-04] `_run_detection_loop` non rilascia `cap` se il loop viene interrotto da `stop_event` — `person_anonymizer.py:295-383`

**File:** `person_anonymizer/person_anonymizer.py` righe 295-383

**Problema:**
Nel loop di detection, quando `stop_event.is_set()` è True, il codice esegue `break` e poi cade nel codice post-loop dove `cap.release()` viene chiamato a riga 370. Questo è corretto. Tuttavia, se `_process_single_frame` lancia un'eccezione non gestita (es. `cv2.error` su frame malformato), il codice salta `cap.release()`. Stesso pattern di CRIT-01 e CRIT-02.

Il problema è sistematico: tutti i `VideoCapture` del progetto dovrebbero usare `try/finally`.

---

### [ALTA-05] `obscure_polygon` modifica il frame in-place ma restituisce anche il frame — `anonymization.py:40-70`

**File:** `person_anonymizer/anonymization.py` righe 40-70

**Problema:**
La funzione modifica `frame` in-place tramite:
```python
frame[y : y + h, x : x + w] = np.where(...)
```
E poi `return frame`. Il contratto è ambiguo: il chiamante non sa se può fidarsi del riferimento originale o se deve usare il valore di ritorno. In `rendering.py:115`:
```python
render_frame = obscure_polygon(render_frame, poly, method, intensity)
```
Qui `render_frame` viene riassegnato al valore di ritorno — corretto. Ma in `rendering.py:121`:
```python
render_frame = obscure_polygon(render_frame, poly, method, intensity)
```
Identico. Il problema potrebbe emergere se qualcuno chiama la funzione senza assegnare il ritorno, aspettandosi che la modifica in-place basti — e in effetti la modifica in-place funziona, ma il frame originale `frame` (nel contesto del debug) è già stato modificato quando si chiama `draw_debug_polygons` a riga 126. Questo è il comportamento inteso, ma non documentato.

**Come correggere:**
Documentare esplicitamente nel docstring che la funzione modifica `frame` in-place E restituisce il frame modificato, oppure fare una copia interna e non modificare l'input.

---

## MEDI (7 problemi)

### [MED-01] Type hints mancanti su parametri di ritorno in molte funzioni pubbliche

**File:** `preprocessing.py`, `tracking.py`, `rendering.py`, `postprocessing.py`

Esempi:
- `build_undistortion_maps` — nessun `-> tuple[np.ndarray, np.ndarray]`
- `update_tracker` — nessun `-> list[tuple[int, float, float, float, float, float]]`
- `render_video` — nessun `-> None`
- `encode_with_audio` / `encode_without_audio` — nessun `-> None`

Con mypy o pyright, questi return type sarebbero inferiti come `Any`, eliminando il beneficio del type checking a cascata.

**Come correggere:** Aggiungere i return type alle funzioni pubbliche, almeno nelle API tra moduli.

---

### [MED-02] `FrameProcessors` usa `object` come tipo per i campi opzionali — `person_anonymizer.py:106-114`

**File:** `person_anonymizer/person_anonymizer.py` righe 106-114

**Problema:**
```python
@dataclass
class FrameProcessors:
    clahe_obj: object
    motion_detector: object  # MotionDetector | None
    tracker: object  # BYTETracker | None
    smoother: object  # TemporalSmoother | None
```

I commenti indicano i tipi corretti ma il codice usa `object`. Questo bypassa qualsiasi controllo statico sugli attributi acceduti (`.smooth()`, `.get_ghost_boxes()`, etc.).

**Come correggere:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ultralytics.trackers.byte_tracker import BYTETracker

@dataclass
class FrameProcessors:
    clahe_obj: cv2.CLAHE
    motion_detector: MotionDetector | None
    tracker: BYTETracker | None
    smoother: TemporalSmoother | None
    do_interpolation: bool
```

---

### [MED-03] `camera_calibration.py` usa `sys.exit()` invece di eccezioni — `camera_calibration.py:97,105,119`

**File:** `person_anonymizer/camera_calibration.py` righe 97, 105, 119

**Problema:** Pattern già documentato in MEMORY come anti-pattern per funzioni riusabili. In questo caso `main()` è la sola funzione che chiama `sys.exit()`, e il modulo non è importato altrove — quindi il rischio è basso. Tuttavia, il pattern crea un precedente incoerente con il resto del codebase che usa eccezioni.

**Come correggere:** Estrarre la logica di validazione in funzioni che lanciano eccezioni, lasciare `sys.exit(1)` solo nell'entry point `if __name__ == "__main__":`.

---

### [MED-04] `SSEManager.close` rimuove i subscriber PRIMA che possano ricevere `None` — `web/sse_manager.py:48-53`

**File:** `person_anonymizer/web/sse_manager.py` righe 48-53

**Problema:**
```python
def close(self, job_id: str):
    with self._lock:
        for q in self._subscribers.get(job_id, []):
            q.put(None)
        self._subscribers.pop(job_id, None)  # <-- rimuove DENTRO il lock
```

Il `None` viene messo nelle code (segnale di chiusura) e poi i subscriber vengono rimossi dalla mappa. Nel generatore SSE in `app.py:236-247`, il loop legge `None` e fa `break`, poi il `finally` chiama `sse_manager.unsubscribe(job_id, q)`. Ma a quel punto `job_id` è già stato rimosso da `self._subscribers` — l'`unsubscribe` esegue un `if job_id in self._subscribers:` che è `False`, quindi il `try: self._subscribers[job_id].remove(q)` non viene mai eseguito. Funzionalmente corretto (la coda è già stata rimossa), ma il `finally` in `app.py:247` è un no-op silenzioso.

**Impatto:** Non è un bug ma crea confusione nel debugging: il `finally` sembra fare qualcosa ma non fa nulla se `close()` è già stato chiamato.

---

### [MED-05] `ManualReviewer` non documenta il contratto per `config` come `dict` — `manual_reviewer.py:24-80`

**File:** `person_anonymizer/manual_reviewer.py` righe 24-80

**Problema:** Il costruttore riceve `config: dict` e accede a chiavi specifiche con `.get()` e default hardcoded:
```python
self.auto_color = config.get("auto_color", (0, 255, 0))
```

Ma il resto del codebase usa `PipelineConfig`. Non c'è documentazione dello schema dict atteso. Il wrapper `run_manual_review` passa `config` direttamente — chi lo chiama non sa quali chiavi mettere.

**Come correggere:** Accettare `PipelineConfig` e mappare i campi internamente, oppure documentare lo schema dict con un TypedDict.

---

### [MED-06] `_merge_overlapping_rects` è O(n²) ma il commento dice "n < 50 tipico" senza enforcement — `postprocessing.py:300-315`

**File:** `person_anonymizer/postprocessing.py` righe 300-315

**Problema:** Il commento documenta l'assunzione (corretto), ma non c'è alcun check. Se `normalize_annotations` venisse chiamata su annotazioni generate da un video con molte persone rilevate per frame (>50), il rallentamento sarebbe marcato: O(n²) = 2500 operazioni per frame × numero di frame. Su un video 30fps di 10 minuti = 18000 frame = 45 milioni di comparazioni.

**Come correggere:**
```python
if len(rects) > 100:
    _log.warning("_merge_overlapping_rects: %d rects, performance degraded (O(n^2))", len(rects))
```

---

### [MED-07] Import `logging` dentro la funzione `render_video` — `rendering.py:134-139`

**File:** `person_anonymizer/rendering.py` righe 134-139

**Problema:**
```python
if corrupted_frames > 0:
    import logging
    logging.getLogger(__name__).warning(...)
```

`import logging` dentro il corpo di una funzione è un anti-pattern: viene eseguito ad ogni chiamata che entra nel ramo (sebbene Python cachi i moduli già importati). Il modulo `logging` dovrebbe essere importato a livello di modulo come negli altri file (`tracking.py`, `postprocessing.py`).

**Come correggere:** Aggiungere `import logging` e `_log = logging.getLogger(__name__)` a livello di modulo, come negli altri file.

---

## BASSI / NITPICK (5 problemi)

### [LOW-01] Magic numbers nei codici tasto di `manual_reviewer.py` — righe 13-18

I codici tasto OpenCV sono definiti come costanti nominative (ottimo), ma il valore `255` a riga 428 (`if key == 255`) non ha una costante:
```python
key = cv2.waitKey(30) & 0xFF
if key == 255:
    continue
```
`255` significa "nessun tasto premuto" in OpenCV (0xFF dopo la maschera). Aggiungere `KEY_NONE = 255` tra le costanti.

---

### [LOW-02] `config.py` ha `numpy` come dipendenza di import solo per il type hint — `config.py:12`

```python
import numpy as np
# ...
camera_matrix: np.ndarray | None = None
dist_coefficients: np.ndarray | None = None
```

`numpy` è importato solo per il type hint nei due campi opzionali. Con `from __future__ import annotations` (già presente) e Python 3.10+, si potrebbe usare `"np.ndarray | None"` come stringa oppure importare solo `from numpy import ndarray`. Tuttavia, dato che numpy è comunque una dipendenza del progetto, l'impatto è solo cosmetic.

---

### [LOW-03] `test_config.py:93` verifica `"automatic"` nel commento ma `"auto"` nel codice

**File:** `tests/test_config.py` riga 93-94

```python
def test_operation_mode_is_valid_value(self):
    # Assert — i valori ammessi sono "manual" e "auto"
    assert config.operation_mode in ("manual", "auto")
```

Il commento e il codice sono allineati (corretto). Tuttavia, `PipelineConfig` default è `"manual"` (riga 24 di `config.py`) mentre `_CONFIG_VALIDATORS` in `pipeline_runner.py` ammette solo `"auto"` e `"manual"`. Il test è corretto. Rimane solo il fatto già documentato in MEMORY che il valore `"automatic"` non è mai stato un valore valido — nessun residuo nel codice.

---

### [LOW-04] `web/__init__.py` è vuoto ma necessario per il package — nessun problema

Il file è necessario e corretto. Solo una nota: con Python 3.3+ e namespace packages, non sarebbe strettamente necessario. Ma dato il `sys.path.insert` in `conftest.py` e `app.py`, mantenerlo è la scelta più sicura.

---

### [LOW-05] `find_chessboard_corners` stampa su stdout invece di usare logging — `camera_calibration.py`

```python
print(f"  OK: {Path(path).name}")
print(f"  SKIP: scacchiera non trovata in ...")
```

Incoerente con il resto del codebase che usa `logging`. Non bloccante per un tool CLI standalone.

---

## Problemi di sicurezza

Vedi sezione separata obbligatoria sotto.

---

## Priorità di Intervento

1. **[CRIT-01, CRIT-02, ALTA-04] — Resource leak VideoCapture/VideoWriter** — Pattern sistematico in tre file. Intervento minimo: wrappare in `try/finally`. Rischio concreto su uso web con coda di job.

2. **[CRIT-03] — MotionDetector non thread-safe** — Documentare il vincolo oggi; correggere se il loop viene parallelizzato in futuro.

3. **[ALTA-02] — StdoutCapture monkey-patching sys.stdout** — Soluzione strutturale con logging handler. Non urgente ma il comportamento attuale in produzione è imprevedibile.

4. **[MED-01, MED-02] — Type hints incompleti** — Intervento a basso costo, alto beneficio per la manutenibilità futura.

5. **[MED-07] — Import logging inline** — Fix di 2 righe, elimina un anti-pattern.

---

## Verdict finale

Codebase ben strutturato con separazione dei moduli chiara e test suite copertura buona sui casi funzionali. I tre problemi critici sono tutti varianti dello stesso pattern (mancanza di `try/finally` sui resource handle), risolvibili con un refactoring mirato di 20 righe in totale. Il monkey-patching di `sys.stdout` è l'unica scelta architetturale che potrebbe creare sorprese in produzione.

---

## SECURITY REPORT

### ✅ Nessuna vulnerabilità critica (🔴) rilevata

---

### 🟠 ALTO — `StdoutCapture` può leakare messaggi interni di werkzeug al client SSE

**File:** `person_anonymizer/web/pipeline_runner.py` righe 272-274
**Problema:** `sys.stdout` viene sostituito globalmente. Se werkzeug (in modalità threaded) stampa messaggi di debug/accesso su stdout durante l'elaborazione, questi messaggi vengono catturati e inviati al frontend come eventi SSE `log`. I messaggi werkzeug possono contenere IP del client, URL chiamate, stack trace parziali.
**Come fixare:** Sostituire il monkey-patch con un `logging.Handler` custom che intercetta solo i log del package `person_anonymizer`, lasciando werkzeug sul suo handler.

---

### 🟡 MEDIO — Coordinamento sanitizzazione path: `secure_filename` non garantisce unicità tra job

**File:** `person_anonymizer/web/app.py` righe 123-126
**Problema:** Due upload con lo stesso filename (es. `video.mp4`) in job diversi producono path distinti perché sono in `UPLOAD_DIR/{job_id}/`. Ma due upload nello stesso job sovrascrivono il file precedente senza avviso. Non è un vettore di attacco autonomo in un contesto single-user, ma in un contesto multi-utente sarebbe una race condition scrivibile.
**Rischio:** Basso nel contesto attuale (single-user, `PipelineRunner` limita a un job attivo).

---

### 🟡 MEDIO — `review_update_annotations` non valida i valori numerici delle coordinate come bounded

**File:** `person_anonymizer/web/app.py` righe 340-345
**Problema:** `_validate_annotation_frame` verifica che le coordinate siano `int` o `float` ma non verifica che siano in range valido (es. `[0, frame_w]` × `[0, frame_h]`). Un client malevolo può inviare coordinate come `1e308` (float valido) che causano overflow in numpy quando convertite a `int32` in `obscure_polygon`.
**Impatto:** Crash del processo di rendering con numpy OverflowError, non RCE. Denial of Service su quel job specifico.
**Come fixare:** Aggiungere range check in `_validate_annotation_frame` con un massimo ragionevole (es. 10000 px per dimensione).

---

### ✅ Verifiche positive esplicite

- Nessuna credenziale hardcodata nel codice
- Path traversal bloccato correttamente con `secure_filename` + `resolve() + startswith(UPLOAD_DIR)` su tutti i path
- `validate_job_id` con regex anchored `^[a-f0-9]{12}$` — corretto
- `MAX_CONTENT_LENGTH` configurato (10 GB — intenzionale per video di sorveglianza)
- Handler 413 presente con risposta JSON
- Security headers completi: X-Content-Type-Options, X-Frame-Options, CSP senza unsafe-inline, COOP, CORP, Referrer-Policy, Permissions-Policy
- Rate limiting con flask-limiter su tutti gli endpoint sensibili
- `_ALLOWED_FIELDS` come whitelist esplicita per i parametri configurabili dalla web UI
- Nessun path assoluto nella response API (solo `job_id` + `filename`)
- `yolo_model` validato con whitelist esplicita di modelli noti — blocca path traversal a livello applicativo
- `debug=False` hardcoded nell'entry point di produzione
