# Security Audit Report — Person Anonymizer

**Data audit**: 2026-04-01
**Versione codebase**: 7.1
**Auditor**: Senior Security Engineering (automated static analysis)
**Scope**: Analisi statica completa di tutti i file Python in `person_anonymizer/` e `tests/`

---

## Executive Summary

Person Anonymizer è un'applicazione Flask single-user progettata per girare in locale (`host="127.0.0.1"`). Questo contesto mitiga significativamente la superficie di attacco rispetto a un servizio esposto pubblicamente. Tuttavia, emergono vulnerabilità concrete che possono essere sfruttate in scenari realistici: un utente della stessa macchina, un'applicazione compromessa sul sistema, o un eventuale futuro cambio di deployment.

Il codice mostra una baseline di sicurezza positiva: `MAX_CONTENT_LENGTH` configurato, `debug=False` in produzione, header di sicurezza implementati via `after_request`, `secure_filename` usato su tutti gli upload, validazione `job_id` tramite regex, e path traversal su `video_path` bloccato correttamente (riga 163 di `app.py`). Queste misure dimostrano consapevolezza dei rischi principali.

**Sono stati trovati 7 finding**: 0 CRITICAL, 2 HIGH, 3 MEDIUM, 2 LOW.

Il finding HIGH più urgente è il **path traversal su `review_json`** (P0): il parametro arriva dal client, viene trasmesso dal frontend al backend senza validazione server-side del path, e confluisce direttamente in `open()`. Un attaccante locale può leggere file arbitrari del filesystem. Il secondo finding HIGH riguarda la **disclosure di path assoluti** nella risposta API, che consente enumerazione della struttura interna del server.

---

## Trust Boundary Diagram

```
[Browser/Client]
      |
      | HTTP (localhost:5000)
      v
[Flask Web Layer — app.py]
      |
      |-- /api/upload        → salva file in uploads/{job_id}/
      |-- /api/upload-json   → salva JSON in uploads/{job_id}/
      |-- /api/start         → legge video_path + review_json dal client  ← TRUST BOUNDARY ROTTA
      |-- /api/progress      → SSE stream
      |-- /api/download      → serve file da outputs/{job_id}/
      |-- /api/review/*      → accede a ReviewState (thread pipeline)
      |
      v
[PipelineRunner — pipeline_runner.py]
      |
      | crea thread separato
      v
[run_pipeline — person_anonymizer.py]
      |
      |-- legge video da filesystem
      |-- legge review_json da filesystem  ← NESSUNA VALIDAZIONE PATH
      |-- scrive output in outputs/{job_id}/
      v
[YOLO + OpenCV + ffmpeg]

Legenda trust boundary:
- UPLOAD_DIR/OUTPUT_DIR: sotto controllo server
- video_path: validato correttamente (riga 163)
- review_json: NON validato — attaccante controlla il path
- json_path nella response upload: espone path assoluto al client
```

---

## P0 — Exploitable Now

### Finding 1 — Path Traversal su `review_json` (CWE-22)

**Severità**: HIGH
**Location**: `person_anonymizer/web/app.py:170` e `person_anonymizer/web/pipeline_runner.py:400`

**Issue**: Il parametro `review_json` viene estratto dal payload JSON del client in `/api/start` (riga 170) senza alcuna validazione del path, e trasmesso direttamente come argomento `--review` alla pipeline (riga 400 di `pipeline_runner.py`). La pipeline lo usa in `_load_annotations_from_json` (riga 64 di `person_anonymizer.py`) con `open(review_json)`. Non c'è alcun controllo che il path risieda dentro `UPLOAD_DIR`.

Il flusso normale vede il client inviare il `json_path` ricevuto dalla risposta di `/api/upload-json`. Ma nulla impedisce al client di inviare un path arbitrario come `review_json: "/etc/passwd"` o `review_json: "/home/user/.ssh/id_rsa"`.

**Impact**: Lettura di file arbitrari del filesystem con i permessi del processo Flask. Su sistemi Linux tipici, questo include file di configurazione, chiavi SSH, variabili d'ambiente di altri processi in `/proc/*/environ`, e potenzialmente credenziali applicative.

**Fix**:
```diff
# In app.py, dopo riga 170
review_json = data.get("review_json")
+ if review_json is not None:
+     resolved_json = Path(review_json).resolve()
+     if not str(resolved_json).startswith(str(UPLOAD_DIR.resolve())):
+         return jsonify({"error": "Path review_json non autorizzato"}), 403
+     if not resolved_json.exists():
+         return jsonify({"error": "File JSON non trovato"}), 404
```

---

### Finding 2 — Disclosure di Path Assoluti del Server (CWE-209)

**Severità**: HIGH
**Location**: `person_anonymizer/web/app.py:108` e `app.py:140`

**Issue**: Due endpoint restituiscono path assoluti del filesystem nei payload JSON di risposta:
- `POST /api/upload` risponde con `"path": str(dest)` — es. `/home/user/video-anonimizer/person_anonymizer/web/uploads/abc123def456/video.mp4`
- `POST /api/upload-json` risponde con `"json_path": str(dest)` — il path assoluto viene poi ri-inviato dal frontend a `/api/start` come `review_json`

Esporre path assoluti del server è una violazione della difesa in profondità (CWE-209). Un attaccante conosce: struttura directory, username di sistema, percorso di installazione. Nel caso di `json_path`, il path viene attivamente riusato come vettore per il finding 1.

**Impact**: Enumerazione della struttura del server. Il `json_path` funge da trasportatore del path assoluto che abilita il path traversal (Finding 1). Anche dopo aver fixato il Finding 1, esporre il path reale è una best practice violata.

**Fix**:
```diff
# In app.py, upload_video() — riga 107-109
- return jsonify(
-     {"job_id": job_id, "filename": safe_name, "size_mb": round(size_mb, 2), "path": str(dest)}
- )
+ return jsonify(
+     {"job_id": job_id, "filename": safe_name, "size_mb": round(size_mb, 2)}
+ )

# In app.py, upload_json() — riga 140
- return jsonify({"json_path": str(dest), "filename": safe_name})
+ return jsonify({"filename": safe_name})

# In app.py, start_pipeline() — riga 170
# Il client non deve più fornire il path: lo ricostruisce il server
- review_json = data.get("review_json")
+ review_json_filename = data.get("review_json_filename")
+ review_json = None
+ if review_json_filename:
+     safe = secure_filename(review_json_filename)
+     candidate = (UPLOAD_DIR / job_id / safe).resolve()
+     if str(candidate).startswith(str(UPLOAD_DIR.resolve())) and candidate.exists():
+         review_json = str(candidate)
+     else:
+         return jsonify({"error": "File JSON non trovato per questo job"}), 404
```

---

## P1 — Exploitable with Effort

### Finding 3 — Mancanza di Rate Limiting su Upload e Avvio Pipeline (CWE-770)

**Severità**: MEDIUM
**Location**: `person_anonymizer/web/app.py:74` (`/api/upload`), `app.py:146` (`/api/start`)

**Issue**: Non è presente alcun rate limiting sugli endpoint di upload video e avvio pipeline. Un attaccante locale (o un'applicazione compromessa) può:
1. Inviare upload multipli rapidi per esaurire lo spazio disco con la directory `uploads/`
2. Avviare pipeline multiple in rapida successione. Nota: `PipelineRunner.start()` previene l'avvio di una seconda pipeline se una è già in esecuzione (riga 321 di `pipeline_runner.py`), ma solo per lo stesso processo. Non esiste protezione contro le richieste in sé.
3. Saturare la CPU con operazioni di inferenza YOLO attraverso richieste successive di `/api/start`.

Il limite `MAX_CONTENT_LENGTH = 10 GB` protegge da un singolo upload gigante, ma non da cento upload da 100 MB ciascuno.

**Impact**: DoS locale — esaurimento disco, CPU al 100%, sistema non responsivo. In un contesto multi-utente o con accesso di rete, l'impatto sale.

**Fix**:
```python
# Aggiungere flask-limiter al requirements.txt
# flask-limiter==3.5.0

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app=app, key_func=get_remote_address)

@app.route("/api/upload", methods=["POST"])
@limiter.limit("10 per minute")
def upload_video():
    ...

@app.route("/api/start", methods=["POST"])
@limiter.limit("5 per minute")
def start_pipeline():
    ...
```

---

### Finding 4 — Validazione Filename JSON con `endswith` Prima di `secure_filename` (CWE-20)

**Severità**: MEDIUM
**Location**: `person_anonymizer/web/app.py:121`

**Issue**: La validazione del filename per l'upload JSON avviene in ordine sbagliato:
```python
if not f.filename or not f.filename.endswith(".json"):   # riga 121 — controlla PRIMA
    return jsonify({"error": "File deve essere .json"}), 400
# ...
safe_name = secure_filename(f.filename)   # riga 134 — sanitizza DOPO
```

Il problema è che `f.filename` è un valore interamente sotto controllo del client (header `Content-Disposition` multipart). Un filename come `../../etc/shadow.json` supererebbe il check `endswith(".json")`. Sebbene `secure_filename` poi elimini i `../`, il check a riga 121 opera su dati non sanitizzati.

Il rischio effettivo qui è basso perché `secure_filename` viene chiamato immediatamente dopo e il salvataggio usa il risultato sanificato, non il filename originale. Tuttavia, il pattern è sbagliato e può ingannare chi modifica il codice in futuro credendo che `f.filename` sia già sicuro dopo il check a riga 121.

**Fix**:
```diff
- if not f.filename or not f.filename.endswith(".json"):
-     return jsonify({"error": "File deve essere .json"}), 400
- # ...
- safe_name = secure_filename(f.filename)
- if not safe_name:
+ safe_name = secure_filename(f.filename or "")
+ if not safe_name:
+     return jsonify({"error": "Nome file non valido"}), 400
+ if not safe_name.endswith(".json"):
      return jsonify({"error": "File deve essere .json"}), 400
```

---

### Finding 5 — Assenza di Global Error Handler Flask (CWE-209)

**Severità**: MEDIUM
**Location**: `person_anonymizer/web/app.py` — file intero

**Issue**: Flask in modalità `debug=False` (correttamente impostato a riga 384) non espone traceback HTML di default. Tuttavia, in assenza di un `@app.errorhandler(500)` esplicito, eventuali eccezioni non gestite in un endpoint restituiscono la pagina di errore HTML di Werkzeug. In certi scenari (es. errore in `config_defaults()` durante la serializzazione) questo include informazioni sulla versione di Flask, il nome dell'applicazione, e a volte frammenti dello stack.

L'endpoint `/api/config/defaults` (riga 299) esegue `from config import PipelineConfig` e `asdict(cfg)` all'interno della route, senza try/except. Se `PipelineConfig` fallisse per qualsiasi ragione, verrebbe restituita una risposta HTML non strutturata invece di un JSON di errore.

**Fix**:
```python
@app.errorhandler(500)
def internal_error(e):
    app.logger.exception("Internal server error")
    return jsonify({"error": "Errore interno del server"}), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File troppo grande", "max_mb": 10240}), 413
```

---

## P2 — Hardening

### Finding 6 — HSTS Assente negli Header di Sicurezza (CWE-319)

**Severità**: LOW
**Location**: `person_anonymizer/web/app.py:47-60`

**Issue**: La funzione `add_security_headers` imposta correttamente `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy` e `Content-Security-Policy`, ma omette `Strict-Transport-Security`. Attualmente l'applicazione gira su HTTP (`http://127.0.0.1:5000`), quindi HSTS non è applicabile nell'uso corrente. Ma se l'applicazione venisse esposta via HTTPS (es. dietro un reverse proxy nginx), la mancanza di HSTS permetterebbe downgrade attack.

**Fix**:
```python
# Aggiungere condizionalmente se l'applicazione è servita su HTTPS
if request.is_secure:
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
```

---

### Finding 7 — Video di Sorveglianza Reali nelle Directory `uploads/` e `outputs/` non Coperte da Gitignore (CWE-312)

**Severità**: LOW
**Location**: `.gitignore` + directory `person_anonymizer/web/uploads/` e `person_anonymizer/web/outputs/`

**Issue**: Il `.gitignore` copre correttamente i pattern `*.mp4`, `*.avi` etc. e le directory `input/` e `output/` (senza slash finale). Tuttavia le directory `person_anonymizer/web/uploads/` e `person_anonymizer/web/outputs/` non sono esplicitamente nel `.gitignore`. La verifica con `git ls-files` ha confermato che solo i file `.gitkeep` sono tracciati, non i video. Il rischio è quindi basso allo stato attuale, ma la protezione è implicita (basata sul match dell'estensione) piuttosto che esplicita (basata sulla directory).

I video di sorveglianza sono dati personali ai sensi del GDPR. Se un `.avi` o `.mp4` sfuggisse al gitignore (es. perché il codice scrivesse un file con estensione inattesa come `.tmp` poi rinominato), il dato personale finirebbe nel repository.

**Fix**:
```gitignore
# Aggiungere esplicitamente al .gitignore
person_anonymizer/web/uploads/*/
person_anonymizer/web/outputs/*/
!person_anonymizer/web/uploads/.gitkeep
!person_anonymizer/web/outputs/.gitkeep
```

---

## Elementi Verificati — Nessuna Vulnerabilità

Le seguenti categorie sono state verificate e risultano gestite correttamente:

- **SQL Injection**: non applicabile — nessun database relazionale
- **Command Injection**: `ffmpeg` viene invocato tramite la libreria `ffmpeg-python` che costruisce il comando internamente senza `shell=True`. Nessun `subprocess.run(..., shell=True)` nel codebase
- **Path Traversal su `video_path`**: correttamente bloccato con `Path.resolve()` e `startswith(UPLOAD_DIR)` a riga 163 di `app.py`
- **Path Traversal su `job_id`**: correttamente bloccato dalla regex `^[a-f0-9]{12}$` in `validate_job_id()` (riga 44)
- **Insecure Deserialization**: nessun uso di `pickle.loads()` o `yaml.load()` senza SafeLoader
- **SSRF**: nessun fetch di URL forniti dall'utente
- **Hardcoded secrets**: nessuna credenziale nel sorgente
- **XSS**: l'applicazione non riflette input utente in template HTML. I dati JSON sono serializzati tramite `jsonify()`
- **Debug mode**: `debug=False` confermato a riga 384 di `app.py`
- **File upload extension**: `SUPPORTED_EXTENSIONS` allowlist a riga 31, `secure_filename` applicato, estensione verificata su `Path(f.filename).suffix.lower()`
- **Variabili d'ambiente**: nessuna credenziale in `.env` committata (`.env` è in `.gitignore`)
- **Race condition pipeline**: `PipelineRunner` usa correttamente `threading.Lock` su `_current_job_id` (riga 310 di `pipeline_runner.py`)
- **CORS**: non configurato (default Flask: nessun header CORS), appropriato per app localhost
- **CSP script-src**: non include `unsafe-eval` o `unsafe-inline`

---

## Recommended Security Roadmap

### Sprint immediato (P0/P1 — blocca deploy su rete)

1. **Fix Finding 1 — Path traversal su `review_json`**
   Aggiungere validazione `resolve() + startswith(UPLOAD_DIR)` su `review_json` in `app.py:170` prima di passarlo a `pipeline_runner.start()`. È la stessa logica già applicata correttamente a `video_path` alla riga 163. Costo: 5 righe.

2. **Fix Finding 2 — Eliminare `path` e `json_path` dalle risposte API**
   Rimuovere `"path"` dalla risposta di `/api/upload` e `"json_path"` da `/api/upload-json`. Modificare il frontend (`app.js:263`) per non memorizzare il path. In `/api/start`, ricostruire il path lato server da `job_id` + `filename` invece di accettarlo dal client.

### Sprint corrente (P1)

3. **Fix Finding 3 — Rate limiting**
   Aggiungere `flask-limiter` a `requirements.txt` e decorare `/api/upload` (10/min) e `/api/start` (5/min). Impedisce abuso dello spazio disco e saturazione CPU da parte di client non cooperativi.

4. **Fix Finding 4 — Ordine validazione filename JSON**
   Invertire l'ordine: chiamare `secure_filename()` prima di `endswith()`. Costo: 3 righe.

5. **Fix Finding 5 — Global error handler**
   Aggiungere `@app.errorhandler(500)` e `@app.errorhandler(413)` che restituiscono JSON strutturato. Impedisce leak di informazioni su errori imprevisti.

### Prossimo sprint (P2)

6. **Fix Finding 6 — HSTS condizionale**
   Aggiungere HSTS solo se `request.is_secure`. Nessun impatto sull'uso corrente localhost.

7. **Fix Finding 7 — Gitignore esplicito per uploads/outputs**
   Aggiungere regole esplicite di directory al `.gitignore`. Protezione proattiva per contenuti GDPR-rilevanti.

---

## Note sul Contesto

L'applicazione è progettata per uso locale single-user (`host="127.0.0.1"`). Questo contiene significativamente il blast radius di tutti i finding: un attaccante deve avere accesso locale alla macchina. Se il deployment dovesse cambiare (es. nginx reverse proxy per accesso su rete locale aziendale), i finding P0 diventerebbero immediatamente critici e andrebbero fixati prima di qualsiasi cambio di binding.

Il finding più importante da fixare **indipendentemente dal contesto** è il Finding 1 (path traversal su `review_json`): richiede meno di 10 righe di codice e la logica di fix è già presente nel codebase (riga 162-164 di `app.py` per `video_path`). Non c'è giustificazione per non applicarla anche a `review_json`.
