# Security Audit Report — Person Anonymizer v7.1

**Data:** 2026-04-01
**Auditor:** Claude Opus 4.6 (automated)
**Scope:** Tutti i file sorgente Python, JavaScript, HTML, configurazione
**Metodologia:** Analisi statica manuale di tutto il codice sorgente

---

## Riepilogo Esecutivo

Il progetto presenta una buona postura di sicurezza di base: validazione job_id, `secure_filename`, path traversal check, security headers, rate limiting, CSP e sanitizzazione output SSE. Tuttavia l'audit ha identificato **17 problemi** di cui **3 critici**, **5 alti**, **6 medi** e **3 bassi**.

| Severita | Conteggio |
|----------|-----------|
| CRITICO  | 3         |
| ALTO     | 5         |
| MEDIO    | 6         |
| BASSO    | 3         |

---

## Problemi Identificati

---

### [C-01] CRITICO — Nessuna pulizia automatica dei file uploadati (Disk Exhaustion / DoS)

**File:** `person_anonymizer/web/app.py` (righe 31-34, 119-131)

**Descrizione:**
I file caricati in `uploads/` e gli output in `outputs/` non vengono mai cancellati. Un attaccante puo esaurire lo spazio disco inviando upload ripetuti entro il rate limit (10/min = 100 GB/10 min con file da 10 GB).

**Impatto:** Denial of Service per esaurimento disco, crash del server e potenzialmente dell'intero host.

**Remediation:**
```python
# 1. Implementare pulizia asincrona con TTL (es. 1 ora)
import time, threading

def _cleanup_old_jobs(upload_dir, output_dir, max_age_seconds=3600):
    """Rimuove job directory piu vecchie di max_age_seconds."""
    now = time.time()
    for base_dir in (upload_dir, output_dir):
        if not base_dir.exists():
            continue
        for job_dir in base_dir.iterdir():
            if job_dir.is_dir():
                age = now - job_dir.stat().st_mtime
                if age > max_age_seconds:
                    shutil.rmtree(job_dir, ignore_errors=True)

# 2. Avviare un thread di pulizia periodica
def _start_cleanup_thread(upload_dir, output_dir, interval=600):
    def loop():
        while True:
            _cleanup_old_jobs(upload_dir, output_dir)
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

# 3. Limitare il numero massimo di job contemporanei
MAX_CONCURRENT_JOBS = 20
```

---

### [C-02] CRITICO — MAX_CONTENT_LENGTH di 10 GB eccessivo

**File:** `person_anonymizer/web/app.py` (riga 26)

**Descrizione:**
```python
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB max upload
```
10 GB per singolo upload e un valore estremo. Combinato con l'assenza di pulizia (C-01), un attaccante puo saturare il disco molto rapidamente anche restando nei rate limit.

**Impatto:** Amplifica il rischio di C-01. Un singolo upload malevolo puo consumare 10 GB di disco.

**Remediation:**
- Ridurre `MAX_CONTENT_LENGTH` a un valore ragionevole (es. 2 GB o 500 MB a seconda del caso d'uso reale)
- Aggiungere validazione della dimensione effettiva del file dopo il salvataggio
- Implementare quota per job_id (max 1 video per job)

---

### [C-03] CRITICO — Secret Key Flask mancante (sessioni non firmate)

**File:** `person_anonymizer/web/app.py` (riga 25)

**Descrizione:**
L'applicazione Flask non configura `app.secret_key`. Sebbene al momento non usi sessioni Flask, qualsiasi futuro uso di `session`, `flash()`, o middleware CSRF risultera insicuro. Inoltre, flask-limiter in alcune configurazioni puo dipendere dalla session per il tracking.

**Impatto:** Se in futuro venissero aggiunte sessioni o CSRF protection, sarebbero vulnerabili. Attualmente rischio latente.

**Remediation:**
```python
import os
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
```
Aggiungere `FLASK_SECRET_KEY` a `.env.example`.

---

### [A-01] ALTO — Assenza di CSRF protection sugli endpoint POST/PUT

**File:** `person_anonymizer/web/app.py` (tutti gli endpoint POST/PUT)

**Descrizione:**
Nessun endpoint POST/PUT ha protezione CSRF. Un attaccante puo creare una pagina malevola che induce un utente autenticato (o sullo stesso network) a:
- Caricare file video arbitrari (`POST /api/upload`)
- Avviare pipeline di processing (`POST /api/start`)
- Modificare annotazioni (`PUT /api/review/annotations/<idx>`)
- Confermare review (`POST /api/review/confirm`)
- Fermare pipeline in esecuzione (`POST /api/stop`)

**Impatto:** Cross-Site Request Forgery — un utente che visita un sito malevolo mentre ha il tool aperto puo subire azioni non autorizzate.

**Remediation:**
```python
# Opzione 1: Flask-WTF CSRF
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)

# Per le API JSON, verificare header custom (non inviabile da form cross-origin):
@app.before_request
def csrf_check():
    if request.method in ("POST", "PUT", "DELETE"):
        if not request.is_json and "multipart/form-data" not in request.content_type:
            pass  # form upload, gestire con token
        # Verificare header X-Requested-With per XHR
        if not request.headers.get("X-Requested-With"):
            # Per le chiamate fetch, verificare Origin header
            origin = request.headers.get("Origin", "")
            if origin and origin not in ALLOWED_ORIGINS:
                return jsonify({"error": "CSRF check failed"}), 403
```

---

### [A-02] ALTO — Endpoint `/api/stop` senza rate limit

**File:** `person_anonymizer/web/app.py` (righe 262-271)

**Descrizione:**
L'endpoint `POST /api/stop` non ha `@limiter.limit()`. Un attaccante puo bombardare questo endpoint per interferire con pipeline in esecuzione. A differenza degli altri endpoint, non c'e alcun throttling.

**Impatto:** Possibile interferenza con l'elaborazione, flood di richieste senza limiti.

**Remediation:**
```python
@app.route("/api/stop", methods=["POST"])
@limiter.limit("10 per minute")
def stop_pipeline():
```

---

### [A-03] ALTO — Endpoint `/api/status` e `/api/review/status` senza rate limit

**File:** `person_anonymizer/web/app.py` (righe 277-292)

**Descrizione:**
Gli endpoint `GET /api/status` e `GET /api/review/status` non hanno rate limiting. Possono essere usati per DoS polling.

**Remediation:**
```python
@app.route("/api/status")
@limiter.limit("60 per minute")
def status():

@app.route("/api/review/status")
@limiter.limit("60 per minute")
def review_status():
```

---

### [A-04] ALTO — Upload JSON senza validazione del contenuto

**File:** `person_anonymizer/web/app.py` (righe 137-163)

**Descrizione:**
L'endpoint `POST /api/upload-json` salva il file JSON sul disco senza validarne il contenuto. Un file `.json` malformato o con payload gigantesco potrebbe:
1. Causare crash alla deserializzazione successiva (in `_load_annotations_from_json`)
2. Contenere strutture profondamente nidificate per consumare memoria (Billion Laughs / JSON bomb)

**Impatto:** DoS per crash o consumo memoria alla fase di caricamento annotazioni.

**Remediation:**
```python
import json

# Dopo il salvataggio, validare il JSON:
try:
    content = f.read()
    if len(content) > 100 * 1024 * 1024:  # max 100 MB
        return jsonify({"error": "File JSON troppo grande"}), 400
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or "frames" not in parsed:
        return jsonify({"error": "Struttura JSON non valida"}), 400
except json.JSONDecodeError:
    return jsonify({"error": "JSON non valido"}), 400
```

---

### [A-05] ALTO — Pipeline config `yolo_model` non sufficientemente restrittivo

**File:** `person_anonymizer/web/pipeline_runner.py` (righe 26, 386-390)

**Descrizione:**
Il validatore accetta solo `"yolov8x.pt"` e `"yolov8n.pt"` come valori per `yolo_model` (riga 26), il che e buono. Tuttavia, il percorso del modello viene costruito concatenando con il path del modulo:
```python
yolo_path = pa_dir / config.yolo_model
if yolo_path.exists():
    config.yolo_model = str(yolo_path)
```
Se la validazione venisse aggirata o modificata, questo potrebbe portare a path traversal.

**Impatto:** Attualmente mitigato dalla whitelist stretta. Rischio se la whitelist viene ampliata senza cautela.

**Remediation:**
Aggiungere un check esplicito che il percorso risolto sia dentro `pa_dir`:
```python
yolo_path = (pa_dir / config.yolo_model).resolve()
if not str(yolo_path).startswith(str(pa_dir.resolve())):
    raise ValueError(f"Percorso modello non autorizzato: {yolo_path}")
```

---

### [M-01] MEDIO — SSE stream senza timeout di connessione globale

**File:** `person_anonymizer/web/app.py` (righe 221-256)

**Descrizione:**
La connessione SSE (`/api/progress`) ha un heartbeat ogni 60 secondi ma nessun timeout massimo di connessione. Un client puo mantenere una connessione aperta indefinitamente, consumando risorse server (thread, file descriptor).

**Impatto:** Resource exhaustion con connessioni SSE persistenti. Il limite di 5 subscriber per job mitiga ma non elimina il problema.

**Remediation:**
```python
def generate():
    max_duration = 7200  # 2 ore max
    start = time.monotonic()
    # ...
    while True:
        if time.monotonic() - start > max_duration:
            yield 'event: timeout\ndata: {"message": "Connessione scaduta"}\n\n'
            break
        # ... resto del loop
```

---

### [M-02] MEDIO — `debug=False` hardcodato ma nessun guard per produzione

**File:** `person_anonymizer/web/app.py` (riga 464)

**Descrizione:**
```python
app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
```
Il debug e disabilitato, il binding e su localhost. Buono. Tuttavia:
1. Il server di sviluppo Werkzeug non e adatto alla produzione
2. Non c'e documentazione su come deployare con un WSGI server (gunicorn/waitress)
3. Se qualcuno cambiasse `host` in `"0.0.0.0"`, il servizio sarebbe esposto senza TLS

**Impatto:** Rischio di esposizione accidentale in rete senza protezioni adeguate.

**Remediation:**
```python
if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Usa gunicorn o waitress per la produzione: "
        "gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 web.app:app",
        stacklevel=1,
    )
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
```

---

### [M-03] MEDIO — `sys.path.insert(0, ...)` puo causare import hijacking

**File:** `person_anonymizer/web/app.py` (righe 18-19)

**Descrizione:**
```python
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
```
Manipolare `sys.path` con `insert(0, ...)` da priorita a quel percorso su tutto il resto. Se un attaccante riesce a scrivere un file Python in quella directory (es. tramite upload con nome creativo), potrebbe eseguire codice arbitrario.

**Impatto:** Import hijacking se combinato con la possibilita di scrivere file nella directory parent. Attualmente mitigato da `secure_filename` che impedisce directory traversal negli upload.

**Remediation:**
- Ristrutturare come package Python con `__init__.py` appropriati e usare import relativi o assoluti
- In alternativa, usare `sys.path.append()` invece di `insert(0, ...)`

---

### [M-04] MEDIO — Monkey-patching globale di tqdm e stdout non thread-safe

**File:** `person_anonymizer/web/pipeline_runner.py` (righe 171-310)

**Descrizione:**
`TqdmCapture.install()` e `StdoutCapture.install()` modificano globalmente `tqdm.tqdm` e `sys.stdout`. Se due pipeline venissero eseguite concorrentemente (es. se in futuro si rimuovesse il lock singolo), il monkey-patching interferirebbe tra i job.

**Impatto:** Attualmente mitigato dal lock che impedisce pipeline concorrenti. Rischio di race condition se il design cambia.

**Remediation:**
- Documentare esplicitamente che il design supporta solo 1 pipeline alla volta
- Aggiungere un assert nel codice: `assert not self._thread.is_alive()` prima del patching
- Valutare l'uso di `contextlib.redirect_stdout` per scope piu limitato

---

### [M-05] MEDIO — Nessuna validazione `frame_idx` in `update_annotations`

**File:** `person_anonymizer/web/review_state.py` (righe 177-188)

**Descrizione:**
```python
def update_annotations(self, frame_idx, frame_data):
    with self._lock:
        self._annotations[frame_idx] = copy.deepcopy(frame_data)
```
Non c'e verifica che `frame_idx` sia nel range `[0, total_frames)`. Un client malevolo puo iniettare annotazioni per indici arbitrari (negativi o superiori al totale frame), che verrebbero poi passate alla pipeline di rendering.

**Nota:** L'endpoint `PUT /api/review/annotations/<int:frame_idx>` in `app.py` (riga 348) non valida il range del frame_idx contro i metadati della review.

**Impatto:** Annotazioni fuori range potrebbero causare comportamenti imprevisti nel rendering.

**Remediation:**
```python
# In app.py, prima di rs.update_annotations():
meta = rs.get_metadata()
if not (0 <= frame_idx < meta["total_frames"]):
    return jsonify({"error": "frame_idx fuori range"}), 400
```

---

### [M-06] MEDIO — Dipendenza `flask-limiter` senza storage backend configurato

**File:** `person_anonymizer/web/app.py` (riga 29)

**Descrizione:**
```python
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=[])
```
Nessun `storage_uri` configurato. Flask-Limiter usa per default `memory://` che:
1. Non persiste tra riavvii
2. Non funziona con multiple istanze/worker (se si usa gunicorn con >1 worker)
3. Puo consumare memoria illimitata se molti IP diversi generano richieste

**Impatto:** Rate limiting non affidabile in deployment multi-worker o dopo riavvii.

**Remediation:**
```python
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",  # esplicito; usare "redis://..." per produzione
    default_limits=[],
)
```

---

### [B-01] BASSO — CSP manca `script-src` esplicito

**File:** `person_anonymizer/web/app.py` (righe 70-76)

**Descrizione:**
```python
"Content-Security-Policy": (
    "default-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'"
)
```
Non c'e `script-src 'self'`. Poiche `default-src 'self'` copre `script-src`, gli script inline non sono permessi (buono). Tuttavia, e best practice dichiarare esplicitamente `script-src` per chiarezza e per evitare errori futuri.

**Remediation:**
```python
"Content-Security-Policy": (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'"
)
```

---

### [B-02] BASSO — Version disclosure nell'HTML

**File:** `person_anonymizer/web/templates/index.html` (riga 6, 16)

**Descrizione:**
```html
<title>Person Anonymizer v7.1</title>
<h1>Person Anonymizer <span class="version">v7.1</span></h1>
```
Il numero di versione esatto e esposto nell'interfaccia. Questo puo aiutare un attaccante a identificare vulnerabilita specifiche della versione.

**Impatto:** Information disclosure minore.

**Remediation:**
Rimuovere la versione esatta dal frontend o renderla generica. Mantenere la versione solo nei log interni o endpoint `/api/version` protetto.

---

### [B-03] BASSO — Nessun header `X-Request-ID` per tracciabilita

**File:** `person_anonymizer/web/app.py`

**Descrizione:**
Manca un request ID univoco nelle response headers. Questo rende difficile correlare richieste HTTP con log server per debugging e incident response.

**Remediation:**
```python
@app.before_request
def add_request_id():
    request.request_id = uuid.uuid4().hex[:16]

@app.after_request
def add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(request, "request_id", "unknown")
    return response
```

---

## Aspetti Positivi

L'audit ha identificato diverse best practice gia implementate:

1. **Path Traversal Protection:** `secure_filename()` + `.resolve()` + `startswith()` check in `start_pipeline()` (righe 189-191, 202-204)
2. **Job ID Validation:** regex `^[a-f0-9]{12}$` per prevenire injection nel job_id
3. **Security Headers:** `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `CORP`, `COOP`, CSP, HSTS condizionale
4. **Rate Limiting:** presente sui principali endpoint di upload e processing
5. **XSS Prevention nel frontend:** uso di `textContent` invece di `innerHTML` per dati utente in `app.js`
6. **Input Validation:** validazione approfondita dei parametri config in `pipeline_runner.py` con whitelist e range check
7. **Error Handling:** errori generici al client, dettagli solo nei log (`app.logger.exception`)
8. **SSE Sanitization:** path assoluti rimossi dai messaggi SSE inviati al client (`StdoutCapture._sanitize_message`)
9. **Subscriber Limit:** max 5 subscriber per job SSE
10. **Debug Mode Off:** `debug=False` nel server Flask
11. **Localhost Binding:** `host="127.0.0.1"` di default
12. **Dipendenze pinnate:** `requirements.txt` con versioni esatte

---

## Analisi Dipendenze

| Pacchetto | Versione | Note |
|-----------|----------|------|
| ultralytics | 8.4.21 | Verificare CVE recenti — aggiornamenti frequenti |
| opencv-python | 4.13.0.92 | OK, versione recente |
| ffmpeg-python | 0.2.0 | Wrapper; la sicurezza dipende dal binario ffmpeg installato |
| flask | 3.1.0 | OK |
| flask-limiter | 3.12 | OK |
| numpy | 2.4.2 | OK |
| pytest | 8.3.4 | Solo dev, no rischio produzione |
| tqdm | 4.67.3 | OK |

**Nota:** Manca `pip-audit` o un equivalente nella CI pipeline per scansione automatica vulnerabilita.

**Remediation:**
```bash
pip install pip-audit
pip-audit -r requirements.txt
```

---

## Matrice Rischio e Priorita

| ID | Severita | Sforzo Fix | Priorita |
|----|----------|------------|----------|
| C-01 | CRITICO | Medio | 1 |
| C-02 | CRITICO | Basso | 1 |
| C-03 | CRITICO | Basso | 1 |
| A-01 | ALTO | Medio | 2 |
| A-02 | ALTO | Basso | 2 |
| A-03 | ALTO | Basso | 2 |
| A-04 | ALTO | Medio | 2 |
| A-05 | ALTO | Basso | 3 |
| M-01 | MEDIO | Basso | 3 |
| M-02 | MEDIO | Basso | 3 |
| M-03 | MEDIO | Medio | 4 |
| M-04 | MEDIO | Basso | 4 |
| M-05 | MEDIO | Basso | 3 |
| M-06 | MEDIO | Basso | 3 |
| B-01 | BASSO | Basso | 5 |
| B-02 | BASSO | Basso | 5 |
| B-03 | BASSO | Basso | 5 |

---

## Raccomandazioni Generali

1. **Implementare pulizia automatica** dei file uploadati e degli output (C-01) — e la priorita assoluta
2. **Ridurre MAX_CONTENT_LENGTH** a un valore ragionevole per il caso d'uso reale (C-02)
3. **Aggiungere CSRF protection** almeno tramite verifica dell'header `Origin` (A-01)
4. **Aggiungere rate limit** a tutti gli endpoint mancanti (A-02, A-03)
5. **Validare il contenuto JSON** prima di salvarlo (A-04)
6. **Integrare `pip-audit`** nel workflow CI/CD
7. **Documentare il deployment** con WSGI server (gunicorn/waitress) per produzione
8. **Aggiungere test di sicurezza** nella test suite (path traversal, upload file malevoli, rate limiting)

---

*Report generato automaticamente — verificare manualmente le remediation prima di applicarle.*
