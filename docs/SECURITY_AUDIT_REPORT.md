# Security Audit Report — Person Anonymizer v7.1

**Data**: 2026-03-31
**Scope**: Tutti i 12 file sorgente del progetto
**Tipo**: Security Code Review (P0 → P1 → P2)
**Contesto**: Tool locale Flask senza autenticazione, potenzialmente esposto in rete

---

## Executive Summary

Il progetto è stato progettato come tool locale (`127.0.0.1:5000`), e in quel contesto la superficie d'attacco è minima. Tuttavia, se deployato in rete o condiviso via GitHub, presenta **vulnerabilità serie**: path traversal, SSRF via `video_path` controllato dal client, XSS, assenza di autenticazione e security headers. Non ci sono vulnerabilità crittografiche (non c'è crittografia da rompere), ma i problemi di input validation sono multipli.

**Totale finding**: 12 (CRITICAL 2 · HIGH 3 · MEDIUM 4 · LOW 3)

---

## CRITICAL (2 finding)

### Path Traversal via `video_path` — Arbitrary File Read/Process (CWE-22, CWE-918)
**Location**: `person_anonymizer/web/app.py:120-123`
**Issue**: L'endpoint `/api/start` riceve `video_path` dal client e lo passa direttamente alla pipeline **senza validazione**:
```python
video_path = data.get("video_path")
if not video_path or not Path(video_path).exists():
    return jsonify({"error": "Video non trovato"}), 404
```
Il client invia il path ricevuto dall'upload, ma **niente impedisce di inviare un path arbitrario**. Un attaccante può forzare la pipeline a processare qualsiasi file leggibile dal processo Python, inclusi file di sistema.
**Impact**: Arbitrary file read (il video viene elaborato e i frame estratti sono accessibili via `/api/review/frame/`). In combinazione con la review web, l'attaccante può estrarre frame JPEG da qualsiasi video sul filesystem.
**Fix**:
```diff
- video_path = data.get("video_path")
- if not video_path or not Path(video_path).exists():
+ video_path = data.get("video_path")
+ if not video_path:
+     return jsonify({"error": "video_path mancante"}), 400
+ # Verifica che il path sia dentro UPLOAD_DIR
+ resolved = Path(video_path).resolve()
+ if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
+     return jsonify({"error": "Path non autorizzato"}), 403
+ if not resolved.exists():
      return jsonify({"error": "Video non trovato"}), 404
```
**Priority**: P0

---

### Path Traversal via `job_id` — Directory Traversal (CWE-22)
**Location**: `person_anonymizer/web/app.py:96-103, 305-323`
**Issue**: `job_id` viene usato direttamente nella costruzione di path senza sanitizzazione:
```python
job_dir = UPLOAD_DIR / job_id    # Se job_id = "../../etc" → UPLOAD_DIR/../../etc
job_out = OUTPUT_DIR / job_id     # Stesso problema
```
Negli endpoint `/api/upload-json`, `/api/download/<job_id>/<file_type>`, `/api/outputs/<job_id>`, un `job_id` malevolo come `../../../etc` causa directory traversal.
**Impact**: Lettura di file arbitrari via download, scrittura di file arbitrari via upload-json.
**Fix**:
```python
import re
def validate_job_id(job_id: str) -> bool:
    """Verifica che job_id sia un hex string sicuro."""
    return bool(re.match(r'^[a-f0-9]{12}$', job_id))
```
Applica questa validazione in ogni endpoint che riceve `job_id`.
**Priority**: P0

---

## HIGH (3 finding)

### XSS via `innerHTML` con filename utente (CWE-79)
**Location**: `person_anonymizer/web/static/js/app.js:539-544`
**Issue**: I nomi dei file di output vengono iniettati nel DOM via `innerHTML`:
```javascript
item.innerHTML = `
    <span class="result-name">${f.name}</span>
    <span class="result-size">${f.size_mb} MB</span>
    <a href="/api/download/${jid}/${f.type}" ...>Scarica</a>
`;
```
Se un filename contiene `<script>alert(1)</script>` o `<img onerror=...>`, viene eseguito nel browser.
**Impact**: Stored XSS — l'attaccante carica un file con nome malevolo, l'output viene visualizzato con il nome iniettato.
**Fix**: Usa `document.createElement` + `textContent` per i dati utente:
```javascript
const nameSpan = document.createElement("span");
nameSpan.className = "result-name";
nameSpan.textContent = f.name;  // textContent è sicuro
```
**Priority**: P1

---

### XSS via `showToast()` con `innerHTML` (CWE-79)
**Location**: `person_anonymizer/web/static/js/app.js:67-71`
**Issue**: La funzione `showToast()` usa `innerHTML` con il parametro `message`:
```javascript
toast.innerHTML = `
    ${TOAST_ICONS[type] || TOAST_ICONS.info}
    <span class="toast-message">${message}</span>
    ...
`;
```
I messaggi di errore dal server (es. `data.error`) vengono passati direttamente a `showToast()` in più punti (righe 205, 247, 341, 425). Se il server include input utente nel messaggio di errore (es. il nome del file), è XSS.
**Impact**: Reflected XSS tramite messaggi di errore del server.
**Fix**: Separa l'HTML statico (icone) dal testo dinamico:
```javascript
const msgSpan = toast.querySelector(".toast-message");
msgSpan.textContent = message;  // Sicuro
```
**Priority**: P1

---

### Mancata validazione filename in upload (CWE-22)
**Location**: `person_anonymizer/web/app.py:71`
**Issue**: Il filename dell'upload viene usato dopo `Path(f.filename).name`, che rimuove directory traversal (`../`), ma **non sanitizza caratteri speciali** nel nome stesso. Su alcuni filesystem, caratteri come `;`, `|`, `$`, backtick possono essere problematici se il filename viene successivamente usato in un contesto shell (e ffmpeg lo fa via `ffmpeg-python`).
```python
safe_name = Path(f.filename).name  # "video.mp4; rm -rf /" → "video.mp4; rm -rf /"
```
**Impact**: Potenziale command injection se il filename finisce in un contesto shell.
**Fix**: Usa `werkzeug.utils.secure_filename()`:
```python
from werkzeug.utils import secure_filename
safe_name = secure_filename(f.filename)
```
**Priority**: P1

---

## MEDIUM (4 finding)

### Nessun security header (CWE-693)
**Location**: `person_anonymizer/web/app.py`
**Issue**: L'applicazione Flask non imposta nessun security header: no CSP, no X-Content-Type-Options, no X-Frame-Options, no HSTS.
**Impact**: Il browser non ha istruzioni per proteggersi da XSS, clickjacking, MIME sniffing.
**Fix**: Aggiungi un middleware o usa `flask-talisman`:
```python
@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'"
    return response
```
**Priority**: P2

---

### Nessun rate limiting (CWE-770)
**Location**: Tutti gli endpoint in `app.py`
**Issue**: `MAX_CONTENT_LENGTH = 10 GB` senza rate limiting. Un attaccante può inviare upload da 10 GB in loop, saturando disco e memoria.
**Impact**: Denial of Service tramite disk exhaustion.
**Fix**: Usa `flask-limiter` per limitare le richieste per IP, e riduci `MAX_CONTENT_LENGTH` a un valore ragionevole (es. 2 GB).
**Priority**: P2

---

### `os.chdir()` globale in thread — Race condition (CWE-367)
**Location**: `person_anonymizer/web/pipeline_runner.py:246, 350`
**Issue**: `os.chdir()` è un'operazione **process-wide**, non thread-safe. Mentre il thread della pipeline cambia il cwd, i thread Flask che servono richieste HTTP operano con un cwd inaspettato.
**Impact**: Comportamento imprevedibile per `send_file()` con path relativi, potenziale file serving errato.
**Fix**: Elimina `os.chdir()` e usa path assoluti per il modello YOLO.
**Priority**: P2

---

### Nessuna autenticazione su endpoint sensibili (CWE-306)
**Location**: Tutti gli endpoint in `app.py`
**Issue**: Upload, avvio pipeline, download risultati, modifica annotazioni — tutto accessibile senza autenticazione. Il binding su `127.0.0.1` mitiga il rischio, ma non lo elimina (browser extensions, CSRF da siti malevoli in un contesto same-origin).
**Impact**: Qualsiasi processo locale o pagina web con accesso a localhost può interagire con l'applicazione.
**Fix**: Per un tool locale, almeno aggiungi un token di sessione generato all'avvio e verificato in ogni richiesta. Per deployment in rete, implementa autenticazione completa.
**Priority**: P2

---

## LOW (3 finding)

### Errori esposti al client senza filtro
**Location**: `person_anonymizer/web/pipeline_runner.py:332-340`
**Issue**: L'eccezione viene convertita in stringa e inviata al client via SSE:
```python
except Exception as e:
    self._sse.emit(job_id, "error", {"message": str(e)})
```
Alcune eccezioni Python includono path del filesystem, nomi di moduli interni, o dettagli di configurazione.
**Impact**: Information disclosure — path interni e dettagli implementativi esposti.
**Fix**: Invia un messaggio generico al client, logga i dettagli internamente.
**Priority**: P2

---

### Upload directory non pulita automaticamente
**Location**: `person_anonymizer/web/app.py:24-27`
**Issue**: I file caricati in `uploads/` e i risultati in `outputs/` non vengono mai eliminati. Nessun meccanismo di cleanup.
**Impact**: Disk exhaustion nel tempo, specialmente con video grandi.
**Fix**: Implementa un cleanup periodico (cron job o background thread) che elimina job più vecchi di N ore.
**Priority**: P2

---

### SSE endpoint senza validazione job_id
**Location**: `person_anonymizer/web/app.py:140`
**Issue**: L'endpoint `/api/progress?job_id=X` non valida il formato di `job_id`. Un `job_id` inesistente crea semplicemente una coda vuota che rimane in memoria.
**Impact**: Memory leak se un attaccante genera migliaia di subscription con job_id casuali.
**Fix**: Valida il formato `job_id` e rifiuta con 400 se non è un hex di 12 caratteri.
**Priority**: P2

---

## Checklist di Sicurezza

### P0 — Da fixare PRIMA di qualsiasi merge/deploy
- [ ] Validare `video_path` contro `UPLOAD_DIR` (path traversal)
- [ ] Validare `job_id` come hex string di 12 caratteri in tutti gli endpoint
- [ ] Usare `secure_filename()` per i file uploadati

### P1 — Da fixare nel prossimo sprint
- [ ] Eliminare `innerHTML` con dati utente — usare `textContent` + `createElement`
- [ ] Sanitizzare messaggi di errore prima di inviarli al frontend

### P2 — Hardening
- [ ] Aggiungere security headers (CSP, X-Content-Type-Options, X-Frame-Options)
- [ ] Aggiungere rate limiting sugli endpoint di upload
- [ ] Eliminare `os.chdir()` e usare path assoluti
- [ ] Implementare cleanup periodico dei file temporanei
- [ ] Validare `job_id` nell'endpoint SSE
- [ ] Aggiungere autenticazione basica (token di sessione)

---

## Nota di contesto

Questo progetto è progettato per uso locale (`127.0.0.1`), il che riduce significativamente la superficie d'attacco. Le vulnerabilità P0 e P1 diventano **critiche solo se l'applicazione viene esposta in rete** o se il repository viene usato come base per un deployment condiviso. Tuttavia, fixare le vulnerabilità P0 è buona pratica anche per uso locale, perché:

1. Le abitudini di sviluppo sicuro si trasferiscono ai progetti futuri
2. Il binding su localhost non protegge da attacchi originati dal browser (CSRF, DNS rebinding)
3. Se il progetto finisce su GitHub, qualcuno lo deployerà su `0.0.0.0:5000` senza leggere il README
