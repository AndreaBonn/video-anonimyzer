<div align="center">

# Security Documentation

[English](#security-english) | [Italiano](#security-italiano)

</div>

---

<a id="security-english"></a>
## English

[Passa all'italiano](#security-italiano)

### Overview

Person Anonymizer is designed for **local use or trusted internal networks**. It is not intended to be exposed directly on the public internet. All security measures are implemented following a defense-in-depth approach: even in a controlled environment, every layer of the application validates its inputs, sets appropriate HTTP headers, and limits resource consumption.

---

### 1. Input Validation

#### Job ID
All endpoints that accept a job identifier validate it against the regex `^[a-f0-9]{12}$` before any file system or processing operation. Requests with malformed job IDs are rejected with HTTP 400.

#### File Upload
- `secure_filename()` is applied to every uploaded filename before storage.
- The file extension is checked against the `SUPPORTED_EXTENSIONS` whitelist.
- `MAX_CONTENT_LENGTH` is set to 2 GB to prevent resource exhaustion via oversized uploads.

#### JSON Upload
- The body is parsed and verified to be a `dict` before use.
- Size is capped at 100 MB.

#### Path Traversal Protection
Every file path derived from user input is resolved with `Path.resolve()` and verified to start with the expected upload or output directory (`startswith(UPLOAD_DIR)`). Requests that attempt to escape the designated directories are rejected.

#### Config Parameters
- A `_ALLOWED_FIELDS` whitelist defines the set of parameters that can be set via the API.
- Each parameter has a type validator and an explicit numeric range check.
- `_BOOL_FIELDS` enforces that boolean parameters are cast correctly.

#### Annotation Validation
- Each annotation must be a `dict` with the expected structure.
- Polygons must contain at least 3 points.
- Coordinates must be numeric and within the range [-10000, +10000].

#### Frame Index Validation
`frame_idx` is validated against the range `[0, total_frames)` before any frame is read from disk.

#### YOLO Model Path
Accepted model filenames are restricted to a whitelist (`yolov8x.pt`, `yolov8n.pt`). The resolved path is also verified with `startswith()` against the models directory.

---

### 2. HTTP Security Headers

The following headers are set on every response:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Prevents clickjacking |
| `Content-Security-Policy` | `default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'` | Blocks XSS and inline scripts |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limits referrer leakage |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | Disables sensitive browser APIs |
| `Cross-Origin-Opener-Policy` | `same-origin` | Cross-origin isolation |
| `Cross-Origin-Resource-Policy` | `same-origin` | Blocks cross-origin resource loading |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains; preload` | Enforces HTTPS (when already on a secure connection) |
| `X-Request-ID` | 16-character hex UUID | Request traceability |

---

### 3. Rate Limiting

Rate limits are enforced per IP address using an in-memory store (appropriate for single-instance deployments; use Redis for multi-worker setups).

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `POST /api/upload` | 10/min | Prevents upload flooding |
| `POST /api/upload-json` | 20/min | Prevents JSON upload flooding |
| `POST /api/start` | 5/min | Limits pipeline start attempts |
| `GET /api/progress` | 10/min | Limits SSE connections |
| `POST /api/stop` | 10/min | Prevents stop flooding |
| `GET /api/status` | 60/min | Status polling |
| `GET /api/review/status` | 60/min | Review polling |
| `GET /api/review/frame/<idx>` | 120/min | Frame navigation |
| `PUT /api/review/annotations/<idx>` | 60/min | Annotation updates |
| `POST /api/review/confirm` | 5/min | Review confirmation |

---

### 4. CSRF Protection

- All `POST`, `PUT`, and `DELETE` requests must include the `X-Requested-With` header.
- If an `Origin` header is present, it must match the `Host` of the request.
- Same-origin requests without an `Origin` header are permitted.
- Static file endpoints are excluded from the check.

---

### 5. XSS Prevention

- The CSP policy blocks all inline scripts (`script-src 'self'`).
- The frontend uses `textContent` (not `innerHTML`) for all dynamic data rendering.
- `StdoutCapture._sanitize_message` strips absolute paths from SSE log messages before they reach the client.

---

### 6. File Management and Cleanup

- Uploads are stored in per-job directories: `uploads/<job_id>/`.
- Outputs are stored in per-job directories: `outputs/<job_id>/`.
- A background daemon thread automatically removes directories older than 1 hour, running every 10 minutes.
- `secure_filename()` is applied to all uploaded file names.
- No absolute paths are exposed in API responses.

---

### 7. SSE (Server-Sent Events) Security

| Protection | Detail |
|------------|--------|
| Max subscribers | 5 per job — prevents resource exhaustion |
| Queue size | maxsize 200 — backpressure; silent drop if full |
| Connection timeout | 2 hours — prevents zombie connections |
| Heartbeat | Every 60 seconds |
| Message sanitization | Absolute paths stripped from log messages sent to clients |

---

### 8. Pipeline Security

- **Single-job enforcement**: only one pipeline can run at a time. A second `POST /api/start` while a job is active returns HTTP 409.
- **Cooperative stop**: a `stop_event` is passed through the pipeline and checked at each stage; stopping is non-forceful and clean.
- **YOLO model path validation**: whitelist + `resolve()` + `startswith()` check applied before loading any model.
- **Config validation**: every parameter in `PipelineConfig` is validated for type and range before the pipeline starts.

---

### 9. Configuration

| Setting | Detail |
|---------|--------|
| `SECRET_KEY` | Read from the `FLASK_SECRET_KEY` environment variable |
| Host | Configurable via `FLASK_HOST`; default `127.0.0.1` (localhost only) |
| Port | Configurable via `FLASK_PORT` |
| Debug mode | `debug=False` hardcoded — never enabled in production |
| Development server warning | Explicit warning logged if Werkzeug is used (gunicorn recommended) |

---

### 10. Deployment Recommendations

- **Do not expose directly on the internet** — place behind a reverse proxy (nginx, Caddy).
- Configure TLS/HTTPS on the reverse proxy.
- Set `FLASK_SECRET_KEY` to a cryptographically random value (32+ bytes).
- For production, use: `gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 'person_anonymizer.web.app:app'`
- Consider adding authentication (HTTP Basic Auth or equivalent) if the tool is accessible over a network.

---

### 11. Known Limitations

- No user authentication (designed for local / trusted-network use).
- Rate limiting is in-memory and does not persist across restarts or across multiple workers.
- No audit log of operations performed.
- YOLO models are downloaded from the internet on first run.
- Uploaded JSON annotations are validated for structure but not for semantic correctness.

---

### 12. Reporting Vulnerabilities

If you discover a security vulnerability, please open a **private** issue on GitHub or contact the maintainer directly. Do not disclose vulnerabilities in public issues.

---

<a id="security-italiano"></a>
## Italiano

[Switch to English](#security-english)

### Panoramica

Person Anonymizer è progettato per **uso locale o su reti interne fidate**. Non è concepito per essere esposto direttamente su internet. Tutte le misure di sicurezza seguono un approccio defense-in-depth: anche in un ambiente controllato, ogni livello dell'applicazione valida i propri input, imposta gli header HTTP appropriati e limita il consumo di risorse.

---

### 1. Validazione degli Input

#### Job ID
Tutti gli endpoint che accettano un identificatore di job lo validano con la regex `^[a-f0-9]{12}$` prima di qualsiasi operazione sul filesystem o di elaborazione. Le richieste con job ID malformati vengono rifiutate con HTTP 400.

#### Upload di file
- `secure_filename()` è applicata a ogni nome di file caricato prima del salvataggio.
- L'estensione del file viene verificata contro la whitelist `SUPPORTED_EXTENSIONS`.
- `MAX_CONTENT_LENGTH` è impostato a 2 GB per prevenire l'esaurimento delle risorse tramite upload di dimensioni eccessive.

#### Upload JSON
- Il corpo della richiesta viene analizzato e verificato come `dict` prima dell'uso.
- La dimensione è limitata a 100 MB.

#### Protezione da Path Traversal
Ogni percorso file derivato dall'input utente viene risolto con `Path.resolve()` e verificato che inizi con la directory di upload o output attesa (`startswith(UPLOAD_DIR)`). Le richieste che tentano di uscire dalle directory designate vengono rifiutate.

#### Parametri di Configurazione
- Una whitelist `_ALLOWED_FIELDS` definisce l'insieme dei parametri impostabili tramite API.
- Ogni parametro dispone di un validatore di tipo e un controllo esplicito del range numerico.
- `_BOOL_FIELDS` garantisce che i parametri booleani vengano convertiti correttamente.

#### Validazione delle Annotazioni
- Ogni annotazione deve essere un `dict` con la struttura attesa.
- I poligoni devono contenere almeno 3 punti.
- Le coordinate devono essere numeriche e nel range [-10000, +10000].

#### Validazione dell'Indice Frame
`frame_idx` viene validato nell'intervallo `[0, total_frames)` prima che qualsiasi frame venga letto dal disco.

#### Path del Modello YOLO
I nomi di file di modello accettati sono limitati a una whitelist (`yolov8x.pt`, `yolov8n.pt`). Il path risolto viene anche verificato con `startswith()` rispetto alla directory dei modelli.

---

### 2. Header HTTP di Sicurezza

I seguenti header vengono impostati su ogni risposta:

| Header | Valore | Scopo |
|--------|--------|-------|
| `X-Content-Type-Options` | `nosniff` | Previene il MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Previene il clickjacking |
| `Content-Security-Policy` | `default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'` | Blocca XSS e script inline |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limita la perdita del referrer |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | Disabilita le API browser sensibili |
| `Cross-Origin-Opener-Policy` | `same-origin` | Isolamento cross-origin |
| `Cross-Origin-Resource-Policy` | `same-origin` | Blocca il caricamento di risorse cross-origin |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains; preload` | Forza HTTPS (solo se già su connessione sicura) |
| `X-Request-ID` | UUID hex da 16 caratteri | Tracciabilità delle richieste |

---

### 3. Rate Limiting

I limiti di velocità sono applicati per indirizzo IP tramite uno store in-memory (adeguato per istanza singola; usare Redis per configurazioni multi-worker).

| Endpoint | Limite | Motivo |
|----------|--------|--------|
| `POST /api/upload` | 10/min | Previene l'upload flooding |
| `POST /api/upload-json` | 20/min | Previene il JSON upload flooding |
| `POST /api/start` | 5/min | Limita i tentativi di avvio pipeline |
| `GET /api/progress` | 10/min | Limita le connessioni SSE |
| `POST /api/stop` | 10/min | Previene lo stop flooding |
| `GET /api/status` | 60/min | Polling dello stato |
| `GET /api/review/status` | 60/min | Polling della revisione |
| `GET /api/review/frame/<idx>` | 120/min | Navigazione frame |
| `PUT /api/review/annotations/<idx>` | 60/min | Aggiornamento annotazioni |
| `POST /api/review/confirm` | 5/min | Conferma revisione |

---

### 4. Protezione CSRF

- Tutte le richieste `POST`, `PUT` e `DELETE` devono includere l'header `X-Requested-With`.
- Se è presente l'header `Origin`, deve corrispondere all'`Host` della richiesta.
- Le richieste same-origin prive di header `Origin` sono permesse.
- Gli endpoint per i file statici sono esclusi dal controllo.

---

### 5. Prevenzione XSS

- La policy CSP blocca tutti gli script inline (`script-src 'self'`).
- Il frontend usa `textContent` (non `innerHTML`) per il rendering di tutti i dati dinamici.
- `StdoutCapture._sanitize_message` rimuove i path assoluti dai messaggi SSE di log prima che raggiungano il client.

---

### 6. Gestione dei File e Pulizia Automatica

- Gli upload sono salvati in directory per-job: `uploads/<job_id>/`.
- Gli output sono salvati in directory per-job: `outputs/<job_id>/`.
- Un thread daemon in background rimuove automaticamente le directory più vecchie di 1 ora, con cadenza ogni 10 minuti.
- `secure_filename()` è applicata a tutti i nomi di file caricati.
- Nessun path assoluto è esposto nelle risposte delle API.

---

### 7. Sicurezza SSE (Server-Sent Events)

| Protezione | Dettaglio |
|------------|-----------|
| Subscriber massimi | 5 per job — previene l'esaurimento delle risorse |
| Dimensione coda | maxsize 200 — backpressure; drop silenzioso se la coda è piena |
| Timeout connessione | 2 ore — previene le connessioni zombie |
| Heartbeat | Ogni 60 secondi |
| Sanitizzazione messaggi | Path assoluti rimossi dai messaggi di log inviati al client |

---

### 8. Sicurezza della Pipeline

- **Single-job enforcement**: una sola pipeline può essere in esecuzione alla volta. Una seconda `POST /api/start` mentre un job è attivo restituisce HTTP 409.
- **Stop cooperativo**: uno `stop_event` viene passato attraverso la pipeline e verificato a ogni stadio; l'interruzione è pulita e non forzata.
- **Validazione path modello YOLO**: whitelist + `resolve()` + `startswith()` applicati prima di caricare qualsiasi modello.
- **Validazione configurazione**: ogni parametro in `PipelineConfig` viene validato per tipo e range prima dell'avvio della pipeline.

---

### 9. Configurazione

| Impostazione | Dettaglio |
|--------------|-----------|
| `SECRET_KEY` | Letto dalla variabile d'ambiente `FLASK_SECRET_KEY` |
| Host | Configurabile tramite `FLASK_HOST`; default `127.0.0.1` (solo localhost) |
| Porta | Configurabile tramite `FLASK_PORT` |
| Debug mode | `debug=False` hardcoded — mai abilitato in produzione |
| Avviso server di sviluppo | Warning esplicito loggato se viene usato Werkzeug (consigliato gunicorn) |

---

### 10. Raccomandazioni di Deployment

- **Non esporre direttamente su internet** — usare un reverse proxy (nginx, Caddy).
- Configurare TLS/HTTPS sul reverse proxy.
- Impostare `FLASK_SECRET_KEY` con un valore casuale crittograficamente sicuro (32+ byte).
- Per produzione usare: `gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 'person_anonymizer.web.app:app'`
- Considerare l'aggiunta di autenticazione (HTTP Basic Auth o equivalente) se il tool è accessibile in rete.

---

### 11. Limitazioni Note

- Nessuna autenticazione utente (progettato per uso locale / rete fidata).
- Il rate limiting è in-memory e non persiste tra riavvii né funziona con configurazioni multi-worker.
- Nessun audit log delle operazioni eseguite.
- I modelli YOLO vengono scaricati da internet al primo avvio.
- Le annotazioni JSON caricate vengono validate nella struttura ma non nella correttezza semantica.

---

### 12. Segnalazione di Vulnerabilità

Se scopri una vulnerabilità di sicurezza, apri una issue **privata** su GitHub oppure contatta direttamente il maintainer. Non divulgare vulnerabilità in issue pubbliche.
