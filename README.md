# Person Anonymizer Tool v7.1

Strumento per l'oscuramento automatico di persone in video di sorveglianza.
Pipeline multi-strategia basata su YOLO v8 con rilevamento multi-scala, tracking ByteTrack, revisione manuale interattiva e normalizzazione delle annotazioni.

Disponibile in due modalità: **linea di comando (CLI)** con revisione interattiva OpenCV, oppure **interfaccia web** con dashboard in tempo reale.

Progettato per telecamere fisse con angolazione grandangolare, dove le persone possono apparire di piccole dimensioni (30-100px).

## Indice

- [Funzionalita principali](#funzionalita-principali)
- [Requisiti di sistema](#requisiti-di-sistema)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
  - [Modalità CLI](#modalità-cli)
  - [Parametri CLI](#parametri-cli)
  - [Workflow consigliato](#workflow-consigliato)
- [Interfaccia web](#interfaccia-web)
  - [Avvio dell'interfaccia web](#avvio-dellinterfaccia-web)
  - [Caricare un video](#caricare-un-video)
  - [Configurazione parametri](#configurazione-parametri)
  - [Monitoraggio elaborazione](#monitoraggio-elaborazione)
  - [Download dei risultati](#download-dei-risultati)
  - [Rielaborazione da JSON](#rielaborazione-da-json)
- [Pipeline di elaborazione](#pipeline-di-elaborazione)
- [Interfaccia di revisione manuale](#interfaccia-di-revisione-manuale)
- [Calibrazione fish-eye](#calibrazione-fish-eye)
- [File di output](#file-di-output)
- [Configurazione avanzata](#configurazione-avanzata)
- [Formati video supportati](#formati-video-supportati)
- [Struttura del progetto](#struttura-del-progetto)
- [Limitazioni note](#limitazioni-note)
- [Licenza](#licenza)

## Funzionalita principali

- **Rilevamento multi-strategia**: inferenza YOLO su 4 scale (1.0x, 1.5x, 2.0x, 2.5x) combinata con sliding window 3x3 e Test-Time Augmentation
- **Tracking ByteTrack**: mantiene la continuita degli ID delle persone tra frame consecutivi
- **Temporal smoothing**: media mobile sulle coordinate dei bounding box per eliminare oscillazioni
- **Intensita adattiva**: oscuramento proporzionale all'altezza della figura rilevata
- **Revisione manuale**: interfaccia interattiva OpenCV per aggiungere, modificare o eliminare poligoni di oscuramento
- **Normalizzazione annotazioni**: conversione di poligoni irregolari in rettangoli regolari con unificazione delle aree sovrapposte
- **Correzione fish-eye**: undistortion ottica tramite parametri di calibrazione camera
- **Motion detection**: frame differencing per ottimizzare l'elaborazione sulle sole zone con movimento
- **Interpolazione sub-frame**: generazione di frame virtuali per video a basso framerate (< 15 fps)
- **Verifica post-rendering**: secondo passaggio YOLO sul video oscurato per segnalare persone ancora visibili
- **Oscuramento**: pixelation (default) o gaussian blur, applicato esclusivamente all'interno dei poligoni
- **Interfaccia web**: dashboard Flask con upload drag-and-drop, configurazione parametri, progresso in tempo reale via SSE e download risultati

## Requisiti di sistema

- Python 3.9 o superiore
- ffmpeg (per il reintegro dell'audio nel video di output)
- GPU CUDA consigliata per prestazioni ottimali (funziona anche su CPU)

### Installazione ffmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (con Chocolatey)
choco install ffmpeg
```

## Installazione

```bash
# Clona il repository
git clone <repository-url>
cd video-anonimizer

# Crea e attiva l'ambiente virtuale
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Installa le dipendenze
pip install -r person_anonymizer/requirements.txt
```

### Dipendenze

| Pacchetto     | Versione  | Descrizione                           |
| ------------- | --------- | ------------------------------------- |
| ultralytics   | 8.4.21    | YOLO v8 per il rilevamento oggetti    |
| opencv-python | 4.13.0.92 | Computer vision e interfaccia grafica |
| ffmpeg-python | 0.2.0     | Binding Python per ffmpeg             |
| tqdm          | 4.67.3    | Barre di progresso                    |
| numpy         | 2.4.2     | Calcolo numerico                      |
| flask         | 3.1.0     | Server web per l'interfaccia grafica  |

Al primo avvio il modello YOLO (`yolov8x.pt`) viene scaricato automaticamente (~130 MB).

## Utilizzo

Il tool può essere usato in due modalità: **CLI** (linea di comando) oppure **interfaccia web** (browser).

### Modalità CLI

```bash
# Elaborazione standard con revisione manuale (consigliato)
python person_anonymizer/person_anonymizer.py video.mp4

# Modalita automatica (senza revisione manuale)
python person_anonymizer/person_anonymizer.py video.mp4 -M auto

# Oscuramento con blur invece di pixelation
python person_anonymizer/person_anonymizer.py video.mp4 -m blur

# Output in un percorso specifico
python person_anonymizer/person_anonymizer.py video.mp4 -o /percorso/output.mp4

# Senza video debug e senza report CSV
python person_anonymizer/person_anonymizer.py video.mp4 --no-debug --no-report

# Ricaricare annotazioni JSON per una nuova revisione manuale
python person_anonymizer/person_anonymizer.py video.mp4 --review video_annotations.json

# Normalizzare i poligoni e renderizzare direttamente
python person_anonymizer/person_anonymizer.py video.mp4 --review video_annotations.json --normalize

# Normalizzare con blur
python person_anonymizer/person_anonymizer.py video.mp4 --review video_annotations.json --normalize -m blur --no-debug
```

### Parametri CLI

| Parametro        | Tipo                   | Default                  | Descrizione                                                                                                        |
| ---------------- | ---------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `input`          | posizionale            | *obbligatorio*           | Percorso del video da elaborare                                                                                    |
| `-M`, `--mode`   | `manual` \| `auto`     | `manual`                 | Modalita operativa. `manual` apre la revisione interattiva dopo la detection, `auto` salta la revisione            |
| `-o`, `--output` | stringa                | `{input}_anonymized.mp4` | Percorso del file video di output                                                                                  |
| `-m`, `--method` | `pixelation` \| `blur` | `pixelation`             | Metodo di oscuramento                                                                                              |
| `--no-debug`     | flag                   | disattivo                | Disabilita la generazione del video debug                                                                          |
| `--no-report`    | flag                   | disattivo                | Disabilita la generazione del report CSV                                                                           |
| `--review`       | stringa                | `None`                   | Percorso di un file JSON di annotazioni esistente. Salta la detection e carica i poligoni dal JSON                 |
| `--normalize`    | flag                   | disattivo                | Normalizza i poligoni in rettangoli e unifica le aree sovrapposte. Richiede `--review`. Salta la revisione manuale |

### Workflow consigliato

Il workflow tipico prevede tre passaggi:

**1. Detection + revisione manuale**

```bash
python person_anonymizer/person_anonymizer.py video.mp4
```

Esegue la detection automatica, apre l'interfaccia di revisione per correggere eventuali errori, poi produce il video oscurato e il JSON delle annotazioni.

**2. Controllo del risultato**

Guardare il video debug (`video_debug.mp4`) per verificare che tutte le persone siano state coperte e non ci siano false positive.

**3. (Opzionale) Normalizzazione e re-rendering**

Se i poligoni manuali hanno forme irregolari o si sovrappongono:

```bash
python person_anonymizer/person_anonymizer.py video.mp4 --review video_annotations.json --normalize --no-debug
```

## Interfaccia web

L'interfaccia web offre una dashboard completa per gestire l'elaborazione dei video direttamente dal browser, senza bisogno di usare il terminale.

### Avvio dell'interfaccia web

```bash
# Assicurarsi di aver attivato l'ambiente virtuale
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Avviare il server Flask
python person_anonymizer/web/app.py
```

Il server si avvia su `http://127.0.0.1:5000`. Aprire questo indirizzo nel browser.

### Caricare un video

1. Nella sezione **Video Input** (pannello sinistro), trascinare il file video nell'area tratteggiata oppure cliccare per selezionarlo dal file system
2. Formati supportati: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`
3. Dopo il caricamento vengono mostrati il nome del file e la dimensione

### Configurazione parametri

L'interfaccia web permette di configurare tutti i parametri di elaborazione tramite menu espandibili nel pannello sinistro. I parametri sono organizzati in sezioni.

#### Impostazioni base

| Parametro | Opzioni | Default | Descrizione |
|-----------|---------|---------|-------------|
| Modalità operativa | Auto / Manuale | Auto | `Auto` elabora senza revisione manuale. `Manuale` apre l'interfaccia OpenCV per la revisione interattiva |
| Metodo di oscuramento | Pixelation / Blur | Pixelation | Tipo di effetto applicato sulle persone rilevate |
| Intensità | 2 — 50 | 10 | Dimensione del blocco (pixelation) o raggio del kernel (blur). Valori più alti = oscuramento più forte |
| Padding | 0 — 60 px | 15 | Margine aggiuntivo intorno alla persona rilevata, in pixel |

#### Impostazioni di rilevamento

| Parametro | Range | Default | Descrizione |
|-----------|-------|---------|-------------|
| Modello YOLO | `yolov8x.pt` / `yolov8n.pt` | `yolov8x.pt` | `yolov8x.pt` è più preciso ma più lento. `yolov8n.pt` è veloce ma meno accurato |
| Confidenza detection | 0.05 — 0.95 | 0.20 | Soglia minima di confidenza: valori bassi rilevano più persone (con più falsi positivi), valori alti sono più selettivi |
| Soglia NMS IoU | 0.1 — 0.9 | 0.55 | Soglia di sovrapposizione per eliminare detection duplicate. Valori bassi eliminano più duplicati |

#### Funzionalità avanzate

Ogni funzionalità può essere attivata o disattivata tramite un interruttore (toggle):

| Funzionalità | Default | Descrizione |
|-------------|---------|-------------|
| Sliding Window | Attivo | Inferenza su griglia 3x3 con 30% di sovrapposizione per rilevare persone piccole |
| Tracking ByteTrack | Attivo | Mantiene l'identità delle persone tra frame consecutivi |
| Temporal Smoothing | Attivo | Stabilizza i bounding box con media mobile per ridurre oscillazioni |
| Intensità adattiva | Attivo | Regola automaticamente l'intensità dell'oscuramento in base alla dimensione della persona |
| Verifica post-rendering | Attivo | Secondo passaggio YOLO sul video oscurato per verificare che nessuna persona sia rimasta visibile |
| Correzione fish-eye | Attivo | Corregge la distorsione delle lenti grandangolari (richiede parametri di calibrazione) |
| Motion detection | Disattivo | Limita l'analisi alle sole zone con movimento rilevato |
| Interpolazione sub-frame | Disattivo | Genera frame virtuali intermedi per video a basso framerate (< 15 fps) |
| Video debug | Attivo | Genera un secondo video con i poligoni di oscuramento visibili |
| Report CSV | Attivo | Genera un file CSV con le statistiche di rilevamento per ogni frame |

### Monitoraggio elaborazione

Dopo aver premuto **Avvia Elaborazione**, il pannello destro mostra:

1. **Indicatore di fase**: 5 fasi visualizzate con stato (in attesa / attiva / completata):
   - Rilevamento — detection automatica delle persone
   - Revisione — revisione manuale (solo in modalità manuale)
   - Rendering — applicazione dell'oscuramento
   - Verifica — controllo post-rendering
   - Audio — reintegro dell'audio originale

2. **Barra di progresso**: percentuale di completamento con conteggio dei frame elaborati e velocità (frame/secondo)

3. **Console**: log in tempo reale dal backend, con messaggi colorati per tipo (info, successo, errore, cambio fase)

L'elaborazione può essere interrotta in qualsiasi momento premendo **Interrompi**.

### Download dei risultati

Al termine dell'elaborazione, la sezione **Risultati** elenca tutti i file generati con dimensione e pulsante di download individuale:

- `*_anonymized.mp4` — video finale con persone oscurate e audio
- `*_debug.mp4` — video con poligoni visibili (se attivato)
- `*_report.csv` — statistiche di rilevamento (se attivato)
- `*_annotations.json` — annotazioni riutilizzabili per rielaborazione

### Rielaborazione da JSON

La sezione **Revisione da JSON** permette di ricaricare un file `_annotations.json` generato da un'elaborazione precedente per:

- Rielaborare il video con parametri diversi (es. cambiare metodo da pixelation a blur)
- Normalizzare i poligoni in rettangoli e unificare le aree sovrapposte
- Applicare l'oscuramento senza ripetere la fase di detection

Procedura:
1. Caricare il video originale nella sezione Video Input
2. Espandere la sezione **Revisione da JSON**
3. Caricare il file `_annotations.json` corrispondente
4. Opzionalmente attivare **Normalizza poligoni**
5. Premere **Avvia Elaborazione**

## Pipeline di elaborazione

La pipeline si articola in 5 fasi sequenziali:

### Fase 1 — Rilevamento automatico

Per ogni frame del video:

1. **Correzione fish-eye** (se configurata con parametri di calibrazione)
2. **Miglioramento qualita** tramite CLAHE e sharpening
3. **Motion detection** per identificare le zone con movimento (opzionale)
4. **Interpolazione sub-frame** per video a basso framerate
5. **Inferenza multi-scala** su 4 scale (1.0x, 1.5x, 2.0x, 2.5x) con TTA
6. **Sliding window** su griglia 3x3 con 30% overlap
7. **Non-Maximum Suppression** per eliminare i duplicati
8. **Tracking ByteTrack** per assegnare ID persistenti alle persone
9. **Temporal smoothing** con media mobile su 15 frame
10. **Calcolo intensita adattiva** in base all'altezza della figura

Se viene fornito `--review`, questa fase carica il JSON esistente invece di eseguire la detection.

### Fase 2 — Revisione manuale

Apre l'interfaccia interattiva OpenCV per correggere le detection. Saltata in modalita `auto` o con `--normalize`.

### Fase 3 — Rendering

Applica l'oscuramento (pixelation o blur) su tutti i poligoni per ogni frame e scrive il video di output e il video debug.

### Fase 4 — Verifica post-rendering

Esegue un secondo passaggio YOLO sul video oscurato per rilevare eventuali persone ancora visibili. Segnala i frame problematici nella console.

### Fase 5 — Post-processing

Reintegra l'audio dal video originale tramite ffmpeg, salva il report CSV e il JSON delle annotazioni.

## Interfaccia di revisione manuale

L'interfaccia si apre automaticamente in modalita `manual` dopo la Fase 1.

### Controlli

| Tasto                   | Azione                                 |
| ----------------------- | -------------------------------------- |
| Freccia destra / Spazio | Frame successivo                       |
| Freccia sinistra        | Frame precedente                       |
| Click sinistro          | Aggiunge un punto al poligono in corso |
| Invio                   | Chiude il poligono (minimo 3 punti)    |
| Ctrl+Z                  | Annulla l'ultimo punto inserito        |
| D                       | Attiva/disattiva modalita elimina      |
| Esc                     | Annulla il poligono in corso           |
| Q                       | Conferma le modifiche e prosegue       |

### Visualizzazione

- **Verde**: poligoni rilevati automaticamente
- **Arancione**: poligoni aggiunti manualmente
- **Ciano**: poligono in fase di disegno
- Trasparenza di riempimento: 35%

## Calibrazione fish-eye

Per telecamere con distorsione grandangolare, è possibile calibrare i parametri della camera usando lo script dedicato:

```bash
python person_anonymizer/camera_calibration.py --images ./foto_scacchiera/ --output calibration.npz
```

| Parametro      | Default           | Descrizione                                           |
| -------------- | ----------------- | ----------------------------------------------------- |
| `--images`     | *obbligatorio*    | Cartella con le foto della scacchiera di calibrazione |
| `--output`     | `calibration.npz` | File di output con i parametri calcolati              |
| `--board-cols` | 9                 | Numero di angoli interni della scacchiera (colonne)   |
| `--board-rows` | 6                 | Numero di angoli interni della scacchiera (righe)     |

I parametri ottenuti (`camera_matrix` e `dist_coefficients`) vanno inseriti nelle costanti `CAMERA_MATRIX` e `DIST_COEFFICIENTS` in `person_anonymizer.py`.

## File di output

Per un video di input chiamato `video.mp4`, vengono generati:

| File                     | Descrizione                                                    |
| ------------------------ | -------------------------------------------------------------- |
| `video_anonymized.mp4`   | Video con le persone oscurate e audio reintegrato              |
| `video_debug.mp4`        | Video con i poligoni di oscuramento visibili (verde/arancione) |
| `video_report.csv`       | Statistiche di rilevamento per ogni frame                      |
| `video_annotations.json` | Poligoni e metadati, riutilizzabile con `--review`             |

### Struttura del JSON annotazioni

```json
{
  "video": "video.mp4",
  "total_frames": 1500,
  "mode": "manual",
  "generated": "2026-03-08T14:10:05",
  "frames": {
    "0": {
      "auto": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
      "manual": []
    },
    "1": {
      "auto": [],
      "manual": [[[x1, y1], [x2, y2], [x3, y3]]]
    }
  }
}
```

### Colonne del report CSV

| Colonna               | Descrizione                                |
| --------------------- | ------------------------------------------ |
| `frame_number`        | Indice del frame                           |
| `persons_detected`    | Numero di persone rilevate                 |
| `avg_confidence`      | Confidenza media delle detection           |
| `min_confidence`      | Confidenza minima                          |
| `max_confidence`      | Confidenza massima                         |
| `motion_zones`        | Numero di zone con movimento rilevato      |
| `sliding_window_hits` | Patch della griglia con detection positive |
| `multiscale_hits`     | Scale con detection positive               |
| `post_check_alerts`   | Rilevamenti residui dopo l'oscuramento     |

## Configurazione avanzata

I parametri di configurazione si trovano nella sezione `CONFIGURAZIONE` all'inizio di `person_anonymizer.py`. Le impostazioni principali:

### Rilevamento

| Parametro                | Default              | Descrizione                                  |
| ------------------------ | -------------------- | -------------------------------------------- |
| `YOLO_MODEL`             | `yolov8x.pt`         | Modello YOLO (x = extra large, piu preciso)  |
| `DETECTION_CONFIDENCE`   | 0.35                 | Soglia minima di confidenza per le detection |
| `NMS_IOU_THRESHOLD`      | 0.45                 | Soglia IoU per la Non-Maximum Suppression    |
| `INFERENCE_SCALES`       | [1.0, 1.5, 2.0, 2.5] | Scale di inferenza multi-scala               |
| `SLIDING_WINDOW_GRID`    | 3                    | Dimensione griglia sliding window (3x3)      |
| `SLIDING_WINDOW_OVERLAP` | 0.3                  | Sovrapposizione tra patch (30%)              |

### Oscuramento

| Parametro                   | Default      | Descrizione                                   |
| --------------------------- | ------------ | --------------------------------------------- |
| `ANONYMIZATION_METHOD`      | `pixelation` | Metodo di oscuramento                         |
| `ANONYMIZATION_INTENSITY`   | 10           | Intensita base (blocco pixel o kernel blur)   |
| `PERSON_PADDING`            | 15           | Padding in pixel intorno alla detection       |
| `ADAPTIVE_REFERENCE_HEIGHT` | 80           | Altezza di riferimento per intensita adattiva |

### Tracking e smoothing

| Parametro               | Default | Descrizione                                          |
| ----------------------- | ------- | ---------------------------------------------------- |
| `TRACK_MAX_AGE`         | 45      | Frame massimi per mantenere un track senza detection |
| `TRACK_MATCH_THRESH`    | 0.6     | Soglia di matching per il tracker                    |
| `SMOOTHING_WINDOW_SIZE` | 15      | Ampiezza della finestra di media mobile              |

### Verifica post-rendering

| Parametro                      | Default | Descrizione                                          |
| ------------------------------ | ------- | ---------------------------------------------------- |
| `POST_RENDER_CHECK_CONFIDENCE` | 0.45    | Soglia di confidenza per segnalare detection residue |

## Formati video supportati

`.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`

## Struttura del progetto

```
video-anonimizer/
├── person_anonymizer/
│   ├── person_anonymizer.py      # Pipeline principale
│   ├── manual_reviewer.py        # Interfaccia revisione manuale
│   ├── camera_calibration.py     # Calibrazione camera fish-eye
│   ├── requirements.txt          # Dipendenze Python
│   └── web/                      # Interfaccia web Flask
│       ├── app.py                # Server Flask e API REST
│       ├── pipeline_runner.py    # Esecutore pipeline in thread separato
│       ├── sse_manager.py        # Distribuzione eventi Server-Sent Events
│       ├── templates/
│       │   └── index.html        # Pagina principale
│       ├── static/
│       │   ├── css/style.css     # Tema dark con glassmorphism
│       │   └── js/app.js         # Logica frontend
│       ├── uploads/              # Video caricati (temporanei)
│       └── outputs/              # Risultati per job
├── doc-progetto/
│   └── specifiche_tecniche_person_anonymizer_v6.md
├── docs/
│   └── REPORT_ATTIVITA.md
├── input/                        # Video di input (non versionato)
└── README.md
```

## Limitazioni note

- Ottimizzato per **telecamere fisse**; non gestisce movimenti di camera (pan/tilt/zoom)
- Progettato per **video brevi** (< 5 minuti); video lunghi richiedono tempo di elaborazione significativo
- Il modello `yolov8x.pt` richiede circa 130 MB di spazio disco e memoria GPU sufficiente
- La correzione fish-eye necessita di calibrazione manuale con scacchiera
- Senza GPU CUDA l'elaborazione e sensibilmente piu lenta
- Il secondo passaggio di verifica (Fase 4) puo produrre falsi positivi su aree gia oscurate
- L'interfaccia web opera in modalità `auto` per default; la modalità `manuale` richiede l'interfaccia OpenCV nativa (non disponibile nel browser)

# 
