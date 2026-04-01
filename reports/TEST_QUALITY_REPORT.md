# Test Quality Report — Person Anonymizer

**Data:** 2026-04-01
**Analisi su:** test suite in `tests/` vs sorgenti in `person_anonymizer/`

---

## 1. Inventario Test Esistenti

### test_config.py (19 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestPipelineConfigInvariants` | 12 test | Invarianti di default: scale non vuote, confidence nel range, NMS threshold, CLAHE grid, ghost expansion, smoothing alpha, max refinement, operation mode, anonymization method |
| `TestPipelineConfigCustomValues` | 7 test | Override parametri custom: mode, confidence, method, padding, scales, motion detection, parametri multipli |
| `TestSupportedExtensions` | 5 test | Verifica set estensioni (.mp4, .mov, .avi, .mkv, .webm) |
| `TestVersion` | 2 test | VERSION e stringa non vuota |

### test_config_validation.py (20 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestValidateConfigRejectsAttacks` | 13 test | Input malevoli: refinement DoS, path traversal YOLO, stringa come confidence, grid negativa/zero, confidence fuori range, bool come int/stringa, inference scales malevole, augmentation invalide, scales vuote |
| `TestValidateConfigAcceptsValid` | 6 test | Input legittimi: config vuota, confidence valida, modello noto, booleano, config completa, valori boundary |
| `TestBuildConfigIntegration` | 5 test | Integrazione _build_config: raises ValueError per input invalidi, ritorna PipelineConfig per input validi |

### test_anonymization.py (17 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestComputeAdaptiveIntensity` | 6 test | Intensita adattiva: reference height, minimo garantito, altezza doppia, reference zero, minimo >= 3, monotonia |
| `TestBoxToPolygon` | 7 test | Box a poligono: no padding, con padding, clamping bordi, edge padding multiplier, tipo ritorno, zero padding |
| `TestPolygonToBbox` | 6 test | Poligono a bbox: triangolo, rettangolo, punto singolo, formato, pentagono, round-trip |

### test_detection.py (14 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestComputeIouBoxes` | 6 test | IoU: identiche, no overlap, overlap parziale, contenimento, bordi a contatto, simmetria |
| `TestApplyNms` | 4 test (skipif cv2) | NMS: lista vuota, singolo box, overlap, non-overlap |
| `TestGetWindowPatches` | 6 test | Sliding window patches: conteggio 3x3/2x2, formato, copertura top-left, limiti frame, overlap |
| `TestPatchIntersectsMotion` | 5 test | Intersezione patch-motion: true, false, bordi, regioni multiple, vuote, contenimento |

### test_pipeline_errors.py (5 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestPipelineExceptions` | 5 test | Eccezioni: PipelineError e Exception, PipelineInputError e PipelineError, messaggio, cattura |

### test_postprocessing.py (22 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestRectsOverlap` | 6 test | Overlap rettangoli: sovrapposti, separati, bordi, contenimento, simmetria |
| `TestMergeRects` | 4 test | Merge rettangoli: due sovrapposti, separati, identici, contenimento |
| `TestMergeOverlappingRects` | 8 test | Merge iterativo: no overlap, tutti sovrapposti, catena transitiva, vuoto, singolo, contenimento, bbox corretto, due gruppi |
| `TestNormalizeAnnotations` | 5 test | Normalizzazione: annotazioni vuote, frame senza poligoni, frame multipli vuoti, struttura chiavi, intensita disabilitata |
| `TestFilterArtifactDetections` | 5 test | Filtro artefatti: vuoto, tutti artefatti, genuino, mix, nessuna annotazione |

### test_rendering.py (7 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestComputeReviewStats` | 5 test | Review stats: nessuna modifica, poligoni aggiunti, rimossi, misti, vuoto |
| `TestRenderVideo` | 3 test | Rendering: produce output, applica oscuramento, output debug |

### test_tracking.py (11 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestSmooth` | 3 test | EMA: nuovo track, track esistente, reset ghost countdown |
| `TestGetGhostBoxes` | 5 test | Ghost boxes: no ghost iniziali, ghost dopo stale, countdown decrementa, expansion, rimozione, persistenza |
| `TestClearStale` | 2 test | Clear stale: marca inattivi per ghost, attivi non marcati |

### test_web.py (29 test)
| Classe | Test | Copertura |
|--------|------|-----------|
| `TestValidateJobId` | 5 test | Validazione job_id: valido, troppo corto, uppercase, caratteri speciali, vuoto |
| `TestSecurityHeaders` | 4 test | Header sicurezza: presenti, X-Content-Type-Options, X-Frame-Options, CSP no unsafe-inline |
| `TestStartPipeline` | 4 test | Endpoint /api/start: payload mancante, job_id mancante, job_id invalido, path traversal |
| `TestUpload` | 3 test | Endpoint /api/upload: no file, formato non supportato, filename vuoto |
| `TestConfigDefaults` | 4 test | Endpoint /api/config/defaults: 200 JSON, campi pipeline, confidence range, tuple serializzate |
| `TestDownload` | 3 test | Endpoint /api/download: job_id invalido, job inesistente, tipo invalido |
| `TestStartPipelineSecurity` | 2 test | Sicurezza /api/start: video_filename mancante, review_json traversal |
| `TestErrorHandlers` | 1 test | 404 su route sconosciuta |
| `TestUploadSecurity` | 1 test | Upload non espone path assoluti |
| `TestAnnotationValidation` | 8 test | Validazione annotazioni: payload valido, non dict, auto non lista, poligono < 3 punti, punto non coppia, coordinate non numeriche, coordinate float, endpoint 404 senza review |
| `TestSSESubscriberCap` | 2 test | Cap subscriber SSE: raise dopo limite, liberazione slot |
| `TestSecurityHeadersNew` | 2 test | Header COOP/CORP |
| `TestConfigDefaultsFiltered` | 1 test | Esclusione campi interni |

**Totale: ~144 test in 8 file**

---

## 2. Gap di Copertura (Funzioni/Classi Non Testate)

### CRITICAL — Nessun test

| Modulo | Funzione/Classe | Motivo criticita |
|--------|----------------|------------------|
| `person_anonymizer.py` | `run_pipeline()` | Funzione principale della pipeline. Zero test di integrazione anche parziale. |
| `person_anonymizer.py` | `_run_detection_loop()` | Core loop di detection. Nessun test nemmeno con mock del modello YOLO. |
| `person_anonymizer.py` | `_process_single_frame()` | Orchestrazione detection+tracking+smoothing per frame singolo. |
| `person_anonymizer.py` | `_run_refinement_loop()` | Loop di auto-refinement iterativo. |
| `person_anonymizer.py` | `_run_manual_review()` | Bridge tra pipeline e revisione manuale (web/CLI). |
| `person_anonymizer.py` | `_save_outputs()` | Salvataggio CSV, JSON, encoding H.264. |
| `person_anonymizer.py` | `_load_annotations_from_json()` | Caricamento JSON annotazioni. |
| `person_anonymizer.py` | `_init_frame_processors()` | Inizializzazione processori (richiede cv2/YOLO). |
| `person_anonymizer.py` | `parse_args()` | Parser CLI. |
| `person_anonymizer.py` | `OutputPaths`, `VideoMeta`, `PipelineResult`, `FrameProcessors` | Dataclass di supporto (non critico, ma nessun test). |
| `anonymization.py` | `obscure_polygon()` | Funzione core di oscuramento. Richiede cv2 ma testabile con frame sintetici. |
| `anonymization.py` | `draw_debug_polygons()` | Debug visivo. Testabile con frame sintetici. |
| `detection.py` | `run_sliding_window()` | Richiede modello YOLO. |
| `detection.py` | `detect_and_rescale()` | Richiede modello YOLO. |
| `detection.py` | `run_multiscale_inference()` | Richiede modello YOLO. |
| `detection.py` | `run_full_detection()` | Orchestrazione detection. Richiede YOLO. |
| `tracking.py` | `create_tracker()` | Crea BYTETracker. Richiede ultralytics. |
| `tracking.py` | `update_tracker()` | Aggiorna tracker. Richiede ultralytics. |
| `preprocessing.py` | `build_undistortion_maps()` | Calibrazione camera. |
| `preprocessing.py` | `undistort_frame()` | Undistortion frame. |
| `preprocessing.py` | `enhance_frame()` | CLAHE enhancement. Testabile con frame sintetici. |
| `preprocessing.py` | `MotionDetector` | Frame differencing. Testabile con frame sintetici. |
| `preprocessing.py` | `interpolate_frames()` | Interpolazione sub-frame. Testabile con numpy. |
| `preprocessing.py` | `should_interpolate()` | Funzione pura, trivialmente testabile. |
| `postprocessing.py` | `encode_with_audio()` | Encoding ffmpeg. |
| `postprocessing.py` | `encode_without_audio()` | Encoding ffmpeg. |
| `postprocessing.py` | `run_post_render_check()` | Secondo passaggio YOLO. Richiede modello. |
| `camera_calibration.py` | `find_chessboard_corners()` | Calibrazione. |
| `camera_calibration.py` | `calibrate_camera()` | Calibrazione. |
| `camera_calibration.py` | `main()` | CLI calibrazione. |
| `manual_reviewer.py` | `ManualReviewer` (intera classe) | UI interattiva OpenCV. |
| `web/pipeline_runner.py` | `PipelineRunner` | Thread wrapper pipeline. |
| `web/pipeline_runner.py` | `TqdmCapture` | Monkey-patch tqdm per SSE. |
| `web/pipeline_runner.py` | `StdoutCapture` | Cattura stdout per SSE. |
| `web/review_state.py` | `ReviewState` | Stato condiviso thread-safe per review web. |

### MAJOR — Parzialmente testati

| Modulo | Funzione | Lacuna |
|--------|----------|--------|
| `postprocessing.py` | `normalize_annotations()` | Testato solo con poligoni vuoti. Manca test con poligoni reali (richiede cv2.boundingRect). |
| `rendering.py` | `render_video()` | Testato solo con pixelation. Manca test con blur, fisheye, stop_event, frame corrotti. |

---

## 3. Valutazione Qualita dei Test Esistenti

### Punti di Forza

- **Pattern AAA rispettato**: tutti i test seguono Arrange/Act/Assert con commenti espliciti
- **Naming descrittivo**: i nomi dei test indicano chiaramente lo scenario (es. `test_rejects_yolo_model_path_traversal`)
- **Isolamento**: nessun test dipende da ordine di esecuzione o stato condiviso
- **Edge case ben coperti**: bordi a contatto (IoU=0), liste vuote, punti singoli, valori boundary
- **Test negativi (error cases)**: buona copertura di input malevoli nella validazione config e web
- **Test di sicurezza**: path traversal, header HTTP, input sanitization
- **Skipif appropriato**: `@pytest.mark.skipif(not CV2_AVAILABLE)` per test che richiedono cv2
- **Frame sintetici**: `test_rendering.py` crea video sintetici con numpy, evitando file di test esterni
- **Mock minimi**: quasi nessun mock eccessivo; i test operano su funzioni pure

### Problemi Identificati

| ID | Severita | Problema | File |
|----|----------|----------|------|
| Q-01 | Minor | `test_rendering.py` importa `cv2` al top level senza guard. Se cv2 non e disponibile, l'intero file fallisce con ImportError anziche skip. | `test_rendering.py` |
| Q-02 | Minor | `test_config.py` testa solo valori di default e override semplici. Non testa combinazioni invalide (es. `anonymization_method="invalid"`). La validazione e in `pipeline_runner.py`, non nel dataclass. | `test_config.py` |
| Q-03 | Minor | `test_anonymization.py:TestComputeAdaptiveIntensity` calcola manualmente i risultati attesi nei commenti, rendendoli fragili se la formula cambia. Meglio usare property-based testing (monotonia, minimo). Alcuni test gia lo fanno (es. `test_intensity_increases_with_height`). | `test_anonymization.py` |
| Q-04 | Minor | `test_web.py:TestStartPipelineSecurity.test_start_review_json_traversal_blocked` accetta 429 come risultato valido, rendendo il test meno determinante. Il rate limiter dovrebbe essere disabilitato nei test (e lo e nel fixture, ma potrebbe non funzionare per tutte le route). | `test_web.py` |
| Q-05 | Minor | `conftest.py` manipola `sys.path` globalmente. Meglio usare un `pyproject.toml` con `[tool.pytest.ini_options] pythonpath`. | `conftest.py` |

---

## 4. Suggerimenti per Nuovi Test

### Critical

| ID | Test Proposto | Motivo | File Target |
|----|--------------|--------|-------------|
| N-01 | **Test `obscure_polygon` con frame sintetico** — Verifica che pixelation e blur alterino effettivamente i pixel dentro il poligono e non quelli fuori. | Funzione core dell'anonimizzazione, zero test. | `test_anonymization.py` |
| N-02 | **Test `_load_annotations_from_json`** — Verifica caricamento JSON valido, JSON malformato, file mancante, JSON con struttura inattesa. | Input utente non validato, potenziale crash silenzioso. | `test_pipeline_errors.py` o nuovo `test_pipeline.py` |
| N-03 | **Test `should_interpolate` e `interpolate_frames`** — Funzioni pure, triviali da testare: `should_interpolate(10, 15)` -> True, `interpolate_frames` con due frame numpy. | Funzioni pure non testate nel modulo preprocessing. | nuovo `test_preprocessing.py` |
| N-04 | **Test `enhance_frame`** — Frame scuro (mean < threshold) deve essere modificato, frame luminoso no. | Logica condizionale non coperta, testabile con frame sintetici. | nuovo `test_preprocessing.py` |
| N-05 | **Test `MotionDetector`** — Primo frame ritorna None, due frame identici ritornano lista vuota, due frame diversi ritornano regioni non vuote. | Classe stateful con logica critica, testabile con frame sintetici. | nuovo `test_preprocessing.py` |

### Major

| ID | Test Proposto | Motivo | File Target |
|----|--------------|--------|-------------|
| N-06 | **Test `normalize_annotations` con poligoni reali** — Frame con poligoni sovrapposti che vengono mergiati; verifica conteggio before/after. | Attualmente testato solo con input vuoti. | `test_postprocessing.py` |
| N-07 | **Test `render_video` con metodo blur** — Verifica che anche blur produca output diverso dall'input. | Solo pixelation e testato. | `test_rendering.py` |
| N-08 | **Test `render_video` con stop_event** — Passa un Event gia settato, verifica che il rendering si interrompa. | Funzionalita di interruzione non testata. | `test_rendering.py` |
| N-09 | **Test `StdoutCapture._sanitize_message`** — Verifica che path assoluti vengano rimossi dai messaggi. Funzione pura, @classmethod. | Prevenzione leak di informazioni sensibili. | `test_web.py` o nuovo `test_pipeline_runner.py` |
| N-10 | **Test `validate_config_params` con campi sconosciuti** — Un campo non in `_CONFIG_VALIDATORS` ne in `_BOOL_FIELDS` viene ignorato silenziosamente. Verificare se e intenzionale. | Potenziale typo nei parametri non rilevato. | `test_config_validation.py` |
| N-11 | **Test `PipelineRunner.start` con pipeline gia in esecuzione** — Verifica che ritorni `(False, "Una pipeline e gia in esecuzione")`. | Concorrenza non testata. | nuovo `test_pipeline_runner.py` |
| N-12 | **Test integrazione `run_pipeline` con mock YOLO** — Mock del modello YOLO che ritorna detection fisse, verifica che l'intero flusso produca output. | La funzione principale non ha alcun test. | nuovo `test_pipeline_integration.py` |

### Minor

| ID | Test Proposto | Motivo | File Target |
|----|--------------|--------|-------------|
| N-13 | **Test `parse_args` con argomenti validi e invalidi** — Verifica parsing CLI. | Nessun test per CLI. | nuovo `test_cli.py` |
| N-14 | **Test `ReviewState` lifecycle** — setup -> get_metadata -> update_annotations -> complete -> wait_for_completion. | Stato condiviso thread-safe non testato. | nuovo `test_review_state.py` |
| N-15 | **Test `SSEManager.emit` e `SSEManager.close`** — Verifica che emit metta eventi nelle code e close metta None. | Solo subscribe/unsubscribe/cap testati. | `test_web.py` |
| N-16 | **Test `compute_review_stats` con poligoni modificati (non solo aggiunti/rimossi)** — Verifica che un poligono spostato venga contato come 1 rimosso + 1 aggiunto. | Scenario realistico non coperto. | `test_rendering.py` |
| N-17 | **Test `box_to_polygon` con coordinate negative** — Verifica clamping a 0. | Edge case non testato. | `test_anonymization.py` |
| N-18 | **Test `compute_iou_boxes` con box di area zero** — Un box dove x1==x2 o y1==y2. Verifica che non causi divisione per zero. | Robustezza edge case. | `test_detection.py` |

---

## 5. Riepilogo

| Metrica | Valore |
|---------|--------|
| File di test | 8 (+1 conftest) |
| Test totali | ~144 |
| Moduli sorgente | 10 (+ 4 web) |
| Funzioni/classi con test | ~20 |
| Funzioni/classi senza test | ~35 |
| Copertura logica stimata | ~35-40% |
| Qualita test esistenti | Buona (AAA, naming, isolamento, edge case) |
| Gap principale | `person_anonymizer.py` (pipeline core), `preprocessing.py`, `manual_reviewer.py` — zero test |

### Priorita di intervento

1. **Critical**: Test per `obscure_polygon`, funzioni pure di `preprocessing.py` (`should_interpolate`, `enhance_frame`, `MotionDetector`, `interpolate_frames`), `_load_annotations_from_json`
2. **Major**: Test integrazione pipeline con mock YOLO, `normalize_annotations` con poligoni reali, `render_video` blur/stop_event, `StdoutCapture._sanitize_message`
3. **Minor**: CLI parsing, `ReviewState`, `SSEManager` emit/close, edge case aggiuntivi
