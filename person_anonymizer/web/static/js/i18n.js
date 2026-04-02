/**
 * i18n — Sistema di internazionalizzazione client-side (IT/EN).
 *
 * Uso:
 *   t("key")             → restituisce la stringa tradotta nella lingua corrente
 *   setLanguage("en")    → cambia lingua, aggiorna DOM, salva in localStorage
 *   applyLanguage()      → ri-applica traduzioni al DOM (dopo render dinamici)
 *   getCurrentLang()     → "it" | "en"
 */
const I18n = (function () {
    "use strict";

    const STORAGE_KEY = "pa_lang";
    let currentLang = localStorage.getItem(STORAGE_KEY) || "it";

    // =========================================================================
    // Dizionario traduzioni
    // =========================================================================
    const translations = {
        // --- Header ---
        "status.ready": { it: "Pronto", en: "Ready" },
        "status.processing": { it: "Elaborazione...", en: "Processing..." },
        "status.completed": { it: "Completato", en: "Completed" },
        "status.error": { it: "Errore", en: "Error" },

        // --- Upload ---
        "upload.title": { it: "Video di Input", en: "Input Video" },
        "upload.dropzone.aria": {
            it: "Trascina un video o clicca per selezionare",
            en: "Drag a video or click to select",
        },
        "upload.dropzone.text": {
            it: "Trascina un video qui",
            en: "Drag a video here",
        },
        "upload.dropzone.hint": {
            it: "oppure clicca per selezionare",
            en: "or click to select",
        },
        "upload.remove.title": { it: "Rimuovi", en: "Remove" },
        "upload.remove.aria": { it: "Rimuovi file", en: "Remove file" },
        "upload.log": { it: "Upload: {0} ({1} MB)...", en: "Upload: {0} ({1} MB)..." },
        "upload.error": { it: "Errore upload: {0}", en: "Upload error: {0}" },
        "upload.error.invalid": { it: "Errore upload: risposta non valida", en: "Upload error: invalid response" },
        "upload.error.connection": { it: "Errore di connessione durante l'upload", en: "Connection error during upload" },
        "upload.error.connection.log": { it: "Errore upload: connessione fallita", en: "Upload error: connection failed" },
        "upload.success": { it: "Video caricato: {0}", en: "Video uploaded: {0}" },
        "upload.toast.error": { it: "Errore durante l'upload", en: "Error during upload" },

        // --- Impostazioni Base ---
        "settings.base.title": { it: "Impostazioni Base", en: "Basic Settings" },
        "settings.mode.label": { it: "Modalit\u00e0 operativa", en: "Operation mode" },
        "settings.mode.help": {
            it: "Automatica: elabora tutto senza intervento. Manuale: permette di rivedere e correggere i rilevamenti nel browser prima del rendering finale.",
            en: "Automatic: processes everything without intervention. Manual: allows reviewing and correcting detections in the browser before final rendering.",
        },
        "settings.mode.aria": { it: "Aiuto: Modalit\u00e0 operativa", en: "Help: Operation mode" },
        "settings.mode.auto": { it: "Automatica", en: "Automatic" },
        "settings.mode.manual": { it: "Manuale (revisione nel browser)", en: "Manual (browser review)" },
        "settings.method.label": { it: "Metodo di oscuramento", en: "Anonymization method" },
        "settings.method.help": {
            it: "Pixelation: rende irriconoscibile la persona con effetto mosaico. Blur: sfoca l'area con effetto gaussiano.",
            en: "Pixelation: makes the person unrecognizable with a mosaic effect. Blur: blurs the area with a Gaussian effect.",
        },
        "settings.method.aria": { it: "Aiuto: Metodo di oscuramento", en: "Help: Anonymization method" },
        "settings.intensity.label": { it: "Intensit\u00e0 oscuramento:", en: "Anonymization intensity:" },
        "settings.intensity.help": {
            it: "Valore pi\u00f9 alto = oscuramento pi\u00f9 forte. Controlla la dimensione dei pixel (pixelation) o il raggio di sfocatura (blur).",
            en: "Higher value = stronger anonymization. Controls pixel size (pixelation) or blur radius (blur).",
        },
        "settings.intensity.aria": { it: "Aiuto: Intensit\u00e0 oscuramento", en: "Help: Anonymization intensity" },
        "settings.padding.label": { it: "Padding persona (px):", en: "Person padding (px):" },
        "settings.padding.help": {
            it: "Pixel aggiuntivi intorno alla persona rilevata. Valori pi\u00f9 alti oscurano un'area pi\u00f9 ampia attorno alla sagoma per maggiore sicurezza.",
            en: "Extra pixels around the detected person. Higher values anonymize a larger area around the silhouette for greater safety.",
        },
        "settings.padding.aria": { it: "Aiuto: Padding persona", en: "Help: Person padding" },

        // --- Rilevamento ---
        "detection.title": { it: "Rilevamento", en: "Detection" },
        "detection.model.label": { it: "Modello YOLO", en: "YOLO Model" },
        "detection.model.help": {
            it: "YOLOv8x: pi\u00f9 accurato ma pi\u00f9 lento, consigliato per persone piccole o distanti. YOLOv8n: molto pi\u00f9 veloce, adatto per persone ben visibili.",
            en: "YOLOv8x: more accurate but slower, recommended for small or distant people. YOLOv8n: much faster, suitable for clearly visible people.",
        },
        "detection.model.aria": { it: "Aiuto: Modello YOLO", en: "Help: YOLO Model" },
        "detection.model.opt.x": { it: "YOLOv8x (accurato, lento)", en: "YOLOv8x (accurate, slow)" },
        "detection.model.opt.n": { it: "YOLOv8n (veloce, meno accurato)", en: "YOLOv8n (fast, less accurate)" },
        "detection.confidence.label": { it: "Confidenza rilevamento:", en: "Detection confidence:" },
        "detection.confidence.help": {
            it: "Soglia minima di confidenza per accettare un rilevamento. Pi\u00f9 bassa = pi\u00f9 rilevamenti (anche falsi positivi), pi\u00f9 alta = solo rilevamenti certi.",
            en: "Minimum confidence threshold to accept a detection. Lower = more detections (including false positives), higher = only certain detections.",
        },
        "detection.confidence.aria": { it: "Aiuto: Confidenza rilevamento", en: "Help: Detection confidence" },
        "detection.nms.label": { it: "NMS IoU threshold:", en: "NMS IoU threshold:" },
        "detection.nms.help": {
            it: "Soglia per eliminare rilevamenti duplicati sovrapposti. Pi\u00f9 bassa = rimozione duplicati pi\u00f9 aggressiva, pi\u00f9 alta = mantiene rilevamenti vicini.",
            en: "Threshold to remove overlapping duplicate detections. Lower = more aggressive duplicate removal, higher = keeps nearby detections.",
        },
        "detection.nms.aria": { it: "Aiuto: NMS IoU threshold", en: "Help: NMS IoU threshold" },

        // --- Funzionalit\u00e0 Avanzate ---
        "advanced.title": { it: "Funzionalit\u00e0 Avanzate", en: "Advanced Features" },
        "advanced.sliding.label": { it: "Sliding Window", en: "Sliding Window" },
        "advanced.sliding.help": {
            it: "Analizza il frame in sotto-regioni sovrapposte per rilevare persone piccole o ai bordi che il modello potrebbe non vedere nell'immagine completa.",
            en: "Analyzes the frame in overlapping sub-regions to detect small or edge people that the model might miss in the full image.",
        },
        "advanced.sliding.aria": { it: "Aiuto: Sliding Window", en: "Help: Sliding Window" },
        "advanced.tracking.label": { it: "ByteTrack Tracking", en: "ByteTrack Tracking" },
        "advanced.tracking.help": {
            it: "Associa le persone tra frame consecutivi per mantenere un'identit\u00e0 consistente e ridurre falsi negativi temporanei.",
            en: "Associates people across consecutive frames to maintain consistent identity and reduce temporary false negatives.",
        },
        "advanced.tracking.aria": { it: "Aiuto: ByteTrack Tracking", en: "Help: ByteTrack Tracking" },
        "advanced.smoothing.label": { it: "Temporal Smoothing (EMA)", en: "Temporal Smoothing (EMA)" },
        "advanced.smoothing.help": {
            it: "Smussa le coordinate dei bounding box nel tempo con media mobile esponenziale, riducendo il tremolio dell'oscuramento.",
            en: "Smooths bounding box coordinates over time with exponential moving average, reducing anonymization jitter.",
        },
        "advanced.smoothing.aria": { it: "Aiuto: Temporal Smoothing", en: "Help: Temporal Smoothing" },
        "advanced.adaptive.label": { it: "Intensit\u00e0 Adattiva", en: "Adaptive Intensity" },
        "advanced.adaptive.help": {
            it: "Regola automaticamente l'intensit\u00e0 in base alla dimensione della persona: pi\u00f9 piccola = pi\u00f9 intensa per garantire l'anonimizzazione.",
            en: "Automatically adjusts intensity based on person size: smaller = more intense to ensure anonymization.",
        },
        "advanced.adaptive.aria": { it: "Aiuto: Intensit\u00e0 Adattiva", en: "Help: Adaptive Intensity" },
        "advanced.postcheck.label": { it: "Verifica Post-Rendering", en: "Post-Render Check" },
        "advanced.postcheck.help": {
            it: "Dopo il rendering, rianalizza il video anonimizzato per verificare che non siano rimaste persone visibili. Aggiunge un passaggio di sicurezza.",
            en: "After rendering, re-analyzes the anonymized video to verify no visible people remain. Adds a safety step.",
        },
        "advanced.postcheck.aria": { it: "Aiuto: Verifica Post-Rendering", en: "Help: Post-Render Check" },
        "advanced.fisheye.label": { it: "Correzione Fish-eye", en: "Fish-eye Correction" },
        "advanced.fisheye.help": {
            it: "Compensa la distorsione delle telecamere grandangolari, migliorando l'accuratezza del rilevamento ai bordi dell'immagine.",
            en: "Compensates for wide-angle camera distortion, improving detection accuracy at image edges.",
        },
        "advanced.fisheye.aria": { it: "Aiuto: Correzione Fish-eye", en: "Help: Fish-eye Correction" },
        "advanced.motion.label": { it: "Motion Detection", en: "Motion Detection" },
        "advanced.motion.help": {
            it: "Analizza solo le aree con movimento rilevato, velocizzando l'elaborazione su scene con ampie zone statiche.",
            en: "Analyzes only areas with detected motion, speeding up processing on scenes with large static zones.",
        },
        "advanced.motion.aria": { it: "Aiuto: Motion Detection", en: "Help: Motion Detection" },
        "advanced.interpolation.label": { it: "Interpolazione Sub-frame", en: "Sub-frame Interpolation" },
        "advanced.interpolation.help": {
            it: "Genera oscuramenti intermedi tra i frame analizzati per video ad alto framerate, evitando discontinuit\u00e0 nell'oscuramento.",
            en: "Generates intermediate anonymizations between analyzed frames for high-framerate video, avoiding anonymization gaps.",
        },
        "advanced.interpolation.aria": { it: "Aiuto: Interpolazione Sub-frame", en: "Help: Sub-frame Interpolation" },
        "advanced.debug.label": { it: "Genera video debug", en: "Generate debug video" },
        "advanced.debug.help": {
            it: "Produce un video aggiuntivo con bounding box, ID di tracking e confidenze visibili per verificare la qualit\u00e0 del rilevamento.",
            en: "Produces an additional video with visible bounding boxes, tracking IDs, and confidences to verify detection quality.",
        },
        "advanced.debug.aria": { it: "Aiuto: Video debug", en: "Help: Debug video" },
        "advanced.report.label": { it: "Genera report CSV", en: "Generate CSV report" },
        "advanced.report.help": {
            it: "Crea un file CSV con statistiche dettagliate: confidenze per frame, numero di persone rilevate, aree coperte.",
            en: "Creates a CSV file with detailed statistics: per-frame confidences, number of detected people, covered areas.",
        },
        "advanced.report.aria": { it: "Aiuto: Report CSV", en: "Help: CSV report" },

        // --- Revisione JSON ---
        "review.json.title": { it: "Revisione da JSON", en: "Review from JSON" },
        "review.json.hint": {
            it: "Carica un file JSON di annotazioni esistente per riprendere l'elaborazione.",
            en: "Load an existing annotation JSON file to resume processing.",
        },
        "review.json.remove.title": { it: "Rimuovi", en: "Remove" },
        "review.json.remove.aria": { it: "Rimuovi file JSON", en: "Remove JSON file" },
        "review.json.normalize.label": { it: "Normalizza poligoni in rettangoli", en: "Normalize polygons to rectangles" },
        "review.json.normalize.help": {
            it: "Converte eventuali poligoni irregolari nelle annotazioni JSON in rettangoli regolari (bounding box) per un oscuramento uniforme.",
            en: "Converts any irregular polygons in the JSON annotations to regular rectangles (bounding boxes) for uniform anonymization.",
        },
        "review.json.normalize.aria": { it: "Aiuto: Normalizza poligoni", en: "Help: Normalize polygons" },

        // --- Azioni ---
        "action.start": { it: "Avvia Elaborazione", en: "Start Processing" },
        "action.stop": { it: "Interrompi", en: "Stop" },

        // --- Progresso ---
        "progress.title": { it: "Progresso", en: "Progress" },
        "progress.aria": { it: "Fasi di elaborazione", en: "Processing phases" },
        "progress.bar.aria": { it: "Progresso elaborazione", en: "Processing progress" },
        "phase.detection": { it: "Rilevamento", en: "Detection" },
        "phase.review": { it: "Revisione", en: "Review" },
        "phase.rendering": { it: "Rendering", en: "Rendering" },
        "phase.verification": { it: "Verifica", en: "Verification" },
        "phase.audio": { it: "Audio", en: "Audio" },

        // --- Revisione Manuale ---
        "review.title": { it: "Revisione Manuale", en: "Manual Review" },
        "review.frame": { it: "Frame {0} / {1}", en: "Frame {0} / {1}" },
        "review.polygons": { it: "{0} poligoni", en: "{0} polygons" },
        "review.prev": { it: "\u2190 Prec", en: "\u2190 Prev" },
        "review.prev.aria": { it: "Frame precedente", en: "Previous frame" },
        "review.next": { it: "Succ \u2192", en: "Next \u2192" },
        "review.next.aria": { it: "Frame successivo", en: "Next frame" },
        "review.help.click": { it: "aggiungi punto", en: "add point" },
        "review.help.enter": { it: "chiudi poligono", en: "close polygon" },
        "review.help.d": { it: "elimina", en: "delete" },
        "review.help.esc": { it: "annulla", en: "cancel" },
        "review.help.nav": { it: "naviga", en: "navigate" },
        "review.help.undo": { it: "undo", en: "undo" },
        "review.confirm": { it: "Conferma e Continua", en: "Confirm and Continue" },

        // --- Console ---
        "console.title": { it: "Console", en: "Console" },
        "console.clear.title": { it: "Pulisci console", en: "Clear console" },
        "console.clear.aria": { it: "Pulisci console", en: "Clear console" },
        "console.log.aria": { it: "Log di elaborazione", en: "Processing log" },
        "console.waiting": { it: "In attesa di avvio...", en: "Waiting to start..." },

        // --- Risultati ---
        "results.title": { it: "Risultati", en: "Results" },
        "results.download": { it: "Scarica", en: "Download" },

        // --- Pipeline messages ---
        "pipeline.starting": { it: "Avvio pipeline...", en: "Starting pipeline..." },
        "pipeline.started": { it: "Pipeline avviata", en: "Pipeline started" },
        "pipeline.completed": { it: "Pipeline completata con successo!", en: "Pipeline completed successfully!" },
        "pipeline.error.start": { it: "Errore avvio: {0}", en: "Start error: {0}" },
        "pipeline.error.toast": { it: "Errore di avvio pipeline", en: "Pipeline start error" },
        "pipeline.stop.log": { it: "Interruzione richiesta...", en: "Stop requested..." },
        "pipeline.stop.toast": { it: "Interruzione richiesta", en: "Stop requested" },
        "pipeline.review.ready": {
            it: "Revisione manuale pronta \u2014 usa l'editor qui sotto",
            en: "Manual review ready \u2014 use the editor below",
        },
        "pipeline.review.toast": { it: "Revisione manuale pronta", en: "Manual review ready" },
        "pipeline.error.generic": { it: "Errore: {0}", en: "Error: {0}" },
        "pipeline.progress.frames": { it: "{0} / {1} frame", en: "{0} / {1} frames" },
        "pipeline.progress.rate": { it: "{0} frame/s", en: "{0} frames/s" },

        // --- Upload JSON ---
        "upload.json.error": { it: "Errore upload JSON: {0}", en: "JSON upload error: {0}" },
        "upload.json.error.toast": { it: "Errore upload JSON", en: "JSON upload error" },
        "upload.json.success": { it: "JSON caricato: {0}", en: "JSON loaded: {0}" },

        // --- Toast ---
        "toast.close.aria": { it: "Chiudi", en: "Close" },

        // --- Language switcher ---
        "lang.toggle.aria": { it: "Cambia lingua", en: "Change language" },
    };

    // =========================================================================
    // API
    // =========================================================================

    /**
     * Restituisce la stringa tradotta, con sostituzione placeholder {0}, {1}...
     */
    function t(key) {
        const entry = translations[key];
        if (!entry) return key;
        const tmpl = entry[currentLang] || entry["it"] || key;
        // Supporta argomenti extra: t("key", val0, val1)
        const args = Array.prototype.slice.call(arguments, 1);
        if (args.length === 0) return tmpl;
        return tmpl.replace(/\{(\d+)\}/g, function (m, i) {
            return args[parseInt(i)] !== undefined ? args[parseInt(i)] : m;
        });
    }

    /**
     * Applica tutte le traduzioni al DOM corrente.
     */
    function applyLanguage() {
        document.documentElement.lang = currentLang;

        // data-i18n → textContent
        document.querySelectorAll("[data-i18n]").forEach(function (el) {
            var key = el.getAttribute("data-i18n");
            var entry = translations[key];
            if (entry) el.textContent = entry[currentLang] || entry["it"];
        });

        // data-i18n-help → data-help attribute
        document.querySelectorAll("[data-i18n-help]").forEach(function (el) {
            var key = el.getAttribute("data-i18n-help");
            var entry = translations[key];
            if (entry) el.setAttribute("data-help", entry[currentLang] || entry["it"]);
        });

        // data-i18n-aria → aria-label
        document.querySelectorAll("[data-i18n-aria]").forEach(function (el) {
            var key = el.getAttribute("data-i18n-aria");
            var entry = translations[key];
            if (entry) el.setAttribute("aria-label", entry[currentLang] || entry["it"]);
        });

        // data-i18n-title → title
        document.querySelectorAll("[data-i18n-title]").forEach(function (el) {
            var key = el.getAttribute("data-i18n-title");
            var entry = translations[key];
            if (entry) el.setAttribute("title", entry[currentLang] || entry["it"]);
        });

    }

    /**
     * Cambia lingua e aggiorna il DOM.
     */
    function setLanguage(lang) {
        if (lang !== "it" && lang !== "en") return;
        currentLang = lang;
        localStorage.setItem(STORAGE_KEY, lang);
        applyLanguage();
    }

    /**
     * Toggle tra IT e EN.
     */
    function toggleLanguage() {
        setLanguage(currentLang === "it" ? "en" : "it");
    }

    function getCurrentLang() {
        return currentLang;
    }

    return {
        t: t,
        applyLanguage: applyLanguage,
        setLanguage: setLanguage,
        toggleLanguage: toggleLanguage,
        getCurrentLang: getCurrentLang,
    };
})();
