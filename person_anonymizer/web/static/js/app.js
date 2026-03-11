/**
 * Person Anonymizer — Web GUI
 * Logica frontend: upload, config, SSE progress, download.
 */

(function () {
    "use strict";

    // === State ===
    let jobId = null;
    let videoPath = null;
    let jsonPath = null;
    let eventSource = null;
    let currentPhase = 0;

    // === DOM refs ===
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dropzone = $("#dropzone");
    const fileInput = $("#fileInput");
    const fileInfo = $("#fileInfo");
    const fileName = $("#fileName");
    const fileSize = $("#fileSize");
    const removeFile = $("#removeFile");

    const jsonFileInput = $("#jsonFileInput");
    const jsonFileInfo = $("#jsonFileInfo");
    const jsonFileName = $("#jsonFileName");
    const removeJsonFile = $("#removeJsonFile");

    const btnStart = $("#btnStart");
    const btnStop = $("#btnStop");

    const progressContainer = $("#progressContainer");
    const progressLabel = $("#progressLabel");
    const progressPercent = $("#progressPercent");
    const progressFill = $("#progressFill");
    const progressDetail = $("#progressDetail");
    const progressRate = $("#progressRate");

    const consoleEl = $("#console");
    const btnClearConsole = $("#btnClearConsole");

    const resultsCard = $("#resultsCard");
    const resultsEl = $("#results");

    // === Sezioni collassabili ===
    $$(".collapsible").forEach((el) => {
        el.addEventListener("click", () => {
            el.classList.toggle("collapsed");
            const target = document.getElementById(el.dataset.target);
            if (target) target.classList.toggle("collapsed-body");
        });
    });

    // === Slider value display ===
    const sliders = [
        ["anonymization_intensity", "val-intensity", null],
        ["person_padding", "val-padding", null],
        ["detection_confidence", "val-confidence", (v) => parseFloat(v).toFixed(2)],
        ["nms_iou_threshold", "val-nms", (v) => parseFloat(v).toFixed(2)],
    ];
    sliders.forEach(([id, displayId, fmt]) => {
        const input = document.getElementById(id);
        const display = document.getElementById(displayId);
        if (input && display) {
            input.addEventListener("input", () => {
                display.textContent = fmt ? fmt(input.value) : input.value;
            });
        }
    });

    // === Upload Video: drag & drop + click ===
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleVideoFile(fileInput.files[0]);
        }
    });

    removeFile.addEventListener("click", () => {
        jobId = null;
        videoPath = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        dropzone.classList.remove("hidden");
        updateStartButton();
    });

    function handleVideoFile(file) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        appendLog(`Upload: ${file.name} (${sizeMB} MB)...`, "info");

        const form = new FormData();
        form.append("video", file);

        fetch("/api/upload", { method: "POST", body: form })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(`Errore upload: ${data.error}`, "error");
                    return;
                }
                jobId = data.job_id;
                videoPath = data.path;
                fileName.textContent = data.filename;
                fileSize.textContent = `${data.size_mb} MB`;
                fileInfo.classList.remove("hidden");
                dropzone.classList.add("hidden");
                appendLog(`Video caricato: ${data.filename}`, "success");
                updateStartButton();
            })
            .catch((err) => {
                appendLog(`Errore upload: ${err.message}`, "error");
            });
    }

    // === Upload JSON ===
    jsonFileInput.addEventListener("change", () => {
        if (!jsonFileInput.files.length || !jobId) return;
        const file = jsonFileInput.files[0];
        const form = new FormData();
        form.append("json_file", file);
        form.append("job_id", jobId);

        fetch("/api/upload-json", { method: "POST", body: form })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(`Errore upload JSON: ${data.error}`, "error");
                    return;
                }
                jsonPath = data.json_path;
                jsonFileName.textContent = data.filename;
                jsonFileInfo.classList.remove("hidden");
                appendLog(`JSON caricato: ${data.filename}`, "success");
            })
            .catch((err) => {
                appendLog(`Errore upload JSON: ${err.message}`, "error");
            });
    });

    removeJsonFile.addEventListener("click", () => {
        jsonPath = null;
        jsonFileInput.value = "";
        jsonFileInfo.classList.add("hidden");
    });

    // === Start ===
    btnStart.addEventListener("click", startPipeline);

    function updateStartButton() {
        btnStart.disabled = !videoPath;
    }

    function collectConfig() {
        const getRadio = (name) => {
            const el = document.querySelector(`input[name="${name}"]:checked`);
            return el ? el.value : null;
        };
        const getCheck = (id) => document.getElementById(id).checked;
        const getVal = (id) => {
            const el = document.getElementById(id);
            if (!el) return null;
            const v = el.value;
            // Prova a convertire in numero
            const n = Number(v);
            return isNaN(n) ? v : n;
        };

        return {
            operation_mode: getRadio("operation_mode"),
            anonymization_method: getRadio("anonymization_method"),
            anonymization_intensity: getVal("anonymization_intensity"),
            person_padding: getVal("person_padding"),
            detection_confidence: getVal("detection_confidence"),
            nms_iou_threshold: getVal("nms_iou_threshold"),
            yolo_model: getVal("yolo_model"),
            enable_sliding_window: getCheck("enable_sliding_window"),
            enable_tracking: getCheck("enable_tracking"),
            enable_temporal_smoothing: getCheck("enable_temporal_smoothing"),
            enable_adaptive_intensity: getCheck("enable_adaptive_intensity"),
            enable_post_render_check: getCheck("enable_post_render_check"),
            enable_fisheye_correction: getCheck("enable_fisheye_correction"),
            enable_motion_detection: getCheck("enable_motion_detection"),
            enable_subframe_interpolation: getCheck("enable_subframe_interpolation"),
            enable_debug_video: getCheck("enable_debug_video"),
            enable_confidence_report: getCheck("enable_confidence_report"),
            normalize: getCheck("normalize"),
        };
    }

    function startPipeline() {
        if (!jobId || !videoPath) return;

        const config = collectConfig();

        // Reset UI
        resetProgress();
        resultsCard.classList.add("hidden");
        resultsEl.innerHTML = "";
        btnStart.disabled = true;
        btnStop.disabled = false;

        appendLog("Avvio pipeline...", "phase");

        fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                job_id: jobId,
                video_path: videoPath,
                config: config,
                review_json: jsonPath,
            }),
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(`Errore avvio: ${data.error}`, "error");
                    btnStart.disabled = false;
                    btnStop.disabled = true;
                    return;
                }
                connectSSE(jobId);
            })
            .catch((err) => {
                appendLog(`Errore avvio: ${err.message}`, "error");
                btnStart.disabled = false;
                btnStop.disabled = true;
            });
    }

    // === Stop ===
    btnStop.addEventListener("click", () => {
        if (!jobId) return;
        fetch("/api/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: jobId }),
        })
            .then((r) => r.json())
            .then((data) => {
                appendLog("Interruzione richiesta...", "info");
            });
    });

    // === SSE ===
    function connectSSE(jid) {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(`/api/progress?job_id=${jid}`);

        eventSource.addEventListener("started", (e) => {
            appendLog("Pipeline avviata", "success");
            progressContainer.classList.remove("hidden");
        });

        eventSource.addEventListener("log", (e) => {
            const data = JSON.parse(e.data);
            appendLog(data.message, "info");
        });

        eventSource.addEventListener("phase", (e) => {
            const data = JSON.parse(e.data);
            progressLabel.textContent = data.description;
        });

        eventSource.addEventListener("phase_label", (e) => {
            const data = JSON.parse(e.data);
            setActivePhase(data.phase);
            appendLog(data.label, "phase");
        });

        eventSource.addEventListener("review_ready", (e) => {
            const data = JSON.parse(e.data);
            appendLog("Revisione manuale pronta — usa l'editor qui sotto", "phase");
            setActivePhase(3);
            ReviewEditor.init(data);
        });

        eventSource.addEventListener("progress", (e) => {
            const data = JSON.parse(e.data);
            updateProgress(data);
        });

        eventSource.addEventListener("completed", (e) => {
            const data = JSON.parse(e.data);
            appendLog("Pipeline completata con successo!", "success");
            onPipelineEnd();
            showResults(data.job_id);
        });

        eventSource.addEventListener("error", (e) => {
            // SSE standard error (connessione persa) vs nostro evento error
            if (e.data) {
                const data = JSON.parse(e.data);
                appendLog(`Errore: ${data.message}`, "error");
                onPipelineEnd();
            }
        });
    }

    function onPipelineEnd() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        btnStart.disabled = false;
        btnStop.disabled = true;
    }

    // === Progresso ===
    function resetProgress() {
        progressContainer.classList.add("hidden");
        progressFill.style.width = "0%";
        progressPercent.textContent = "0%";
        progressLabel.textContent = "";
        progressDetail.textContent = "";
        progressRate.textContent = "";
        currentPhase = 0;
        $$(".phase").forEach((el) => {
            el.classList.remove("active", "done");
        });
    }

    function setActivePhase(num) {
        // Marca le precedenti come "done"
        $$(".phase").forEach((el) => {
            const p = parseInt(el.dataset.phase);
            if (p < num) {
                el.classList.remove("active");
                el.classList.add("done");
            } else if (p === num) {
                el.classList.add("active");
                el.classList.remove("done");
            } else {
                el.classList.remove("active", "done");
            }
        });
        currentPhase = num;
    }

    function updateProgress(data) {
        const pct =
            data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
        progressFill.style.width = pct + "%";
        progressPercent.textContent = pct + "%";
        progressDetail.textContent = `${data.current} / ${data.total} frame`;
        if (data.rate > 0) {
            progressRate.textContent = `${data.rate} frame/s`;
        }
    }

    // === Console ===
    function appendLog(msg, type) {
        const line = document.createElement("div");
        line.className = `console-line console-${type || "info"}`;
        line.textContent = msg;
        consoleEl.appendChild(line);
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    btnClearConsole.addEventListener("click", () => {
        consoleEl.innerHTML = "";
    });

    // === Risultati ===
    function showResults(jid) {
        fetch(`/api/outputs/${jid}`)
            .then((r) => r.json())
            .then((data) => {
                if (!data.files || data.files.length === 0) return;

                resultsEl.innerHTML = "";
                data.files.forEach((f) => {
                    const item = document.createElement("div");
                    item.className = "result-item";
                    item.innerHTML = `
                        <span class="result-name">${f.name}</span>
                        <span class="result-size">${f.size_mb} MB</span>
                        <a href="/api/download/${jid}/${f.type}" class="btn-download"
                           download="${f.name}">Scarica</a>
                    `;
                    resultsEl.appendChild(item);
                });
                resultsCard.classList.remove("hidden");

                // Marca tutte le fasi come completate
                $$(".phase").forEach((el) => {
                    el.classList.remove("active");
                    el.classList.add("done");
                });
            });
    }

    // === Init ===
    updateStartButton();
})();
