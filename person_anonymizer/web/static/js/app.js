/**
 * Person Anonymizer — Web GUI
 * Logica frontend: upload, config, SSE progress, download, toast, header status.
 */

(function () {
    "use strict";

    var t = I18n.t;

    // === State ===
    let jobId = null;
    let videoFilename = null;
    let jsonFilename = null;
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

    const uploadProgress = $("#uploadProgress");
    const uploadProgressFill = $("#uploadProgressFill");
    const uploadProgressText = $("#uploadProgressText");

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

    const headerStatusDot = $("#headerStatusDot");
    const headerStatusText = $("#headerStatusText");
    const toastContainer = $("#toastContainer");

    const langBtnIt = $("#langBtnIt");
    const langBtnEn = $("#langBtnEn");

    // === Toast System ===
    const TOAST_ICONS = {
        success: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    };

    function showToast(message, type = "info", duration = 5000) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;

        // Icona (HTML statico sicuro)
        const iconWrapper = document.createElement("span");
        iconWrapper.innerHTML = TOAST_ICONS[type] || TOAST_ICONS.info;
        toast.appendChild(iconWrapper);

        // Messaggio (textContent = sicuro contro XSS)
        const msgSpan = document.createElement("span");
        msgSpan.className = "toast-message";
        msgSpan.textContent = message;
        toast.appendChild(msgSpan);

        // Close button
        const closeBtn = document.createElement("button");
        closeBtn.className = "toast-close";
        closeBtn.setAttribute("aria-label", t("toast.close.aria"));
        closeBtn.textContent = "\u00D7";
        toast.appendChild(closeBtn);

        const dismiss = () => {
            toast.classList.add("toast-out");
            toast.addEventListener("animationend", () => toast.remove());
        };
        closeBtn.addEventListener("click", dismiss);

        toastContainer.appendChild(toast);

        if (duration > 0) {
            setTimeout(dismiss, duration);
        }
    }

    // === Header Status ===
    function setHeaderStatus(state, textKey) {
        headerStatusDot.className = "header-status-dot " + state;
        headerStatusText.textContent = t(textKey);
    }

    // === Language switcher ===
    function updateLangButtons() {
        const lang = I18n.getCurrentLang();
        langBtnIt.classList.toggle("lang-btn--active", lang === "it");
        langBtnEn.classList.toggle("lang-btn--active", lang === "en");
    }
    langBtnIt.addEventListener("click", function () {
        I18n.setLanguage("it");
        updateLangButtons();
    });
    langBtnEn.addEventListener("click", function () {
        I18n.setLanguage("en");
        updateLangButtons();
    });

    // === Sezioni collassabili ===
    $$(".collapsible").forEach((el) => {
        el.addEventListener("click", () => {
            const isCollapsed = el.classList.toggle("collapsed");
            el.setAttribute("aria-expanded", !isCollapsed);
            const target = document.getElementById(el.dataset.target);
            if (target) target.classList.toggle("collapsed-body");
        });
        el.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                el.click();
            }
        });
    });

    // === Help tooltip: DOM element with fixed positioning ===
    const helpTooltip = $("#helpTooltip");
    let activeHelpBtn = null;

    function showHelpTooltip(btn) {
        const text = btn.getAttribute("data-help");
        if (!text) return;

        helpTooltip.textContent = text;
        helpTooltip.className = "help-tooltip";

        // Render offscreen to measure
        helpTooltip.style.left = "-9999px";
        helpTooltip.style.top = "-9999px";
        helpTooltip.classList.add("visible");

        const rect = btn.getBoundingClientRect();
        const ttRect = helpTooltip.getBoundingClientRect();
        const gap = 8;

        // Decide above or below
        const spaceAbove = rect.top;
        const spaceBelow = window.innerHeight - rect.bottom;
        const placeAbove = spaceAbove >= ttRect.height + gap || spaceAbove > spaceBelow;

        let top, arrowDir;
        if (placeAbove) {
            top = rect.top - ttRect.height - gap;
            arrowDir = "above";
        } else {
            top = rect.bottom + gap;
            arrowDir = "below";
        }

        // Horizontal: center on button, clamp to viewport
        let left = rect.left + rect.width / 2 - ttRect.width / 2;
        const margin = 8;
        left = Math.max(margin, Math.min(left, window.innerWidth - ttRect.width - margin));

        // Arrow position relative to tooltip
        const arrowX = rect.left + rect.width / 2 - left;

        helpTooltip.style.top = top + "px";
        helpTooltip.style.left = left + "px";
        helpTooltip.style.setProperty("--arrow-x", arrowX + "px");
        helpTooltip.className = "help-tooltip help-tooltip--" + arrowDir + " visible";

        activeHelpBtn = btn;
        btn.classList.add("active");
    }

    function hideHelpTooltip() {
        helpTooltip.classList.remove("visible");
        if (activeHelpBtn) {
            activeHelpBtn.classList.remove("active");
            activeHelpBtn = null;
        }
    }

    // Click toggle (mobile + desktop)
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".help-btn");
        if (btn) {
            e.preventDefault();
            e.stopPropagation();
            if (activeHelpBtn === btn) {
                hideHelpTooltip();
            } else {
                hideHelpTooltip();
                showHelpTooltip(btn);
            }
            return;
        }
        hideHelpTooltip();
    });

    // Hover support (desktop)
    document.addEventListener("mouseover", (e) => {
        const btn = e.target.closest(".help-btn");
        if (btn && activeHelpBtn !== btn) {
            hideHelpTooltip();
            showHelpTooltip(btn);
        }
    });
    document.addEventListener("mouseout", (e) => {
        const btn = e.target.closest(".help-btn");
        if (btn && activeHelpBtn === btn) {
            hideHelpTooltip();
        }
    });

    // === Slider value display + fill visualization ===
    const sliders = [
        ["anonymization_intensity", "val-intensity", null],
        ["person_padding", "val-padding", null],
        ["detection_confidence", "val-confidence", (v) => parseFloat(v).toFixed(2)],
        ["nms_iou_threshold", "val-nms", (v) => parseFloat(v).toFixed(2)],
    ];

    function updateSliderFill(input) {
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        const val = parseFloat(input.value);
        const pct = ((val - min) / (max - min)) * 100;
        input.style.background = `linear-gradient(90deg, var(--accent) ${pct}%, var(--border) ${pct}%)`;
    }

    sliders.forEach(([id, displayId, fmt]) => {
        const input = document.getElementById(id);
        const display = document.getElementById(displayId);
        if (input && display) {
            input.addEventListener("input", () => {
                display.textContent = fmt ? fmt(input.value) : input.value;
                updateSliderFill(input);
            });
            // Init fill on load
            updateSliderFill(input);
        }
    });

    // === Backend selector: mostra/nasconde modello YOLO ===
    const backendSelect = document.getElementById("detection_backend");
    const yoloModelGroup = document.getElementById("yolo-model-group");
    if (backendSelect && yoloModelGroup) {
        backendSelect.addEventListener("change", () => {
            yoloModelGroup.style.display = backendSelect.value === "sam3" ? "none" : "";
        });
    }

    // === Dropzone: keyboard support ===
    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
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
        videoFilename = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        dropzone.classList.remove("hidden");
        updateStartButton();
        setHeaderStatus("ready", "status.ready");
    });

    function handleVideoFile(file) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        appendLog(t("upload.log", file.name, sizeMB), "info");

        const form = new FormData();
        form.append("video", file);

        // Show upload progress
        uploadProgress.classList.remove("hidden");
        uploadProgressFill.style.width = "0%";
        uploadProgressText.textContent = "0%";

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                uploadProgressFill.style.width = pct + "%";
                uploadProgressText.textContent = pct + "%";
            }
        });

        xhr.addEventListener("load", () => {
            uploadProgress.classList.add("hidden");
            try {
                const data = JSON.parse(xhr.responseText);
                if (data.error) {
                    appendLog(t("upload.error", data.error), "error");
                    showToast(data.error, "error");
                    return;
                }
                jobId = data.job_id;
                videoFilename = data.filename;
                fileName.textContent = data.filename;
                fileSize.textContent = `${data.size_mb} MB`;
                fileInfo.classList.remove("hidden");
                dropzone.classList.add("hidden");
                appendLog(t("upload.success", data.filename), "success");
                showToast(t("upload.success", data.filename), "success");
                setHeaderStatus("ready", "status.ready");
                updateStartButton();
            } catch (err) {
                appendLog(t("upload.error.invalid"), "error");
                showToast(t("upload.toast.error"), "error");
            }
        });

        xhr.addEventListener("error", () => {
            uploadProgress.classList.add("hidden");
            appendLog(t("upload.error.connection.log"), "error");
            showToast(t("upload.error.connection"), "error");
        });

        xhr.open("POST", "/api/upload");
        xhr.send(form);
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
                    appendLog(t("upload.json.error", data.error), "error");
                    showToast(data.error, "error");
                    return;
                }
                jsonFilename = data.filename;
                jsonFileName.textContent = data.filename;
                jsonFileInfo.classList.remove("hidden");
                appendLog(t("upload.json.success", data.filename), "success");
                showToast(t("upload.json.success", data.filename), "success");
            })
            .catch((err) => {
                appendLog(t("upload.json.error", err.message), "error");
                showToast(t("upload.json.error.toast"), "error");
            });
    });

    removeJsonFile.addEventListener("click", () => {
        jsonFilename = null;
        jsonFileInput.value = "";
        jsonFileInfo.classList.add("hidden");
    });

    // === Start ===
    btnStart.addEventListener("click", startPipeline);

    function updateStartButton() {
        btnStart.disabled = !videoFilename;
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
            const n = Number(v);
            return isNaN(n) ? v : n;
        };

        return {
            operation_mode: getRadio("operation_mode"),
            anonymization_method: getRadio("anonymization_method"),
            anonymization_intensity: getVal("anonymization_intensity"),
            person_padding: getVal("person_padding"),
            detection_backend: getVal("detection_backend"),
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
        if (!jobId || !videoFilename) return;

        const config = collectConfig();

        // Reset UI
        resetProgress();
        resultsCard.classList.add("hidden");
        resultsEl.innerHTML = "";
        btnStart.disabled = true;
        btnStop.disabled = false;
        setHeaderStatus("processing", "status.processing");

        appendLog(t("pipeline.starting"), "phase");

        fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                job_id: jobId,
                video_filename: videoFilename,
                config: config,
                review_json_filename: jsonFilename,
            }),
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(t("pipeline.error.start", data.error), "error");
                    showToast(data.error, "error");
                    btnStart.disabled = false;
                    btnStop.disabled = true;
                    setHeaderStatus("error", "status.error");
                    return;
                }
                connectSSE(jobId);
            })
            .catch((err) => {
                appendLog(t("pipeline.error.start", err.message), "error");
                showToast(t("pipeline.error.toast"), "error");
                btnStart.disabled = false;
                btnStop.disabled = true;
                setHeaderStatus("error", "status.error");
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
            .then(() => {
                appendLog(t("pipeline.stop.log"), "info");
                showToast(t("pipeline.stop.toast"), "warning");
            });
    });

    // === SSE ===
    function connectSSE(jid) {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(`/api/progress?job_id=${jid}`);

        eventSource.addEventListener("started", () => {
            appendLog(t("pipeline.started"), "success");
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
            appendLog(t("pipeline.review.ready"), "phase");
            showToast(t("pipeline.review.toast"), "info", 8000);
            setActivePhase(3);
            ReviewEditor.init(data);
        });

        eventSource.addEventListener("progress", (e) => {
            const data = JSON.parse(e.data);
            updateProgress(data);
        });

        eventSource.addEventListener("completed", (e) => {
            const data = JSON.parse(e.data);
            appendLog(t("pipeline.completed"), "success");
            showToast(t("pipeline.completed"), "success", 8000);
            setHeaderStatus("completed", "status.completed");
            onPipelineEnd();
            showResults(data.job_id);
        });

        eventSource.addEventListener("error", (e) => {
            if (e.data) {
                const data = JSON.parse(e.data);
                appendLog(t("pipeline.error.generic", data.message), "error");
                showToast(data.message, "error", 8000);
                setHeaderStatus("error", "status.error");
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

        const progressBar = progressFill.parentElement;
        if (progressBar) {
            progressBar.setAttribute("aria-valuenow", "0");
        }

        $$(".phase").forEach((el) => {
            el.classList.remove("active", "done");
            el.setAttribute("aria-current", "false");
        });
    }

    function setActivePhase(num) {
        $$(".phase").forEach((el) => {
            const p = parseInt(el.dataset.phase);
            if (p < num) {
                el.classList.remove("active");
                el.classList.add("done");
                el.setAttribute("aria-current", "false");
            } else if (p === num) {
                el.classList.add("active");
                el.classList.remove("done");
                el.setAttribute("aria-current", "step");
            } else {
                el.classList.remove("active", "done");
                el.setAttribute("aria-current", "false");
            }
        });
        currentPhase = num;
    }

    function updateProgress(data) {
        const pct =
            data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
        progressFill.style.width = pct + "%";
        progressPercent.textContent = pct + "%";
        progressDetail.textContent = t("pipeline.progress.frames", data.current, data.total);

        const progressBar = progressFill.parentElement;
        if (progressBar) {
            progressBar.setAttribute("aria-valuenow", pct);
        }

        if (data.rate > 0) {
            progressRate.textContent = t("pipeline.progress.rate", data.rate);
        }
    }

    // === Console ===
    function getTimestamp() {
        const now = new Date();
        const hh = String(now.getHours()).padStart(2, "0");
        const mm = String(now.getMinutes()).padStart(2, "0");
        const ss = String(now.getSeconds()).padStart(2, "0");
        return `${hh}:${mm}:${ss}`;
    }

    function appendLog(msg, type) {
        const line = document.createElement("div");
        line.className = `console-line console-${type || "info"}`;

        const ts = document.createElement("span");
        ts.className = "console-timestamp";
        ts.textContent = getTimestamp();

        const text = document.createTextNode(msg);

        line.appendChild(ts);
        line.appendChild(text);
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

                    const nameSpan = document.createElement("span");
                    nameSpan.className = "result-name";
                    nameSpan.textContent = f.name;

                    const sizeSpan = document.createElement("span");
                    sizeSpan.className = "result-size";
                    sizeSpan.textContent = f.size_mb + " MB";

                    const link = document.createElement("a");
                    link.className = "btn-download";
                    link.href = "/api/download/" + jid + "/" + encodeURIComponent(f.type);
                    link.download = f.name;
                    link.textContent = t("results.download");

                    item.appendChild(nameSpan);
                    item.appendChild(sizeSpan);
                    item.appendChild(link);
                    resultsEl.appendChild(item);
                });
                resultsCard.classList.remove("hidden");

                // Marca tutte le fasi come completate
                $$(".phase").forEach((el) => {
                    el.classList.remove("active");
                    el.classList.add("done");
                    el.setAttribute("aria-current", "false");
                });
            });
    }

    // === Init ===
    I18n.applyLanguage();
    updateLangButtons();
    updateStartButton();

    // Init slider fills for frameSlider in review if present
    const frameSlider = $("#frameSlider");
    if (frameSlider) {
        frameSlider.addEventListener("input", () => updateSliderFill(frameSlider));
    }
})();
