/**
 * Person Anonymizer — Web GUI
 * Orchestrazione: upload, avvio/stop, SSE, guida, dettagli tecnici, tooltip.
 */

(function () {
    "use strict";

    const t = I18n.t;
    const { $, $$, showToast, formatClock, CSRF_HEADER } = UI;

    // === State ===
    let jobId = null;
    let videoFilename = null;
    let jsonFilename = null;
    let eventSource = null;
    let running = false;
    let runMode = "auto";
    let thumbUrl = null;

    const dropzone = $("#dropzone");
    const fileInput = $("#fileInput");
    const jsonFileInput = $("#jsonFileInput");
    const btnStart = $("#btnStart");
    const btnStop = $("#btnStop");
    const consoleEl = $("#console");

    function setHeaderStatus(state, textKey) {
        $("#headerStatusDot").className = "header-status-dot " + state;
        const text = $("#headerStatusText");
        text.dataset.i18n = textKey;
        text.textContent = t(textKey);
    }

    // === Lingua ===
    function updateLangButtons() {
        const lang = I18n.getCurrentLang();
        $("#langBtnIt").classList.toggle("lang-btn--active", lang === "it");
        $("#langBtnEn").classList.toggle("lang-btn--active", lang === "en");
    }
    $$(".lang-btn").forEach((b) => b.addEventListener("click", () => {
        I18n.setLanguage(b.dataset.lang);
        updateLangButtons();
    }));

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

    // === Guida in 3 passi ===
    function setWelcomeVisible(visible) {
        $("#welcomeCard").classList.toggle("hidden", !visible);
    }
    $("#btnGuide").addEventListener("click", () => {
        setWelcomeVisible(true);
        const title = $("#welcomeTitle");
        title.setAttribute("tabindex", "-1");
        title.focus();
        $("#welcomeCard").scrollIntoView({ block: "start" });
    });

    // === Upload video ===
    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleVideoFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleVideoFile(fileInput.files[0]);
    });
    $("#removeFile").addEventListener("click", clearVideo);

    function clearVideo() {
        jobId = null;
        videoFilename = null;
        fileInput.value = "";
        if (thumbUrl) URL.revokeObjectURL(thumbUrl);
        thumbUrl = null;
        $("#videoThumb").removeAttribute("src");
        $("#videoThumb").classList.remove("hidden");
        $("#fileInfo").classList.add("hidden");
        dropzone.classList.remove("hidden");
        clearJson();
        updateStartButton();
        setHeaderStatus("ready", "status.ready");
    }

    function videoDetails(data) {
        const parts = [];
        if (data.duration_s !== null && data.duration_s !== undefined) parts.push(formatClock(data.duration_s));
        if (data.width && data.height) parts.push(`${data.width}×${data.height}`);
        parts.push(`${data.size_mb} MB`);
        return parts.join(" · ");
    }

    function onUploadDone(data, file) {
        jobId = data.job_id;
        videoFilename = data.filename;
        $("#fileName").textContent = data.filename;
        $("#fileDetails").textContent = videoDetails(data);
        thumbUrl = URL.createObjectURL(file);
        const thumb = $("#videoThumb");
        // Con preload=metadata il browser non dipinge alcun frame: un seek minimo lo forza
        thumb.addEventListener("loadedmetadata", () => { thumb.currentTime = 0.1; }, { once: true });
        // Codec non decodificabile dal browser (es. MPEG-4 Part 2): meglio nessuna miniatura che un riquadro nero
        thumb.addEventListener("error", () => thumb.classList.add("hidden"), { once: true });
        thumb.src = thumbUrl;
        $("#fileInfo").classList.remove("hidden");
        dropzone.classList.add("hidden");
        appendLog(t("upload.success", data.filename), "success");
        showToast(t("upload.success", data.filename), "success");
        setHeaderStatus("ready", "status.ready");
        updateStartButton();
    }

    function handleVideoFile(file) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        appendLog(t("upload.log", file.name, sizeMB), "info");
        const form = new FormData();
        form.append("video", file);

        const progress = $("#uploadProgress");
        progress.classList.remove("hidden");
        $("#uploadProgressFill").style.width = "0%";
        $("#uploadProgressText").textContent = "0%";

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener("progress", (e) => {
            if (!e.lengthComputable) return;
            const pct = Math.round((e.loaded / e.total) * 100);
            $("#uploadProgressFill").style.width = pct + "%";
            $("#uploadProgressText").textContent = pct + "%";
        });
        xhr.addEventListener("load", () => {
            progress.classList.add("hidden");
            let data;
            try {
                data = JSON.parse(xhr.responseText);
            } catch (err) {
                appendLog(t("upload.error.invalid"), "error");
                showToast(t("upload.error.invalid"), "error");
                return;
            }
            if (data.error) {
                appendLog(t("upload.error", data.error), "error");
                showToast(t("upload.error", data.error), "error");
                return;
            }
            onUploadDone(data, file);
        });
        xhr.addEventListener("error", () => {
            progress.classList.add("hidden");
            appendLog(t("upload.error.connection"), "error");
            showToast(t("upload.error.connection"), "error");
        });
        xhr.open("POST", "/api/upload");
        xhr.setRequestHeader("X-Requested-With", CSRF_HEADER["X-Requested-With"]);
        xhr.send(form);
    }

    // === Upload JSON (riprendi un lavoro) ===
    jsonFileInput.addEventListener("change", () => {
        if (!jsonFileInput.files.length || !jobId) return;
        const form = new FormData();
        form.append("json_file", jsonFileInput.files[0]);
        form.append("job_id", jobId);

        fetch("/api/upload-json", { method: "POST", headers: CSRF_HEADER, body: form })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(t("upload.json.error", data.error), "error");
                    showToast(t("upload.json.error.toast"), "error");
                    return;
                }
                jsonFilename = data.filename;
                $("#jsonFileName").textContent = data.filename;
                $("#jsonFileInfo").classList.remove("hidden");
                appendLog(t("upload.json.success", data.filename), "success");
                showToast(t("upload.json.success", data.filename), "success");
            })
            .catch((err) => {
                appendLog(t("upload.json.error", err.message), "error");
                showToast(t("upload.json.error.toast"), "error");
            });
    });

    function clearJson() {
        jsonFilename = null;
        jsonFileInput.value = "";
        $("#jsonFileInfo").classList.add("hidden");
    }
    $("#removeJsonFile").addEventListener("click", clearJson);

    // === Avvio / stop ===
    function updateStartButton() {
        btnStart.disabled = !videoFilename || running;
        $("#startHint").classList.toggle("hidden", Boolean(videoFilename) || running);
        jsonFileInput.disabled = !jobId;
        $("#jsonNeedsVideo").classList.toggle("hidden", Boolean(jobId));
    }

    function setRunning(isRunning) {
        running = isRunning;
        SettingsUI.setLocked(isRunning);
        $("#removeFile").disabled = isRunning;
        btnStop.classList.toggle("hidden", !isRunning);
        $("#stopConfirm").classList.add("hidden");
        updateStartButton();
    }

    btnStart.addEventListener("click", startPipeline);

    function startPipeline() {
        if (!jobId || !videoFilename) return;
        const config = SettingsUI.collect();
        runMode = config.operation_mode;

        ProgressView.requestNotifications();
        ProgressView.start(config, Boolean(jsonFilename));
        ResultsView.hide();
        setWelcomeVisible(false);
        setRunning(true);
        setHeaderStatus("processing", "status.processing");
        appendLog(t("pipeline.starting"), "phase");

        fetch("/api/start", {
            method: "POST",
            headers: Object.assign({ "Content-Type": "application/json" }, CSRF_HEADER),
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
                    onStartFailed(data.error, t("pipeline.error.toast.run", data.error));
                    return;
                }
                connectSSE(jobId);
            })
            .catch((err) => onStartFailed(err.message, t("pipeline.error.toast")));
    }

    function onStartFailed(reason, toastText) {
        appendLog(t("pipeline.error.start", reason), "error");
        showToast(toastText, "error");
        ProgressView.reset();
        setRunning(false);
        setHeaderStatus("error", "status.error");
    }

    btnStop.addEventListener("click", () => {
        $("#stopConfirm").classList.remove("hidden");
        $("#btnStopCancel").focus();
    });
    $("#btnStopCancel").addEventListener("click", () => {
        $("#stopConfirm").classList.add("hidden");
        btnStop.focus();
    });
    $("#btnStopConfirm").addEventListener("click", () => {
        $("#stopConfirm").classList.add("hidden");
        if (!jobId) return;
        fetch("/api/stop", {
            method: "POST",
            headers: Object.assign({ "Content-Type": "application/json" }, CSRF_HEADER),
            body: JSON.stringify({ job_id: jobId }),
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    appendLog(t("pipeline.error.generic", data.error), "error");
                    showToast(t("pipeline.stop.failed"), "error");
                    return;
                }
                appendLog(t("pipeline.stop.log"), "info");
                showToast(t("pipeline.stop.toast"), "warning");
            })
            .catch(() => showToast(t("pipeline.stop.failed"), "error"));
    });

    // === SSE ===
    function connectSSE(jid) {
        if (eventSource) eventSource.close();
        eventSource = new EventSource(`/api/progress?job_id=${jid}`);

        eventSource.addEventListener("started", () => appendLog(t("pipeline.started"), "success"));
        eventSource.addEventListener("log", (e) => appendLog(JSON.parse(e.data).message, "info"));
        eventSource.addEventListener("phase_label", (e) => {
            const data = JSON.parse(e.data);
            ProgressView.setPhase(data.phase);
            appendLog(data.label, "phase");
        });
        eventSource.addEventListener("review_ready", (e) => {
            appendLog(t("pipeline.review.ready"), "phase");
            showToast(t("pipeline.review.toast"), "info", 8000);
            ProgressView.setPhase(3);
            ReviewEditor.init(JSON.parse(e.data));
        });
        eventSource.addEventListener("progress", (e) => ProgressView.update(JSON.parse(e.data)));
        eventSource.addEventListener("completed", (e) => {
            const data = JSON.parse(e.data);
            appendLog(t("pipeline.completed"), "success");
            showToast(t("pipeline.completed"), "success", 8000);
            setHeaderStatus("completed", "status.completed");
            ProgressView.complete();
            onPipelineEnd();
            showResults(data.job_id, data.post_check || null);
        });
        eventSource.addEventListener("stopped", () => {
            appendLog(t("pipeline.stopped.log"), "info");
            showToast(t("pipeline.stopped.toast"), "warning", 8000);
            setHeaderStatus("ready", "status.stopped");
            ProgressView.reset();
            onPipelineEnd();
        });
        eventSource.addEventListener("error", (e) => {
            if (!e.data) return;
            const data = JSON.parse(e.data);
            appendLog(t("pipeline.error.generic", data.message), "error");
            showToast(t("pipeline.error.toast.run", data.message), "error", 8000);
            setHeaderStatus("error", "status.error");
            $("#consoleDetails").open = true;
            ProgressView.fail();
            onPipelineEnd();
        });
    }

    function onPipelineEnd() {
        // Dopo uno stop o un errore in fase 3 l'editor non deve restare attivo su un job finito
        ReviewEditor.deactivate();
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        setRunning(false);
    }

    function showResults(jid, postCheck) {
        fetch(`/api/outputs/${jid}`)
            .then((r) => r.json())
            .then((data) => {
                if (!data.files || data.files.length === 0) {
                    onResultsMissing(t("results.missing"));
                    return;
                }
                ResultsView.show(jid, data.files, postCheck, runMode, startOver);
            })
            .catch((err) => onResultsMissing(err.message));
    }

    function onResultsMissing(reason) {
        appendLog(t("pipeline.error.generic", reason), "error");
        showToast(t("results.missing.toast"), "error", 8000);
        $("#consoleDetails").open = true;
    }

    function startOver() {
        ResultsView.hide();
        ProgressView.reset();
        clearVideo();
        setWelcomeVisible(true);
        dropzone.focus();
    }

    // === Dettagli tecnici (console) ===
    function getTimestamp() {
        const now = new Date();
        return [now.getHours(), now.getMinutes(), now.getSeconds()]
            .map((n) => String(n).padStart(2, "0"))
            .join(":");
    }

    function updateConsoleCount() {
        const n = consoleEl.querySelectorAll(".console-line").length;
        $("#consoleCount").textContent = t("console.count", n);
    }

    function appendLog(msg, type) {
        const line = document.createElement("div");
        line.className = `console-line console-${type || "info"}`;
        const ts = document.createElement("span");
        ts.className = "console-timestamp";
        ts.textContent = getTimestamp();
        line.appendChild(ts);
        line.appendChild(document.createTextNode(msg));
        consoleEl.appendChild(line);
        consoleEl.scrollTop = consoleEl.scrollHeight;
        updateConsoleCount();
    }

    $("#btnClearConsole").addEventListener("click", () => {
        consoleEl.textContent = "";
        updateConsoleCount();
    });

    // === Init ===
    HelpTooltip.init();
    SettingsUI.init();
    I18n.applyLanguage();
    updateLangButtons();
    updateStartButton();
    updateConsoleCount();
    document.addEventListener("i18n:applied", updateConsoleCount);

    const frameSlider = $("#frameSlider");
    frameSlider.addEventListener("input", () => SettingsUI.updateSliderFill(frameSlider));
})();
