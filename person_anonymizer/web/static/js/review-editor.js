/**
 * ReviewEditor — Editor canvas per revisione manuale poligoni via web.
 *
 * Controlli (bottoni della barra strumenti o tastiera):
 *   Click / "Disegna area"   → aggiunge punto al poligono in corso
 *   Enter / "Chiudi area"    → chiude poligono (min 3 punti)
 *   D / "Elimina area"       → toggle modalità elimina (click su poligono per rimuoverlo)
 *   Esc                      → annulla poligono in corso
 *   Ctrl+Z / "Annulla punto" → undo ultimo punto
 *   Frecce / Spazio, slider  → navigazione frame
 */
const ReviewEditor = (function () {
    "use strict";

    const t = I18n.t;
    const MIN_POLYGON_POINTS = 3;

    // --- Stato ---
    let canvas, ctx;
    let frameImg = new Image();
    // Only the latest frame request may draw: responses can arrive out of order.
    let frameRequestSeq = 0;
    let frameReady = false;
    let metadata = null;         // {total_frames, frame_w, frame_h, fps}
    let annotations = {};        // {frame_idx_str: {auto:[], manual:[], intensities:[]}}
    let currentFrame = 0;
    let scaleFactor = 1.0;
    let deleteMode = false;
    let drawingPoints = [];      // punti del poligono in corso (coordinate originali)
    let active = false;
    let controlsBound = false;
    let addedCount = 0;
    let deletedCount = 0;

    // --- DOM refs ---
    let frameSlider, frameLabel, polyCount, modeIndicator;

    // --- Colori ---
    const COLOR_AUTO = "rgba(0, 200, 0, 0.5)";
    const COLOR_AUTO_STROKE = "rgb(0, 200, 0)";
    const COLOR_MANUAL = "rgba(255, 165, 0, 0.5)";
    const COLOR_MANUAL_STROKE = "rgb(255, 165, 0)";
    const COLOR_DRAWING = "rgba(45, 212, 191, 0.8)";
    const COLOR_DRAWING_FILL = "rgba(45, 212, 191, 0.2)";

    // ============================
    // Inizializzazione
    // ============================
    function init(meta) {
        metadata = meta;
        currentFrame = 0;
        deleteMode = false;
        drawingPoints = [];
        annotations = {};
        addedCount = 0;
        deletedCount = 0;
        active = true;

        canvas = document.getElementById("reviewCanvas");
        ctx = canvas.getContext("2d");
        // A new session invalidates loads still in flight from the previous one
        frameRequestSeq++;
        frameReady = false;
        render();
        frameSlider = document.getElementById("frameSlider");
        frameLabel = document.getElementById("frameLabel");
        polyCount = document.getElementById("polyCount");
        modeIndicator = document.getElementById("modeIndicator");

        frameSlider.min = 0;
        frameSlider.max = metadata.total_frames - 1;
        frameSlider.value = 0;

        const reviewCard = document.getElementById("reviewCard");
        reviewCard.classList.remove("hidden");
        reviewCard.scrollIntoView({ block: "start" });
        updateSummary();

        fetch("/api/review/annotations")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                annotations = data;
                loadFrame(0);
            });

        bindControlsOnce();
        canvas.addEventListener("click", onCanvasClick);
        document.addEventListener("keydown", onKeyDown);
    }

    // Bottoni e slider restano nel DOM tra una revisione e l'altra: un solo binding
    function bindControlsOnce() {
        if (controlsBound) return;
        controlsBound = true;
        frameSlider.addEventListener("input", onSliderChange);
        document.getElementById("btnConfirmReview").addEventListener("click", onConfirm);
        document.getElementById("btnPrevFrame").addEventListener("click", function () { navigateFrame(-1); });
        document.getElementById("btnNextFrame").addEventListener("click", function () { navigateFrame(1); });
        document.getElementById("btnNextPeople").addEventListener("click", jumpToNextWithPeople);
        document.getElementById("toolDraw").addEventListener("click", function () { setDeleteMode(false); });
        document.getElementById("toolDelete").addEventListener("click", function () { setDeleteMode(!deleteMode); });
        document.getElementById("toolClose").addEventListener("click", closePolygon);
        document.getElementById("toolUndo").addEventListener("click", undoPoint);
        document.addEventListener("i18n:applied", function () { if (active) updateInfoBar(); });
    }

    // ============================
    // Caricamento frame
    // ============================
    function loadFrame(idx) {
        if (idx < 0 || idx >= metadata.total_frames) return;
        currentFrame = idx;
        frameSlider.value = idx;
        updateInfoBar();

        // Until the new image is in, the canvas stays blank and ignores clicks:
        // annotations must never be drawn or edited over another frame's image.
        const seq = ++frameRequestSeq;
        frameReady = false;
        render();

        let nextScale = 1.0;
        fetch("/api/review/frame/" + idx)
            .then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                nextScale = parseFloat(r.headers.get("X-Scale-Factor") || "1");
                return r.blob();
            })
            .then(function (blob) {
                if (seq !== frameRequestSeq) return;
                const url = URL.createObjectURL(blob);
                const img = new Image();
                img.onload = function () {
                    URL.revokeObjectURL(url);
                    if (seq !== frameRequestSeq) return;
                    frameImg = img;
                    scaleFactor = nextScale;
                    canvas.width = img.width;
                    canvas.height = img.height;
                    frameReady = true;
                    render();
                };
                img.onerror = function () {
                    URL.revokeObjectURL(url);
                    if (seq === frameRequestSeq) UI.showToast(t("review.frame.error"), "error");
                };
                img.src = url;
            })
            .catch(function () {
                if (seq !== frameRequestSeq) return;
                UI.showToast(t("review.frame.error"), "error");
            });
    }

    // ============================
    // Rendering canvas
    // ============================
    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!frameReady) return;
        ctx.drawImage(frameImg, 0, 0);

        const fdata = getFrameAnnotations(currentFrame);
        drawPolygons(fdata.auto || [], COLOR_AUTO, COLOR_AUTO_STROKE);
        drawPolygons(fdata.manual || [], COLOR_MANUAL, COLOR_MANUAL_STROKE);

        // Poligono in corso (ciano)
        if (drawingPoints.length > 0) {
            ctx.beginPath();
            for (let i = 0; i < drawingPoints.length; i++) {
                const dp = toDisplay(drawingPoints[i]);
                if (i === 0) ctx.moveTo(dp[0], dp[1]);
                else ctx.lineTo(dp[0], dp[1]);
            }
            ctx.strokeStyle = COLOR_DRAWING;
            ctx.lineWidth = 2;
            ctx.stroke();
            if (drawingPoints.length >= MIN_POLYGON_POINTS) {
                ctx.fillStyle = COLOR_DRAWING_FILL;
                ctx.fill();
            }
            for (let j = 0; j < drawingPoints.length; j++) {
                const pp = toDisplay(drawingPoints[j]);
                ctx.beginPath();
                ctx.arc(pp[0], pp[1], 4, 0, Math.PI * 2);
                ctx.fillStyle = COLOR_DRAWING;
                ctx.fill();
            }
        }

        updateInfoBar();
    }

    function drawPolygons(polys, fill, stroke) {
        for (let i = 0; i < polys.length; i++) {
            const poly = polys[i];
            if (poly.length < MIN_POLYGON_POINTS) continue;
            ctx.beginPath();
            for (let j = 0; j < poly.length; j++) {
                const pt = toDisplay(poly[j]);
                if (j === 0) ctx.moveTo(pt[0], pt[1]);
                else ctx.lineTo(pt[0], pt[1]);
            }
            ctx.closePath();
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    // ============================
    // Conversione coordinate
    // ============================
    function toDisplay(pt) {
        return [pt[0] * scaleFactor, pt[1] * scaleFactor];
    }

    function toOriginal(x, y) {
        return [Math.round(x / scaleFactor), Math.round(y / scaleFactor)];
    }

    // ============================
    // Gestione annotazioni
    // ============================
    function getFrameAnnotations(idx) {
        const key = String(idx);
        if (!annotations[key]) {
            annotations[key] = { auto: [], manual: [], intensities: [] };
        }
        return annotations[key];
    }

    function frameHasPeople(idx) {
        const fdata = annotations[String(idx)];
        return Boolean(fdata) && ((fdata.auto || []).length + (fdata.manual || []).length) > 0;
    }

    function saveFrameAnnotations() {
        const key = String(currentFrame);
        const fdata = annotations[key] || { auto: [], manual: [], intensities: [] };
        fetch("/api/review/annotations/" + currentFrame, {
            method: "PUT",
            headers: Object.assign({ "Content-Type": "application/json" }, UI.CSRF_HEADER),
            body: JSON.stringify(fdata),
        });
    }

    // ============================
    // Azioni
    // ============================
    function closePolygon() {
        if (!active) return;
        if (drawingPoints.length < MIN_POLYGON_POINTS) {
            UI.showToast(t("review.close.need_points"), "info");
            return;
        }
        getFrameAnnotations(currentFrame).manual.push(drawingPoints.slice());
        drawingPoints = [];
        addedCount++;
        saveFrameAnnotations();
        updateSummary();
        render();
    }

    function undoPoint() {
        if (!active || drawingPoints.length === 0) return;
        drawingPoints.pop();
        render();
    }

    function setDeleteMode(on) {
        if (!active) return;
        deleteMode = on;
        drawingPoints = [];
        render();
    }

    function jumpToNextWithPeople() {
        if (!active) return;
        for (let idx = currentFrame + 1; idx < metadata.total_frames; idx++) {
            if (frameHasPeople(idx)) {
                drawingPoints = [];
                deleteMode = false;
                loadFrame(idx);
                return;
            }
        }
        UI.showToast(t("review.no_more_people"), "info");
    }

    // ============================
    // Event handlers
    // ============================
    function onCanvasClick(e) {
        if (!active || !frameReady) return;
        const rect = canvas.getBoundingClientRect();
        // Il canvas è scalato via CSS: riporta il click alle coordinate del bitmap
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        const orig = toOriginal(x, y);

        if (deleteMode) {
            deletePolygonAt(orig[0], orig[1]);
            return;
        }
        drawingPoints.push(orig);
        render();
    }

    function onKeyDown(e) {
        if (!active) return;

        switch (e.key) {
            case "Enter":
                if (e.target.closest && e.target.closest("button")) return;
                closePolygon();
                e.preventDefault();
                break;

            case "d":
            case "D":
                setDeleteMode(!deleteMode);
                e.preventDefault();
                break;

            case "Escape":
                // Annulla poligono in corso o esci da delete mode
                if (drawingPoints.length > 0) {
                    drawingPoints = [];
                    render();
                } else if (deleteMode) {
                    setDeleteMode(false);
                }
                e.preventDefault();
                break;

            case "ArrowRight":
            case " ":
                if (e.key === " " && e.target.closest && e.target.closest("button")) return;
                navigateFrame(1);
                e.preventDefault();
                break;

            case "ArrowLeft":
                navigateFrame(-1);
                e.preventDefault();
                break;

            case "z":
                if (e.ctrlKey || e.metaKey) {
                    undoPoint();
                    e.preventDefault();
                }
                break;
        }
    }

    function onSliderChange() {
        if (!active) return;
        drawingPoints = [];
        deleteMode = false;
        loadFrame(parseInt(frameSlider.value, 10));
    }

    function onConfirm() {
        if (!active) return;
        fetch("/api/review/confirm", { method: "POST", headers: UI.CSRF_HEADER })
            .then(function (r) { return r.json(); })
            .then(function () {
                deactivate();
            });
    }

    // ============================
    // Navigazione
    // ============================
    function navigateFrame(delta) {
        if (!active) return;
        const next = currentFrame + delta;
        if (next < 0 || next >= metadata.total_frames) return;
        drawingPoints = [];
        deleteMode = false;
        loadFrame(next);
    }

    // ============================
    // Point-in-polygon test
    // ============================
    function pointInPolygon(px, py, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            const intersect = ((yi > py) !== (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    function deletePolygonAt(px, py) {
        const fdata = getFrameAnnotations(currentFrame);
        let modified = false;

        // Cerca prima nei manuali, poi negli auto
        for (let i = fdata.manual.length - 1; i >= 0; i--) {
            if (pointInPolygon(px, py, fdata.manual[i])) {
                fdata.manual.splice(i, 1);
                modified = true;
                break;
            }
        }
        if (!modified) {
            for (let j = fdata.auto.length - 1; j >= 0; j--) {
                if (pointInPolygon(px, py, fdata.auto[j])) {
                    fdata.auto.splice(j, 1);
                    // Rimuovi anche l'intensità corrispondente
                    if (fdata.intensities && fdata.intensities.length > j) {
                        fdata.intensities.splice(j, 1);
                    }
                    modified = true;
                    break;
                }
            }
        }

        if (modified) {
            deletedCount++;
            saveFrameAnnotations();
            updateSummary();
            render();
        }
    }

    // ============================
    // Info bar e riepilogo
    // ============================
    function frameTime(idx) {
        return metadata.fps > 0 ? UI.formatClock(idx / metadata.fps) : "--:--";
    }

    function updateInfoBar() {
        if (!metadata) return;
        const last = metadata.total_frames - 1;
        frameLabel.textContent = t("review.frame", frameTime(currentFrame), frameTime(last), currentFrame, last);

        const fdata = getFrameAnnotations(currentFrame);
        const count = (fdata.auto || []).length + (fdata.manual || []).length;
        polyCount.textContent = t("review.polygons", count);

        modeIndicator.textContent = t(deleteMode ? "review.mode.delete" : "review.mode.draw");
        modeIndicator.className = deleteMode ? "review-mode-delete" : "review-mode-normal";
        canvas.style.cursor = deleteMode ? "not-allowed" : "crosshair";

        const draw = document.getElementById("toolDraw");
        const del = document.getElementById("toolDelete");
        draw.classList.toggle("tool-btn--active", !deleteMode);
        draw.setAttribute("aria-pressed", String(!deleteMode));
        del.classList.toggle("tool-btn--active", deleteMode);
        del.setAttribute("aria-pressed", String(deleteMode));
    }

    function updateSummary() {
        const el = document.getElementById("reviewSummary");
        el.textContent = addedCount + deletedCount === 0
            ? t("review.summary.none")
            : t("review.summary", addedCount, deletedCount);
    }

    // ============================
    // Disattivazione
    // ============================
    function deactivate() {
        // Chiamata anche a fine pipeline (stop/errore) quando nessuna revisione è stata aperta
        if (!canvas) return;
        active = false;
        canvas.removeEventListener("click", onCanvasClick);
        document.removeEventListener("keydown", onKeyDown);
        document.getElementById("reviewCard").classList.add("hidden");
    }

    return {
        init: init,
        deactivate: deactivate,
    };
})();
