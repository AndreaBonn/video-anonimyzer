/**
 * ReviewEditor — Editor canvas per revisione manuale poligoni via web.
 *
 * Controlli:
 *   Click        → aggiunge punto al poligono in corso
 *   Enter        → chiude poligono (min 3 punti)
 *   D            → toggle modalità elimina (click su poligono per rimuoverlo)
 *   Esc          → annulla poligono in corso
 *   Frecce / Spazio → navigazione frame
 *   Ctrl+Z       → undo ultimo punto
 *   Slider       → navigazione rapida
 */
var ReviewEditor = (function () {
    "use strict";

    // --- Stato ---
    var canvas, ctx;
    var frameImg = new Image();
    var metadata = null;         // {total_frames, frame_w, frame_h, fps}
    var annotations = {};        // {frame_idx_str: {auto:[], manual:[], intensities:[]}}
    var currentFrame = 0;
    var scaleFactor = 1.0;
    var deleteMode = false;
    var drawingPoints = [];      // punti del poligono in corso (coordinate originali)
    var active = false;

    // --- DOM refs ---
    var reviewCard, frameSlider, frameLabel, polyCount, modeIndicator;
    var btnConfirm, btnPrev, btnNext;

    // --- Colori ---
    var COLOR_AUTO = "rgba(0, 200, 0, 0.5)";
    var COLOR_AUTO_STROKE = "rgb(0, 200, 0)";
    var COLOR_MANUAL = "rgba(255, 165, 0, 0.5)";
    var COLOR_MANUAL_STROKE = "rgb(255, 165, 0)";
    var COLOR_DRAWING = "rgba(0, 220, 255, 0.7)";
    var COLOR_DRAWING_FILL = "rgba(0, 220, 255, 0.2)";

    // ============================
    // Inizializzazione
    // ============================
    function init(meta) {
        metadata = meta;
        currentFrame = 0;
        deleteMode = false;
        drawingPoints = [];
        annotations = {};
        active = true;

        // Riferimenti DOM
        reviewCard = document.getElementById("reviewCard");
        canvas = document.getElementById("reviewCanvas");
        ctx = canvas.getContext("2d");
        frameSlider = document.getElementById("frameSlider");
        frameLabel = document.getElementById("frameLabel");
        polyCount = document.getElementById("polyCount");
        modeIndicator = document.getElementById("modeIndicator");
        btnConfirm = document.getElementById("btnConfirmReview");
        btnPrev = document.getElementById("btnPrevFrame");
        btnNext = document.getElementById("btnNextFrame");

        // Configura slider
        frameSlider.min = 0;
        frameSlider.max = metadata.total_frames - 1;
        frameSlider.value = 0;

        // Mostra la card
        reviewCard.classList.remove("hidden");

        // Carica annotazioni
        fetch("/api/review/annotations")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                annotations = data;
                loadFrame(0);
            });

        // Event listener
        canvas.addEventListener("click", onCanvasClick);
        frameSlider.addEventListener("input", onSliderChange);
        btnConfirm.addEventListener("click", onConfirm);
        btnPrev.addEventListener("click", function () { navigateFrame(-1); });
        btnNext.addEventListener("click", function () { navigateFrame(1); });
        document.addEventListener("keydown", onKeyDown);
    }

    // ============================
    // Caricamento frame
    // ============================
    function loadFrame(idx) {
        if (idx < 0 || idx >= metadata.total_frames) return;
        currentFrame = idx;
        frameSlider.value = idx;
        updateInfoBar();

        fetch("/api/review/frame/" + idx)
            .then(function (r) {
                scaleFactor = parseFloat(r.headers.get("X-Scale-Factor") || "1");
                return r.blob();
            })
            .then(function (blob) {
                var url = URL.createObjectURL(blob);
                frameImg.onload = function () {
                    canvas.width = frameImg.width;
                    canvas.height = frameImg.height;
                    render();
                    URL.revokeObjectURL(url);
                };
                frameImg.src = url;
            });
    }

    // ============================
    // Rendering canvas
    // ============================
    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(frameImg, 0, 0);

        var fdata = getFrameAnnotations(currentFrame);

        // Poligoni auto (verde)
        drawPolygons(fdata.auto || [], COLOR_AUTO, COLOR_AUTO_STROKE);

        // Poligoni manuali (arancione)
        drawPolygons(fdata.manual || [], COLOR_MANUAL, COLOR_MANUAL_STROKE);

        // Poligono in corso (ciano)
        if (drawingPoints.length > 0) {
            ctx.beginPath();
            for (var i = 0; i < drawingPoints.length; i++) {
                var dp = toDisplay(drawingPoints[i]);
                if (i === 0) ctx.moveTo(dp[0], dp[1]);
                else ctx.lineTo(dp[0], dp[1]);
            }
            ctx.strokeStyle = COLOR_DRAWING;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Riempimento se >= 3 punti
            if (drawingPoints.length >= 3) {
                ctx.fillStyle = COLOR_DRAWING_FILL;
                ctx.fill();
            }

            // Punti
            for (var j = 0; j < drawingPoints.length; j++) {
                var pp = toDisplay(drawingPoints[j]);
                ctx.beginPath();
                ctx.arc(pp[0], pp[1], 4, 0, Math.PI * 2);
                ctx.fillStyle = COLOR_DRAWING;
                ctx.fill();
            }
        }

        updateInfoBar();
    }

    function drawPolygons(polys, fill, stroke) {
        for (var i = 0; i < polys.length; i++) {
            var poly = polys[i];
            if (poly.length < 3) continue;
            ctx.beginPath();
            for (var j = 0; j < poly.length; j++) {
                var pt = toDisplay(poly[j]);
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
        var key = String(idx);
        if (!annotations[key]) {
            annotations[key] = { auto: [], manual: [], intensities: [] };
        }
        return annotations[key];
    }

    function saveFrameAnnotations() {
        var key = String(currentFrame);
        var fdata = annotations[key] || { auto: [], manual: [], intensities: [] };
        fetch("/api/review/annotations/" + currentFrame, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(fdata),
        });
    }

    // ============================
    // Event handlers
    // ============================
    function onCanvasClick(e) {
        if (!active) return;
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;

        if (deleteMode) {
            // Elimina poligono sotto il cursore
            var orig = toOriginal(x, y);
            deletePolygonAt(orig[0], orig[1]);
            return;
        }

        // Aggiungi punto al poligono in corso
        var pt = toOriginal(x, y);
        drawingPoints.push(pt);
        render();
    }

    function onKeyDown(e) {
        if (!active) return;

        switch (e.key) {
            case "Enter":
                // Chiudi poligono
                if (drawingPoints.length >= 3) {
                    var fdata = getFrameAnnotations(currentFrame);
                    fdata.manual.push(drawingPoints.slice());
                    drawingPoints = [];
                    saveFrameAnnotations();
                    render();
                }
                e.preventDefault();
                break;

            case "d":
            case "D":
                // Toggle modalità elimina
                deleteMode = !deleteMode;
                drawingPoints = [];
                updateInfoBar();
                render();
                e.preventDefault();
                break;

            case "Escape":
                // Annulla poligono in corso o esci da delete mode
                if (drawingPoints.length > 0) {
                    drawingPoints = [];
                    render();
                } else if (deleteMode) {
                    deleteMode = false;
                    updateInfoBar();
                    render();
                }
                e.preventDefault();
                break;

            case "ArrowRight":
            case " ":
                navigateFrame(1);
                e.preventDefault();
                break;

            case "ArrowLeft":
                navigateFrame(-1);
                e.preventDefault();
                break;

            case "z":
                if (e.ctrlKey || e.metaKey) {
                    // Undo ultimo punto
                    if (drawingPoints.length > 0) {
                        drawingPoints.pop();
                        render();
                    }
                    e.preventDefault();
                }
                break;
        }
    }

    function onSliderChange() {
        var idx = parseInt(frameSlider.value);
        drawingPoints = [];
        deleteMode = false;
        loadFrame(idx);
    }

    function onConfirm() {
        if (!active) return;
        fetch("/api/review/confirm", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function () {
                deactivate();
            });
    }

    // ============================
    // Navigazione
    // ============================
    function navigateFrame(delta) {
        var next = currentFrame + delta;
        if (next < 0 || next >= metadata.total_frames) return;
        drawingPoints = [];
        deleteMode = false;
        loadFrame(next);
    }

    // ============================
    // Point-in-polygon test
    // ============================
    function pointInPolygon(px, py, polygon) {
        var inside = false;
        for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            var xi = polygon[i][0], yi = polygon[i][1];
            var xj = polygon[j][0], yj = polygon[j][1];
            var intersect = ((yi > py) !== (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    function deletePolygonAt(px, py) {
        var fdata = getFrameAnnotations(currentFrame);
        var modified = false;

        // Cerca prima nei manuali, poi negli auto
        for (var i = fdata.manual.length - 1; i >= 0; i--) {
            if (pointInPolygon(px, py, fdata.manual[i])) {
                fdata.manual.splice(i, 1);
                modified = true;
                break;
            }
        }
        if (!modified) {
            for (var j = fdata.auto.length - 1; j >= 0; j--) {
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
            saveFrameAnnotations();
            render();
        }
    }

    // ============================
    // Info bar
    // ============================
    function updateInfoBar() {
        if (!metadata) return;
        frameLabel.textContent = "Frame " + currentFrame + " / " + (metadata.total_frames - 1);

        var fdata = getFrameAnnotations(currentFrame);
        var count = (fdata.auto || []).length + (fdata.manual || []).length;
        polyCount.textContent = count + " poligoni";

        if (deleteMode) {
            modeIndicator.textContent = "[DEL]";
            modeIndicator.className = "review-mode-delete";
            canvas.style.cursor = "not-allowed";
        } else {
            modeIndicator.textContent = "[NORM]";
            modeIndicator.className = "review-mode-normal";
            canvas.style.cursor = "crosshair";
        }
    }

    // ============================
    // Disattivazione
    // ============================
    function deactivate() {
        active = false;
        canvas.removeEventListener("click", onCanvasClick);
        document.removeEventListener("keydown", onKeyDown);
        reviewCard.classList.add("hidden");
    }

    // ============================
    // API pubblica
    // ============================
    return {
        init: init,
        deactivate: deactivate,
    };
})();
