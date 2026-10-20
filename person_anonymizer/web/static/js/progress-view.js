/**
 * ProgressView — fasi, barra di avanzamento a parole, tempo stimato,
 * titolo della scheda e notifica a fine lavoro.
 *
 * Le fasi seguono la numerazione della pipeline ("[FASE n/5]"):
 * 1 ricerca persone, 2 controllo automatico, 3 revisione, 4 creazione video, 5 finalizzazione.
 */
const ProgressView = (function () {
    "use strict";

    const t = I18n.t;
    const { $, $$, SECONDS_PER_MINUTE } = UI;

    const PHASE_REVIEW = 3;
    const PHASE_CHECK = 2;
    // Prima di qualche secondo la velocità misurata include il caricamento del modello
    const ETA_MIN_ELAPSED_S = 3;
    const BASE_TITLE = document.title;

    let currentPhase = 0;
    let skipped = new Set();

    function renderPhases() {
        $$(".phase").forEach((el) => {
            const p = parseInt(el.dataset.phase, 10);
            const isSkipped = skipped.has(p);
            el.classList.toggle("skipped", isSkipped);
            el.classList.toggle("done", !isSkipped && p < currentPhase);
            el.classList.toggle("active", p === currentPhase);
            el.setAttribute("aria-current", p === currentPhase ? "step" : "false");
        });
        $("#progressLabel").textContent = t(`phase.plain.${currentPhase}`);
    }

    function reset() {
        currentPhase = 0;
        skipped = new Set();
        $("#progressContainer").classList.add("hidden");
        $("#progressContainer").classList.remove("is-waiting");
        $("#progressFill").style.width = "0%";
        $("#progressPercent").textContent = "0%";
        $("#progressEta").textContent = "";
        $("#progressDetail").textContent = "";
        $("#progressFill").parentElement.setAttribute("aria-valuenow", "0");
        document.title = BASE_TITLE;
        renderPhases();
    }

    /** Segna le fasi che questa configurazione non eseguirà. */
    function start(config, resumedFromJson) {
        reset();
        // A saved JSON always reopens the review, unless it is normalized to rectangles
        const resumesIntoReview = resumedFromJson && !config.normalize;
        if (config.operation_mode === "auto" && !resumesIntoReview) skipped.add(PHASE_REVIEW);
        if (!config.enable_post_render_check || resumedFromJson) skipped.add(PHASE_CHECK);
        $("#progressContainer").classList.remove("hidden");
        renderPhases();
    }

    function setPhase(num) {
        currentPhase = num;
        // La revisione aspetta l'utente: nessuna barra da riempire
        $("#progressContainer").classList.toggle("is-waiting", num === PHASE_REVIEW);
        $("#progressFill").style.width = "0%";
        $("#progressPercent").textContent = "0%";
        $("#progressEta").textContent = "";
        $("#progressDetail").textContent = "";
        renderPhases();
    }

    function etaText(remainingFrames, rate) {
        if (!(rate > 0)) return "";
        const minutes = Math.round(remainingFrames / rate / SECONDS_PER_MINUTE);
        if (minutes < 1) return t("progress.eta.short");
        if (minutes === 1) return t("progress.eta.one");
        return t("progress.eta.minutes", minutes);
    }

    function update(data) {
        const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
        $("#progressFill").style.width = pct + "%";
        $("#progressPercent").textContent = pct + "%";
        $("#progressFill").parentElement.setAttribute("aria-valuenow", pct);
        $("#progressDetail").textContent = t("progress.detail", data.current, data.total);
        $("#progressEta").textContent = data.elapsed >= ETA_MIN_ELAPSED_S
            ? etaText(data.total - data.current, data.rate)
            : "";
        document.title = t("progress.tab.title", pct);
    }

    function requestNotifications() {
        if ("Notification" in window && Notification.permission === "default") {
            Notification.requestPermission().catch(() => {});
        }
    }

    function notifyDone() {
        if (!document.hidden || !("Notification" in window)) return;
        if (Notification.permission === "granted") {
            new Notification(t("notify.done.title"), { body: t("notify.done.body") });
        }
    }

    function complete() {
        currentPhase = Infinity;
        renderPhases();
        $("#progressLabel").textContent = t("pipeline.completed");
        $("#progressContainer").classList.add("hidden");
        document.title = BASE_TITLE;
        notifyDone();
    }

    function fail() {
        document.title = BASE_TITLE;
    }

    document.addEventListener("i18n:applied", () => {
        if (currentPhase > 0 && currentPhase !== Infinity) renderPhases();
    });

    return { reset, start, setPhase, update, complete, fail, requestNotifications };
})();
