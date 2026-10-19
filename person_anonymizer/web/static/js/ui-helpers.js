/**
 * UI — helper condivisi: selettori, toast, formattazione tempi e livelli.
 */
const UI = (function () {
    "use strict";

    const TOAST_ICONS = {
        success: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    };
    const DEFAULT_TOAST_MS = 5000;
    // Required by the server CSRF check (middleware.csrf_check) on every POST/PUT/DELETE
    const CSRF_HEADER = { "X-Requested-With": "XMLHttpRequest" };
    const SECONDS_PER_MINUTE = 60;

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    function showToast(message, type = "info", duration = DEFAULT_TOAST_MS) {
        const container = $("#toastContainer");
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

        const closeBtn = document.createElement("button");
        closeBtn.className = "toast-close";
        closeBtn.type = "button";
        closeBtn.setAttribute("aria-label", I18n.t("toast.close.aria"));
        closeBtn.textContent = "×";
        toast.appendChild(closeBtn);

        const dismiss = () => {
            toast.classList.add("toast-out");
            toast.addEventListener("animationend", () => toast.remove());
        };
        closeBtn.addEventListener("click", dismiss);
        container.appendChild(toast);
        if (duration > 0) setTimeout(dismiss, duration);
    }

    /** Secondi → "m:ss" (o "h:mm:ss" oltre l'ora). */
    function formatClock(totalSeconds) {
        const s = Math.max(0, Math.floor(totalSeconds));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / SECONDS_PER_MINUTE);
        const sec = String(s % SECONDS_PER_MINUTE).padStart(2, "0");
        if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${sec}`;
        return `${String(m).padStart(2, "0")}:${sec}`;
    }

    /**
     * Restituisce la chiave i18n del livello per un valore.
     * levels: [[soglia_superiore_esclusa, chiave], ..., [Infinity, chiave]]
     */
    function levelKey(value, levels) {
        for (let i = 0; i < levels.length; i++) {
            if (value < levels[i][0]) return levels[i][1];
        }
        return levels[levels.length - 1][1];
    }

    return { $, $$, showToast, formatClock, levelKey, SECONDS_PER_MINUTE, CSRF_HEADER };
})();
