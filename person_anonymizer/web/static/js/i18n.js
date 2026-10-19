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
    let currentLang = readStoredLang() || "it";

    function readStoredLang() {
        try {
            return localStorage.getItem(STORAGE_KEY);
        } catch (e) {
            return null;
        }
    }

    // =========================================================================
    // Dizionario traduzioni
    // =========================================================================
    const translations = Object.assign({}, I18N_TEXT_SETUP, I18N_TEXT_RUN);

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

        document.dispatchEvent(new CustomEvent("i18n:applied"));
    }

    /**
     * Cambia lingua e aggiorna il DOM.
     */
    function setLanguage(lang) {
        if (lang !== "it" && lang !== "en") return;
        currentLang = lang;
        try {
            localStorage.setItem(STORAGE_KEY, lang);
        } catch (e) {
            // Storage unavailable (private mode): the choice lasts for this page only
        }
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
