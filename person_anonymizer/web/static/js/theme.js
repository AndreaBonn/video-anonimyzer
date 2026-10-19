/**
 * Applica il tema salvato prima del primo paint (caricato nel <head>).
 * Senza scelta salvata il CSS segue prefers-color-scheme.
 */
(function () {
    "use strict";
    try {
        var saved = localStorage.getItem("pa_theme");
        if (saved === "light" || saved === "dark") {
            document.documentElement.setAttribute("data-theme", saved);
        }
    } catch (e) {
        // Storage unavailable: the system preference applies
    }
})();
