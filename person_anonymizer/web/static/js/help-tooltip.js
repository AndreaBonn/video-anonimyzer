/**
 * HelpTooltip — tooltip dei bottoni "?" (click su mobile, hover su desktop).
 */
const HelpTooltip = (function () {
    "use strict";

    const GAP = 8;
    const MARGIN = 8;
    let tooltip = null;
    let activeBtn = null;

    function show(btn) {
        const text = btn.getAttribute("data-help");
        if (!text) return;

        tooltip.textContent = text;
        tooltip.className = "help-tooltip";
        // Render offscreen to measure
        tooltip.style.left = "-9999px";
        tooltip.style.top = "-9999px";
        tooltip.classList.add("visible");

        const rect = btn.getBoundingClientRect();
        const ttRect = tooltip.getBoundingClientRect();
        const spaceAbove = rect.top;
        const spaceBelow = window.innerHeight - rect.bottom;
        const placeAbove = spaceAbove >= ttRect.height + GAP || spaceAbove > spaceBelow;
        const top = placeAbove ? rect.top - ttRect.height - GAP : rect.bottom + GAP;

        // Horizontal: center on button, clamp to viewport
        let left = rect.left + rect.width / 2 - ttRect.width / 2;
        left = Math.max(MARGIN, Math.min(left, window.innerWidth - ttRect.width - MARGIN));

        tooltip.style.top = top + "px";
        tooltip.style.left = left + "px";
        tooltip.style.setProperty("--arrow-x", rect.left + rect.width / 2 - left + "px");
        tooltip.className = "help-tooltip help-tooltip--" + (placeAbove ? "above" : "below") + " visible";

        tooltip.setAttribute("aria-hidden", "false");
        activeBtn = btn;
        btn.classList.add("active");
        btn.setAttribute("aria-describedby", tooltip.id);
    }

    function hide() {
        tooltip.classList.remove("visible");
        tooltip.setAttribute("aria-hidden", "true");
        if (activeBtn) {
            activeBtn.classList.remove("active");
            activeBtn.removeAttribute("aria-describedby");
            activeBtn = null;
        }
    }

    function init() {
        tooltip = document.getElementById("helpTooltip");

        // Click toggle (mobile + desktop)
        document.addEventListener("click", (e) => {
            const btn = e.target.closest(".help-btn");
            if (!btn) {
                hide();
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            const wasActive = activeBtn === btn;
            hide();
            if (!wasActive) show(btn);
        });

        // Hover support (desktop)
        document.addEventListener("mouseover", (e) => {
            const btn = e.target.closest(".help-btn");
            if (btn && activeBtn !== btn) {
                hide();
                show(btn);
            }
        });
        document.addEventListener("mouseout", (e) => {
            const btn = e.target.closest(".help-btn");
            if (btn && activeBtn === btn) hide();
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") hide();
        });
    }

    return { init };
})();
