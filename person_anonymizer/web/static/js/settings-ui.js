/**
 * SettingsUI — pannello impostazioni: valori a parole, preset di qualità,
 * sensibilità, disponibilità SAM3, tema, blocco durante l'elaborazione.
 */
const SettingsUI = (function () {
    "use strict";

    const t = I18n.t;
    const { $, $$, levelKey } = UI;

    // Preset di qualità: modello YOLO + sliding window. "accurate" = default di PipelineConfig.
    const PRESETS = {
        fast: { yolo_model: "yolov8n.pt", enable_sliding_window: false },
        balanced: { yolo_model: "yolov8n.pt", enable_sliding_window: true },
        accurate: { yolo_model: "yolov8x.pt", enable_sliding_window: true },
    };
    const INTENSITY_LEVELS = [[8, "scale.light"], [21, "scale.medium"], [Infinity, "scale.strong"]];
    const PADDING_LEVELS = [[10, "scale.narrow"], [31, "scale.normal"], [Infinity, "scale.wide"]];
    const SENSITIVITY_LEVELS = [[40, "scale.low"], [70, "scale.medium.f"], [Infinity, "scale.high"]];
    const SENSITIVITY_SCALE = 100;
    const SAM3_BACKENDS = ["yolo+sam3", "sam3"];
    const THEME_KEY = "pa_theme";

    function sensitivityToConfidence(sens) {
        return Math.round(SENSITIVITY_SCALE - Number(sens)) / SENSITIVITY_SCALE;
    }

    // --- Valori a parole sugli slider ---
    const SLIDERS = [
        ["anonymization_intensity", "val-intensity", (v) => t("scale.value", t(levelKey(v, INTENSITY_LEVELS)), v)],
        ["person_padding", "val-padding", (v) => t("scale.value", t(levelKey(v, PADDING_LEVELS)), v + " px")],
        ["detection_sensitivity", "val-sensitivity", (v) => t("scale.value", t(levelKey(v, SENSITIVITY_LEVELS)), sensitivityToConfidence(v).toFixed(2))],
        ["nms_iou_threshold", "val-nms", (v) => v.toFixed(2)],
    ];

    function updateSliderFill(input) {
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        const pct = ((parseFloat(input.value) - min) / (max - min)) * 100;
        input.style.background = `linear-gradient(90deg, var(--accent) ${pct}%, var(--border) ${pct}%)`;
    }

    function renderSliders() {
        SLIDERS.forEach(([id, displayId, fmt]) => {
            const input = document.getElementById(id);
            document.getElementById(displayId).textContent = fmt(parseFloat(input.value));
            updateSliderFill(input);
        });
    }

    // --- Preset ---
    function selectedPreset() {
        const el = $('input[name="quality_preset"]:checked');
        return el ? el.value : null;
    }

    function renderQualityHint() {
        const preset = selectedPreset();
        $("#qualityHint").textContent = t(preset ? `settings.quality.hint.${preset}` : "settings.quality.hint.custom");
    }

    function applyPreset(name) {
        const p = PRESETS[name];
        $("#yolo_model").value = p.yolo_model;
        $("#enable_sliding_window").checked = p.enable_sliding_window;
        renderQualityHint();
    }

    function markCustom() {
        $$('input[name="quality_preset"]').forEach((el) => { el.checked = false; });
        renderQualityHint();
    }

    // --- Disponibilità SAM3 ---
    function loadCapabilities() {
        fetch("/api/capabilities")
            .then((r) => r.json())
            .then((caps) => {
                if (caps.sam3) return;
                SAM3_BACKENDS.forEach((v) => {
                    $(`#detection_backend option[value="${v}"]`).disabled = true;
                });
                $("#sam3Unavailable").classList.remove("hidden");
            })
            .catch(() => {
                // Without the answer the options stay selectable, as before: the start reports the error
            });
    }

    // --- Tema ---
    function currentTheme() {
        const attr = document.documentElement.getAttribute("data-theme");
        if (attr) return attr;
        return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }

    function toggleTheme() {
        const next = currentTheme() === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        try {
            localStorage.setItem(THEME_KEY, next);
        } catch (e) {
            // Storage unavailable: the theme lasts for this page only
        }
    }

    function init() {
        SLIDERS.forEach(([id]) => document.getElementById(id).addEventListener("input", renderSliders));
        $$('input[name="quality_preset"]').forEach((el) => {
            el.addEventListener("change", () => applyPreset(el.value));
        });
        $("#yolo_model").addEventListener("change", markCustom);
        $("#enable_sliding_window").addEventListener("change", markCustom);
        $("#detection_backend").addEventListener("change", (e) => {
            $("#yolo-model-group").classList.toggle("hidden", e.target.value === "sam3");
        });
        $("#btnTheme").addEventListener("click", toggleTheme);
        document.addEventListener("i18n:applied", () => {
            renderSliders();
            renderQualityHint();
        });
        loadCapabilities();
        renderSliders();
    }

    function collect() {
        const getRadio = (name) => {
            const el = $(`input[name="${name}"]:checked`);
            return el ? el.value : null;
        };
        const getCheck = (id) => document.getElementById(id).checked;
        const getNum = (id) => Number(document.getElementById(id).value);

        return {
            operation_mode: getRadio("operation_mode"),
            anonymization_method: getRadio("anonymization_method"),
            anonymization_intensity: getNum("anonymization_intensity"),
            person_padding: getNum("person_padding"),
            detection_backend: $("#detection_backend").value,
            detection_confidence: sensitivityToConfidence(getNum("detection_sensitivity")),
            nms_iou_threshold: getNum("nms_iou_threshold"),
            yolo_model: $("#yolo_model").value,
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

    function setLocked(locked) {
        $("#settingsFieldset").disabled = locked;
        $("#lockMsg").classList.toggle("hidden", !locked);
    }

    return { init, collect, setLocked, updateSliderFill };
})();
