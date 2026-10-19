/**
 * ResultsView — risultato finale: video anonimizzato in evidenza con anteprima,
 * esito del controllo automatico, altri file descritti, scadenza, nuovo video.
 */
const ResultsView = (function () {
    "use strict";

    const t = I18n.t;
    const { $ } = UI;

    const OTHER_TYPES = ["debug", "report", "annotations"];

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function downloadUrl(jobId, type) {
        return "/api/download/" + jobId + "/" + encodeURIComponent(type);
    }

    function checkLine(postCheck, mode) {
        if (!postCheck) return null;
        const status = postCheck.status;
        let text = status === "residual"
            ? t("results.check.residual", postCheck.residual_frames)
            : t(`results.check.${status}`);
        if (status !== "skipped" && mode === "manual") text += t("results.check.manual_note");
        return el("p", `result-check result-check--${status}`, text);
    }

    function videoBlock(jobId, video, postCheck, mode, onAgain) {
        const block = el("section", "result-main");
        block.appendChild(el("h2", "result-title", "✓ " + t("results.ready")));

        const player = el("video", "result-player");
        player.controls = true;
        player.preload = "metadata";
        player.src = downloadUrl(jobId, "video") + "?inline=1";
        player.setAttribute("aria-label", t("results.preview.aria"));
        block.appendChild(player);

        const row = el("div", "result-row");
        const dl = el("a", "btn btn-primary result-download", t("results.download.video"));
        dl.href = downloadUrl(jobId, "video");
        dl.download = video.name;
        row.appendChild(dl);
        row.appendChild(el("span", "result-meta", `${video.name} · ${video.size_mb} MB`));
        block.appendChild(row);

        const check = checkLine(postCheck, mode);
        if (check) block.appendChild(check);
        block.appendChild(el("p", "result-expire", t("results.expire")));

        const again = el("button", "btn btn-ghost result-again", t("results.again"));
        again.type = "button";
        again.addEventListener("click", onAgain);
        block.appendChild(again);
        return block;
    }

    function fileItem(jobId, f) {
        const known = OTHER_TYPES.includes(f.type);
        const item = el("div", "result-item");
        item.appendChild(el("span", "result-name", known ? t(`results.type.${f.type}`) : f.name));
        item.appendChild(el("span", "result-size", f.size_mb + " MB"));
        const link = el("a", "btn-download", t("results.download"));
        link.href = downloadUrl(jobId, f.type);
        link.download = f.name;
        item.appendChild(link);
        item.appendChild(el("span", "result-desc", known ? t(`results.type.${f.type}.desc`) : t("results.type.unknown.desc")));
        if (known) item.appendChild(el("span", "result-file", f.name));
        return item;
    }

    function othersBlock(jobId, others) {
        const block = el("section", "result-others");
        const head = el("h3", "result-others-title", t("results.others") + " ");
        head.appendChild(el("span", "result-others-sub", t("results.others.sub")));
        block.appendChild(head);
        const list = el("div", "results-list");
        others.forEach((f) => list.appendChild(fileItem(jobId, f)));
        block.appendChild(list);
        return block;
    }

    /**
     * files: [{name, type, size_mb}] da /api/outputs; postCheck: {status, residual_frames} | null.
     */
    function show(jobId, files, postCheck, mode, onAgain) {
        const container = $("#results");
        container.textContent = "";
        const video = files.find((f) => f.type === "video");
        const others = files.filter((f) => f.type !== "video");
        if (video) container.appendChild(videoBlock(jobId, video, postCheck, mode, onAgain));
        if (others.length) container.appendChild(othersBlock(jobId, others));
        $("#resultsCard").classList.remove("hidden");
        $("#resultsCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function hide() {
        $("#resultsCard").classList.add("hidden");
        $("#results").textContent = "";
    }

    return { show, hide };
})();
