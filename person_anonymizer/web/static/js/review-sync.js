/**
 * ReviewSync — saves the review edits to the server and keeps the editor honest
 * about them: an edit the server never accepted must not stay on the canvas,
 * because the final video is rendered from the server's copy.
 */
const ReviewSync = (function () {
    "use strict";

    let session = 0;
    let confirmed = {};   // frame key -> last state the server accepted
    let pending = {};     // frame key -> [{seq, undo}] edits not yet accepted
    let seqByFrame = {};
    let queueByFrame = {};
    let inFlight = new Set();

    function emptyFrame() {
        return { auto: [], manual: [], intensities: [] };
    }

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    /** Starts a session from the annotations the server returned. */
    function reset(serverAnnotations) {
        session++;
        confirmed = clone(serverAnnotations);
        pending = {};
        seqByFrame = {};
        queueByFrame = {};
        inFlight = new Set();
    }

    function put(frameIdx, body) {
        return fetch("/api/review/annotations/" + frameIdx, {
            method: "PUT",
            headers: Object.assign({ "Content-Type": "application/json" }, UI.CSRF_HEADER),
            body: JSON.stringify(body),
        }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
        });
    }

    /**
     * Queue the save of one frame. PUTs of the same frame run one at a time, so
     * the server applies them in edit order. If the latest one fails, every edit
     * still pending on that frame is undone and ``onRollback(key, state)`` gets
     * the state the server holds. Resolves to false only on such a rollback.
     */
    function save(frameIdx, state, undo, onRollback) {
        const key = String(frameIdx);
        const mySession = session;
        const seq = (seqByFrame[key] || 0) + 1;
        seqByFrame[key] = seq;
        (pending[key] = pending[key] || []).push({ seq: seq, undo: undo });
        const body = clone(state);

        const attempt = (queueByFrame[key] || Promise.resolve())
            .then(function () { return put(frameIdx, body); })
            .then(function () {
                if (mySession !== session) return true;
                confirmed[key] = body;
                pending[key] = pending[key].filter(function (e) { return e.seq > seq; });
                return true;
            })
            .catch(function () {
                // A later save of the same frame carries this edit too: its outcome decides
                if (mySession !== session || seqByFrame[key] !== seq) return true;
                pending[key].forEach(function (e) { e.undo(); });
                pending[key] = [];
                onRollback(key, clone(confirmed[key] || emptyFrame()));
                return false;
            });
        queueByFrame[key] = attempt;
        inFlight.add(attempt);
        attempt.then(function () { inFlight.delete(attempt); });
        return attempt;
    }

    /** Resolves to true when every queued save has been accepted. */
    function settle() {
        return Promise.all(Array.from(inFlight)).then(function (results) {
            return results.every(Boolean);
        });
    }

    return { reset: reset, save: save, settle: settle };
})();
