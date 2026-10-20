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

    function fetchServerFrame(key) {
        return fetch("/api/review/annotations", { cache: "no-store" }).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
        }).then(function (all) {
            return all[key] || emptyFrame();
        });
    }

    // Key order differs between the editor and Flask's sorted JSON
    function canonical(value) {
        if (Array.isArray(value)) return value.map(canonical);
        if (value === null || typeof value !== "object") return value;
        const out = {};
        Object.keys(value).sort().forEach(function (k) { out[k] = canonical(value[k]); });
        return out;
    }

    function sameFrame(a, b) {
        return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
    }

    /**
     * Queue the save of one frame. PUTs of the same frame run one at a time, so
     * the server applies them in edit order. If the latest one fails, the
     * frame is read back from the server: a PUT whose response was lost may
     * have been applied, and so may an earlier save that a later one
     * superseded. Only the edits the server does not hold are undone, and
     * ``onRollback(key, state)`` gets the state it holds. Resolves to false
     * only on such a rollback.
     */
    function save(frameIdx, state, undo, onRollback) {
        const key = String(frameIdx);
        const mySession = session;
        const seq = (seqByFrame[key] || 0) + 1;
        seqByFrame[key] = seq;
        const body = clone(state);
        (pending[key] = pending[key] || []).push({ seq: seq, undo: undo, body: body });
        // A later save of the same frame carries this edit too: its outcome decides
        const superseded = function () { return mySession !== session || seqByFrame[key] !== seq; };
        const accept = function () {
            if (mySession !== session) return true;
            confirmed[key] = body;
            pending[key] = pending[key].filter(function (e) { return e.seq > seq; });
            return true;
        };
        // The newest pending save whose body the server holds was applied, with all before it
        const reconcile = function (serverState) {
            let applied = 0;
            pending[key].forEach(function (e) { if (sameFrame(e.body, serverState)) applied = e.seq; });
            pending[key].forEach(function (e) { if (e.seq > applied) e.undo(); });
            pending[key] = [];
            confirmed[key] = serverState;
            if (applied === seq) return true;
            onRollback(key, clone(serverState));
            return false;
        };

        const attempt = (queueByFrame[key] || Promise.resolve())
            .then(function () { return put(frameIdx, body); })
            .then(accept)
            .catch(function () {
                if (superseded()) return true;
                return fetchServerFrame(key).then(function (serverState) {
                    return superseded() ? true : reconcile(serverState);
                }, function () {
                    // Server unreachable: the last state it acknowledged is the best guess
                    return superseded() ? true : reconcile(confirmed[key] || emptyFrame());
                });
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
