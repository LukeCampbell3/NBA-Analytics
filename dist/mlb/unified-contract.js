(function (root, factory) {
    const contract = factory();
    if (typeof module === "object" && module.exports) module.exports = contract;
    if (root) root.MlbUnifiedContract = contract;
})(typeof window !== "undefined" ? window : globalThis, function () {
    "use strict";

    async function fetchJson(url, { fetchImpl = fetch, timeoutMs = 8000 } = {}) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetchImpl(url, { cache: "no-store", credentials: "same-origin", signal: controller.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            let payload;
            try { payload = await response.json(); }
            catch (_) { throw new Error("Malformed JSON"); }
            if (!payload || typeof payload !== "object" || Array.isArray(payload) || Object.keys(payload).length === 0) {
                throw new Error("Malformed or empty JSON");
            }
            return payload;
        } catch (error) {
            if (error?.name === "AbortError") throw new Error("Prediction request timed out");
            throw error;
        } finally {
            clearTimeout(timeout);
        }
    }

    function validate(payload, manifest, { today, nowMs = Date.now(), maximumAgeHours = 30 } = {}) {
        if (payload.schema_version !== "unified_mlb_v1") throw new Error("Unified schema mismatch");
        if (!payload.generation_id || !payload.generated_at_utc || !payload.policy_hash || !payload.run_date) throw new Error("Unified artifact is incomplete");
        if (!manifest || payload.policy_hash !== manifest.policy_hash) throw new Error("Unified policy hash mismatch");
        if (today && payload.run_date !== today) throw new Error("Predictions not yet generated for today's slate");
        const generated = Date.parse(payload.generated_at_utc);
        if (!Number.isFinite(generated) || nowMs - generated > maximumAgeHours * 60 * 60 * 1000) throw new Error("Unified artifact is stale");
        if (!Array.isArray(payload.singles) || !payload.parlays || !payload.evidence) throw new Error("Unified artifact contract is incomplete");
        return true;
    }

    async function load({ artifactUrl, manifestUrl, fetchImpl, timeoutMs, today, nowMs } = {}) {
        try {
            const manifest = await fetchJson(manifestUrl, { fetchImpl, timeoutMs });
            const payload = await fetchJson(artifactUrl, { fetchImpl, timeoutMs });
            validate(payload, manifest, { today, nowMs });
            return { state: manifest.active_engine === "unified" ? "PRODUCTION" : "SHADOW", payload, manifest };
        } catch (error) {
            return { state: "LOAD_ERROR", error: error.message };
        }
    }

    return { fetchJson, validate, load };
});
