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

// Full previous-board renderer. This lives in an asset the static builder
// already guarantees for every MLB prediction route, and is browser-only so
// the CommonJS contract above remains unchanged for tests/tooling.
(() => {
    if (typeof document === "undefined" || typeof window === "undefined") return;

    const fetchJson = async (path) => {
        const separator = path.includes("?") ? "&" : "?";
        const response = await fetch(`${path}${separator}v=${Date.now()}`, {
            cache: "no-store",
            credentials: "same-origin",
            headers: { "Cache-Control": "no-cache" },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("Malformed archive payload");
        return payload;
    };

    const playKey = (play) => [
        play?.issuance_id,
        play?.player_id,
        play?.player,
        play?.game_id,
        play?.target || play?.market_type,
        play?.direction || play?.side,
        play?.line ?? play?.market_line,
    ].filter((value) => value !== null && value !== undefined && value !== "").join("|");

    const fingerprint = (payload) => (Array.isArray(payload?.plays) ? payload.plays : [])
        .map(playKey)
        .sort()
        .join("||");

    const distinctFromCurrent = (candidate, current) => {
        const archived = Array.isArray(candidate?.plays) ? candidate.plays : [];
        if (!archived.length) return false;
        const live = Array.isArray(current?.plays) ? current.plays : [];
        return archived.length !== live.length || fingerprint(candidate) !== fingerprint(current);
    };

    const findPrevious = async (current) => {
        const runDate = String(current?.run_date || "").trim();
        if (runDate) {
            try {
                const sameDay = await fetchJson(`data/history/${runDate}.json`);
                if (distinctFromCurrent(sameDay, current)) return { payload: sameDay, date: runDate, sameDay: true };
            } catch (_) { /* no same-day preserved board yet */ }
        }
        try {
            const index = await fetchJson("data/history/index.json");
            for (const date of (Array.isArray(index?.dates) ? index.dates : [])) {
                const token = String(date || "");
                if (!token || token === runDate) continue;
                try {
                    const payload = await fetchJson(`data/history/${token}.json`);
                    if (Array.isArray(payload?.plays) && payload.plays.length) return { payload, date: token, sameDay: false };
                } catch (_) { /* try older preserved slate */ }
            }
        } catch (_) { /* no archive index */ }
        return null;
    };

    const stripArchivedLinks = (value) => {
        if (Array.isArray(value)) return value.map(stripArchivedLinks);
        if (!value || typeof value !== "object") return value;
        const result = {};
        for (const [key, child] of Object.entries(value)) {
            if (["sportsbook_deeplink", "betslip_url", "deeplinks_by_region"].includes(key)) {
                result[key] = key === "deeplinks_by_region" ? null : "";
            } else {
                result[key] = stripArchivedLinks(child);
            }
        }
        return result;
    };

    const loadProduct = async (date, filename) => {
        try {
            return await fetchJson(`data/history/products/${date}/${filename}`);
        } catch (_) {
            return null;
        }
    };

    const labelFor = (record) => {
        if (record.sameDay) return "Earlier today";
        const date = record.date;
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return date || "Previous slate";
        return new Date(`${date}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    };

    const renderProducts = async (record) => {
        if (typeof DailyPredictionsPage === "undefined") return;
        const anchor = document.getElementById("previousPublishedPicks") || document.getElementById("board");
        if (!anchor) return;

        const [sameGame, pitcher, highHit, exotic] = await Promise.all([
            loadProduct(record.date, "same_game_predictions.json"),
            loadProduct(record.date, "pitcher_parlay_predictions.json"),
            loadProduct(record.date, "high_hit_parlay_predictions.json"),
            loadProduct(record.date, "exotic_market_predictions.json"),
        ]);

        let section = document.getElementById("previousPublishedProducts");
        if (!section) {
            section = document.createElement("section");
            section.id = "previousPublishedProducts";
            section.className = "board-section";
            anchor.insertAdjacentElement("afterend", section);
        }
        section.innerHTML = `
            <h2 class="board-section__heading">Previous Parlays &amp; Exotic Picks</h2>
            <p class="parlay-group__note"><strong>${labelFor(record)}</strong> · preserved product snapshots · archived sportsbook links disabled</p>
            <div class="parlay-group">
                <p class="parlay-group__label">Cross-Market</p>
                <div id="previousParlayV2Content" class="daily-parlay"></div>
            </div>
            <div class="parlay-group">
                <p class="parlay-group__label">Same-Game</p>
                <div id="previousSameGameParlayContent" class="daily-parlay"></div>
            </div>
            <div class="parlay-group">
                <p class="parlay-group__label">Pitchers-Only</p>
                <div id="previousPitcherParlayContent" class="daily-parlay"></div>
            </div>
            <div class="parlay-group">
                <p class="parlay-group__label">High-Hit</p>
                <div id="previousHighHitParlayContent" class="daily-parlay"></div>
            </div>
            <div class="parlay-group">
                <p class="parlay-group__label">Exotic Picks</p>
                <div id="previousExoticMarketsContent" class="daily-parlay"></div>
            </div>
        `;

        const renderer = Object.create(DailyPredictionsPage.prototype);
        renderer.data = stripArchivedLinks(record.payload);
        renderer.sameGameData = stripArchivedLinks(sameGame);
        renderer.pitcherParlayData = stripArchivedLinks(pitcher);
        renderer.highHitParlayData = stripArchivedLinks(highHit);
        renderer.exoticMarketsData = stripArchivedLinks(exotic);
        renderer.elements = {
            parlayV2Content: section.querySelector("#previousParlayV2Content"),
            sameGameParlayContent: section.querySelector("#previousSameGameParlayContent"),
            pitcherParlayContent: section.querySelector("#previousPitcherParlayContent"),
            highHitParlayContent: section.querySelector("#previousHighHitParlayContent"),
            exoticMarketsContent: section.querySelector("#previousExoticMarketsContent"),
        };

        try { renderer.renderParlayV2(); } catch (error) { console.warn("Previous cross-market parlay render failed", error); }
        try { renderer.renderSameGameParlay(); } catch (error) { console.warn("Previous same-game parlay render failed", error); }
        try { renderer.renderPitcherParlay(); } catch (error) { console.warn("Previous pitcher parlay render failed", error); }
        try { renderer.renderHighHitParlay(); } catch (error) { console.warn("Previous high-hit parlay render failed", error); }
        try { renderer.renderExoticMarkets(); } catch (error) { console.warn("Previous exotic render failed", error); }
    };

    const start = () => window.setTimeout(async () => {
        try {
            let current = {};
            try { current = await fetchJson("data/daily_predictions.json"); } catch (_) { /* retain history even if live load fails */ }
            const record = await findPrevious(current);
            if (record) await renderProducts(record);
        } catch (error) {
            console.warn("Previous MLB product board unavailable", error);
        }
    }, 50);

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
    else start();
})();
