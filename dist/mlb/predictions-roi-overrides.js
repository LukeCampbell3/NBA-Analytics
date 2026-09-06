// Additive frontend corrections for the tighter MLB publication policies.
// Loaded after predictions.js and before DOMContentLoaded fires, so these
// prototype overrides apply without duplicating the full board implementation.
(() => {
    if (typeof DailyPredictionsPage === "undefined") return;

    // Number(null) === 0 in JavaScript. Missing no-vig data from a one-sided
    // alternate market is unknown, not 0%, so preserve missingness explicitly.
    DailyPredictionsPage.prototype.formatPct = function formatPct(value) {
        if (value === null || value === undefined || value === "") return "n/a";
        const number = Number(value);
        return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "n/a";
    };

    DailyPredictionsPage.prototype.renderPitcherKLeg = function renderPitcherKLeg(leg, index) {
        if (!leg || !window.CardVault) return "";
        const name = String(leg.pitcher_name || "").trim() || "Unknown pitcher";
        const nameParts = name.split(/\s+/).filter(Boolean);
        const monogram = nameParts.length >= 2
            ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase()
            : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
        const side = leg.side === "under" ? "Under" : "Over";
        const matchup = [leg.team, leg.opponent].filter(Boolean).join(" vs. ");
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: `${side} ${this.formatNumber(leg.line, 1)} Strikeouts`,
            context: matchup,
            metrics: [
                ["Probability", this.formatPct(leg.model_probability)],
                ["Leg EV", this.formatSignedPct(leg.expected_value_per_unit)],
                ["Odds", this.formatAmerican(leg.price_american)],
                ["Book", leg.sportsbook || ""],
            ],
            betslipUrl: leg.sportsbook_deeplink || "",
            deeplinksByRegion: leg.deeplinks_by_region || null,
            settlementRow: leg,
        });
    };

    // PARLAY_POLICY_V2 remains frozen for its research stream. The new public
    // quality overlay can only WITHHOLD presentation; it never changes the
    // recorded frozen action. Show that distinction explicitly.
    const originalRenderParlayV2 = DailyPredictionsPage.prototype.renderParlayV2;
    DailyPredictionsPage.prototype.renderParlayV2 = function renderParlayV2TightQuality() {
        const parlay = this.data?.parlays || {};
        const quality = parlay.public_quality_overlay;
        if (!quality || quality.action !== "ABSTAIN") {
            return originalRenderParlayV2.call(this);
        }

        const content = this.elements?.parlayV2Content;
        if (!content) return;
        const pair = parlay.selected_parlay || parlay.shadow_candidate;
        const legs = pair ? [pair.leg_1, pair.leg_2].filter(Boolean) : [];
        const joint = this.formatPct(quality.joint_probability);
        const ev = this.formatSignedPct(quality.expected_value_per_unit);
        const price = this.formatNumber(quality.combined_decimal_price, 2);
        const reasons = Array.isArray(quality.blocking_reasons)
            ? quality.blocking_reasons.map((reason) => String(reason).replaceAll("_", " ")).join("; ")
            : "tight quality gates failed";

        content.innerHTML = `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("withheld", "Tight-quality abstain") : ""}
                ${window.CardVault ? window.CardVault.renderParlaySettlementBadge(legs) : ""}
            </div>
            <p class="daily-parlay__empty">The frozen V2 research decision is preserved for prospective evidence, but this pair is withheld from the tighter public candidate set.</p>
            ${pair ? this.renderParlayV2Legs(pair) : ""}
            <div class="same-game-parlay__metrics">
                <span>Research joint probability <strong>${joint}</strong></span>
                <span>Combined decimal price <strong>${price}</strong></span>
                <span>Price-adjusted model EV <strong>${ev}</strong></span>
            </div>
            <p class="daily-parlay__state">Tight gate: ${this.escapeHtml(reasons)}. Frozen policy action was not modified.</p>
        `;
    };

    // The same-game combo pipeline now publishes real, priced combos under
    // a NEW field (game.exploratory_ev_candidates) once none clear the
    // tighter headline gate (game.combo_candidates then stays empty by
    // design -- see same_game_quality_selector.py). The base renderer only
    // ever knew about combo_candidates, so a real slate with real priced
    // combos was rendering as "no combo cleared pricing" even though real
    // exploratory candidates existed. This shows the best real exploratory
    // candidate, honestly labeled as not clearing the tight quality gate,
    // reusing the existing renderSameGameCombo() card layout verbatim.
    const originalRenderSameGameParlay = DailyPredictionsPage.prototype.renderSameGameParlay;
    DailyPredictionsPage.prototype.renderSameGameParlay = function renderSameGameParlayWithExploratory() {
        const content = this.elements?.sameGameParlayContent;
        const data = this.sameGameData;
        if (!content || !data || data.status !== "ok" || !Array.isArray(data.games) || !data.games.length) {
            return originalRenderSameGameParlay.call(this);
        }

        const games = data.games;
        const headlineCombos = [];
        for (const game of games) {
            if (Array.isArray(game.combo_candidates)) {
                for (const combo of game.combo_candidates) headlineCombos.push({ game, combo });
            }
        }
        if (headlineCombos.length) {
            return originalRenderSameGameParlay.call(this);
        }

        const exploratory = [];
        for (const game of games) {
            if (Array.isArray(game.exploratory_ev_candidates)) {
                for (const combo of game.exploratory_ev_candidates) exploratory.push({ game, combo });
            }
        }

        const authorizedCount = Number(data.candidate_authorized_count) || 0;
        const exploratoryCount = Number(data.exploratory_candidate_count) || exploratory.length;
        const policyName = (data.selection_policy && typeof data.selection_policy === "object")
            ? (data.selection_policy.name || "same_game_quality_v1")
            : (data.selection_policy || "same_game_quality_v1");
        const statusFooter = `Policy: ${this.escapeHtml(policyName)} / Games scheduled: ${games.length} / Exploratory candidates: ${exploratoryCount} / Authorized: ${authorizedCount}`;

        if (!exploratory.length) {
            const reason = data.odds_status && data.odds_status !== "success"
                ? "Live market odds not yet available for today's slate."
                : "No real cross-market combo cleared pricing for today's slate.";
            content.innerHTML = this.sameGameParlayHeader() + `
                <p class="daily-parlay__empty">${this.escapeHtml(reason)} A same-game combo will appear here once real moneyline/total/F5 lines are posted and priced.</p>
                <p class="daily-parlay__state">${this.escapeHtml(statusFooter)}</p>
            `;
            return;
        }

        exploratory.sort((a, b) => (Number(b.combo.expected_value_per_unit) ?? -Infinity) - (Number(a.combo.expected_value_per_unit) ?? -Infinity));
        const best = exploratory[0];
        const extraCount = exploratory.length - 1;

        content.innerHTML = this.sameGameParlayHeader() + `
            <p class="daily-parlay__empty">No combo cleared today's headline gate (&ge;3pp edge, &ge;5% synthetic EV) -- this is the best real priced combo below that bar, shown for transparency, not as a published pick.</p>
            <div class="same-game-parlay__grid">${this.renderSameGameCombo(best.game, best.combo)}</div>
            ${extraCount > 0 ? `<p class="daily-parlay__state">+${extraCount} more real exploratory combo${extraCount === 1 ? "" : "s"} priced across today's slate</p>` : ""}
            <p class="daily-parlay__state">${this.escapeHtml(statusFooter)}</p>
        `;
    };

    // HIGH_HIT_PARLAY_ROI_V2 now includes price and EV gates. Extend its
    // footer so users can see that "high-hit" is no longer synonymous with
    // accepting tiny payouts.
    const originalRenderHighHitParlay = DailyPredictionsPage.prototype.renderHighHitParlay;
    DailyPredictionsPage.prototype.renderHighHitParlay = function renderHighHitParlayRoiAware() {
        originalRenderHighHitParlay.call(this);
        const content = this.elements?.highHitParlayContent;
        const construction = this.highHitParlayData?.construction || {};
        if (!content || !this.highHitParlayData) return;
        const state = content.querySelector(".daily-parlay__state:last-child");
        if (!state) return;
        const minPrice = Number(construction.min_combined_decimal_price);
        const minEv = Number(construction.min_expected_value_per_unit);
        if (Number.isFinite(minPrice) && Number.isFinite(minEv)) {
            state.textContent += ` / Min payout: ${minPrice.toFixed(2)} / Min EV: ${(minEv * 100).toFixed(1)}%`;
        }
    };
})();

// Keep the last published MLB board visible alongside the live board. The
// preservation pipeline maintains data/history/<run_date>.json as the richest
// board users actually saw, so later inference/refresh runs cannot erase it.
(() => {
    const fetchJson = async (path) => {
        const separator = path.includes("?") ? "&" : "?";
        const response = await fetch(`${path}${separator}v=${Date.now()}`, {
            cache: "no-store",
            credentials: "same-origin",
            headers: { "Cache-Control": "no-cache" },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
            throw new Error("Malformed MLB history payload");
        }
        return payload;
    };

    const escapeHtml = (value) => String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

    const playKey = (play) => [
        play?.issuance_id,
        play?.player_id,
        play?.player,
        play?.game_id,
        play?.target || play?.market_type,
        play?.direction || play?.side,
        play?.line ?? play?.market_line,
    ].filter((value) => value !== null && value !== undefined && value !== "").join("|");

    const boardFingerprint = (payload) => {
        const plays = Array.isArray(payload?.plays) ? payload.plays : [];
        return plays.map(playKey).sort().join("||");
    };

    const hasDistinctPicks = (candidate, current) => {
        const candidatePlays = Array.isArray(candidate?.plays) ? candidate.plays : [];
        if (!candidatePlays.length) return false;
        const currentPlays = Array.isArray(current?.plays) ? current.plays : [];
        if (candidatePlays.length !== currentPlays.length) return true;
        return boardFingerprint(candidate) !== boardFingerprint(current);
    };

    const easternDate = () => {
        const parts = new Intl.DateTimeFormat("en-US", {
            timeZone: "America/New_York",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
        }).formatToParts(new Date());
        const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
        return `${values.year}-${values.month}-${values.day}`;
    };

    const findPreviousBoard = async (current) => {
        const runDate = String(current?.run_date || "").trim();
        const today = easternDate();

        // Same-day comparison only applies when the live artifact is actually
        // today's slate. A stale live file is itself historical evidence and
        // must not cause its date to be skipped.
        if (runDate && runDate === today) {
            try {
                const sameDay = await fetchJson(`data/history/${runDate}.json`);
                if (hasDistinctPicks(sameDay, current)) {
                    return { payload: sameDay, sameDay: true };
                }
            } catch (_) { /* same-day archive may not exist yet */ }
        }

        try {
            const index = await fetchJson("data/history/index.json");
            const dates = Array.isArray(index?.dates)
                ? index.dates.map(String).filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date)).sort().reverse()
                : [];
            for (const date of dates) {
                // Previous is chronological relative to today, not relative
                // to a potentially stale daily_predictions.json run_date.
                if (!date || date >= today) continue;
                try {
                    const archived = await fetchJson(`data/history/${date}.json`);
                    if (Array.isArray(archived?.plays) && archived.plays.length) {
                        return { payload: archived, sameDay: false };
                    }
                } catch (_) { /* try the next preserved date */ }
            }
        } catch (_) { /* history index is optional */ }

        // If the index itself is stale, prefer the older live artifact over
        // jumping back another calendar day.
        if (runDate && runDate < today && Array.isArray(current?.plays) && current.plays.length) {
            return { payload: current, sameDay: false };
        }
        return null;
    };

    const settlementBucket = (play) => {
        const status = String(play?.settlement_status || play?.outcome || play?.result || "pending").trim().toLowerCase();
        if (["won", "win", "w"].includes(status)) return "won";
        if (["lost", "loss", "lose", "l"].includes(status)) return "lost";
        if (["push", "pushed", "tie", "void"].includes(status)) return "push";
        return "pending";
    };

    const resultSummary = (plays) => {
        const counts = { won: 0, lost: 0, push: 0, pending: 0 };
        for (const play of plays) counts[settlementBucket(play)] += 1;
        return `${counts.won}W · ${counts.lost}L · ${counts.push} Push · ${counts.pending} Pending`;
    };

    const archiveLabel = (payload, sameDay) => {
        if (sameDay) return "Earlier today";
        const date = String(payload?.run_date || "");
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return date || "Previous slate";
        try {
            return new Date(`${date}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
        } catch (_) {
            return date;
        }
    };

    const sanitizedArchivedPlay = (play) => ({
        ...play,
        sportsbook_deeplink: "",
        deeplinks_by_region: null,
        betslip_url: "",
        board_publication_status: "archived",
    });

    const fallbackCard = (play) => {
        const name = play?.player_display_name || play?.player || "MLB pick";
        const side = String(play?.direction || play?.side || "").toUpperCase();
        const line = play?.market_line ?? play?.line ?? "";
        const market = play?.target || play?.market_type || "";
        const status = settlementBucket(play);
        const actual = play?.settlement_actual_value;
        const result = status === "pending"
            ? "Pending"
            : `${status === "won" ? "Won" : status === "lost" ? "Lost" : "Push"}${actual === null || actual === undefined ? "" : ` · Actual ${actual}`}`;
        return `<article class="prediction-card"><h3>${escapeHtml(name)}</h3><p class="prediction-card__market">${escapeHtml(`${side} ${line} ${market}`.trim())}</p><p>${escapeHtml(result)}</p></article>`;
    };

    const renderPreviousBoard = (record) => {
        const board = document.getElementById("board");
        if (!board || !record?.payload) return;
        const payload = record.payload;
        const plays = Array.isArray(payload.plays) ? payload.plays : [];
        if (!plays.length) return;

        let section = document.getElementById("previousPublishedPicks");
        if (!section) {
            section = document.createElement("section");
            section.id = "previousPublishedPicks";
            section.className = "board-section";
            section.setAttribute("aria-labelledby", "previousPublishedPicksHeading");
            board.insertAdjacentElement("afterend", section);
        }

        const label = archiveLabel(payload, record.sameDay);
        section.innerHTML = `
            <h2 id="previousPublishedPicksHeading" class="board-section__heading">Previous Published Picks</h2>
            <p class="parlay-group__note"><strong>${escapeHtml(label)}</strong> · ${escapeHtml(resultSummary(plays))}</p>
            <p class="parlay-group__note">Preserved publication snapshot. Results update as games are settled; archived sportsbook links are disabled.</p>
            <div id="previousPredictionCards" class="vault-board"></div>
        `;

        const target = section.querySelector("#previousPredictionCards");
        if (!target) return;
        if (window.CardVault?.renderPredictionCard) {
            target.innerHTML = plays
                .map((play, index) => window.CardVault.renderPredictionCard(sanitizedArchivedPlay(play), index))
                .join("");
        } else {
            target.innerHTML = plays.map(fallbackCard).join("");
        }
    };

    const loadPreviousBoard = async () => {
        let current = {};
        try {
            current = await fetchJson("data/daily_predictions.json");
        } catch (_) { /* previous results should survive a current-board load failure */ }
        const record = await findPreviousBoard(current);
        if (record) renderPreviousBoard(record);
    };

    const start = () => window.setTimeout(() => {
        loadPreviousBoard().catch((error) => console.warn("Previous MLB picks unavailable", error));
    }, 0);

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, { once: true });
    } else {
        start();
    }
})();
