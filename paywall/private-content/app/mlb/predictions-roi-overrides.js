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
            <p class="daily-parlay__empty">No combo cleared today's tight headline gate (&ge;50% joint probability, &ge;3pp edge, &ge;5% synthetic EV) -- this is the best real priced combo below that bar, shown for transparency, not as a published pick.</p>
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
