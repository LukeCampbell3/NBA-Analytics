// Additive frontend corrections for the tighter MLB publication policies.
(() => {
    if (typeof DailyPredictionsPage === "undefined") return;

    DailyPredictionsPage.prototype.formatPct = function formatPct(value) {
        if (value === null || value === undefined || value === "") return "n/a";
        const number = Number(value);
        return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "n/a";
    };

    DailyPredictionsPage.prototype.renderPitcherKLeg = function renderPitcherKLeg(leg, index) {
        if (!leg || !window.CardVault) return "";
        const name = String(leg.pitcher_name || "").trim() || "Unknown pitcher";
        const nameParts = name.split(/\s+/).filter(Boolean);
        const monogram = nameParts.length >= 2 ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase() : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
        const side = leg.side === "under" ? "Under" : "Over";
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: `${side} ${this.formatNumber(leg.line, 1)} Strikeouts`,
            context: [leg.team, leg.opponent].filter(Boolean).join(" vs. "),
            metrics: [["Probability", this.formatPct(leg.model_probability)], ["Leg EV", this.formatSignedPct(leg.expected_value_per_unit)], ["Odds", this.formatAmerican(leg.price_american)], ["Book", leg.sportsbook || ""]],
            betslipUrl: leg.sportsbook_deeplink || "",
            deeplinksByRegion: leg.deeplinks_by_region || null,
            settlementRow: leg,
        });
    };

    const originalRenderParlayV2 = DailyPredictionsPage.prototype.renderParlayV2;
    DailyPredictionsPage.prototype.renderParlayV2 = function renderParlayV2TightQuality() {
        const parlay = this.data?.parlays || {};
        const quality = parlay.public_quality_overlay;
        if (!quality || quality.action !== "ABSTAIN") return originalRenderParlayV2.call(this);
        const content = this.elements?.parlayV2Content;
        if (!content) return;
        const pair = parlay.selected_parlay || parlay.shadow_candidate;
        const legs = pair ? [pair.leg_1, pair.leg_2].filter(Boolean) : [];
        const reasons = Array.isArray(quality.blocking_reasons) ? quality.blocking_reasons.map((reason) => String(reason).replaceAll("_", " ")).join("; ") : "tight quality gates failed";
        content.innerHTML = `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("withheld", "Tight-quality abstain") : ""}
                ${window.CardVault ? window.CardVault.renderParlaySettlementBadge(legs) : ""}
            </div>
            <p class="daily-parlay__empty">The frozen V2 research decision is preserved for prospective evidence, but this pair is withheld from the tighter public candidate set.</p>
            ${pair ? this.renderParlayV2Legs(pair) : ""}
            <div class="same-game-parlay__metrics">
                <span>Research joint probability <strong>${this.formatPct(quality.joint_probability)}</strong></span>
                <span>Combined decimal price <strong>${this.formatNumber(quality.combined_decimal_price, 2)}</strong></span>
                <span>Price-adjusted model EV <strong>${this.formatSignedPct(quality.expected_value_per_unit)}</strong></span>
            </div>
            <p class="daily-parlay__state">Tight gate: ${this.escapeHtml(reasons)}. Frozen policy action was not modified.</p>`;
    };

    const originalRenderHighHitParlay = DailyPredictionsPage.prototype.renderHighHitParlay;
    DailyPredictionsPage.prototype.renderHighHitParlay = function renderHighHitParlayRoiAware() {
        originalRenderHighHitParlay.call(this);
        const state = this.elements?.highHitParlayContent?.querySelector(".daily-parlay__state:last-child");
        const construction = this.highHitParlayData?.construction || {};
        if (!state) return;
        const minPrice = Number(construction.min_combined_decimal_price);
        const minEv = Number(construction.min_expected_value_per_unit);
        if (Number.isFinite(minPrice) && Number.isFinite(minEv)) state.textContent += ` / Min payout: ${minPrice.toFixed(2)} / Min EV: ${(minEv * 100).toFixed(1)}%`;
    };
})();
