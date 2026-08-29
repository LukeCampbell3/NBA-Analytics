// Additive frontend corrections for the ROI-oriented MLB pitcher parlay.
// Loaded after predictions.js and before DOMContentLoaded fires, so these
// prototype overrides apply to the page instance created by predictions.js
// without duplicating or rewriting the full board implementation.
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
})();
