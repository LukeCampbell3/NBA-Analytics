class PredictionAboutPage {
    constructor() {
        this.data = null;
        this.elements = {
            runFacts: document.getElementById("aboutRunFacts"),
            overview: document.getElementById("accuracyOverview"),
            byTarget: document.getElementById("accuracyByTarget"),
            byDirection: document.getElementById("accuracyByDirection"),
            boardSummary: document.getElementById("boardSummary"),
            backtestOverview: document.getElementById("backtestOverview"),
            backtestPicks: document.getElementById("backtestPicks"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.renderRunFacts();
            this.renderOverview();
            this.renderBoardSummary();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load prediction metadata: ${error.message}`;
            this.elements.overview.innerHTML = '<div class="prediction-about-empty">Board metrics unavailable.</div>';
            this.elements.byTarget.innerHTML = '<div class="prediction-about-empty">No target metrics available.</div>';
            this.elements.byDirection.innerHTML = '<div class="prediction-about-empty">No direction metrics available.</div>';
        }

        // A separate, independent fetch -- real, disclosed historical
        // backtest evidence, not part of today's live board payload, so a
        // failure here must never block the run-facts/board-snapshot
        // sections above.
        try {
            const response = await fetch(`data/v14_backtest_validation.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.backtestData = await response.json();
            this.renderBacktest();
        } catch (error) {
            console.error(error);
            if (this.elements.backtestOverview) {
                this.elements.backtestOverview.innerHTML = '<div class="prediction-about-empty">Backtest evidence unavailable.</div>';
            }
            if (this.elements.backtestPicks) {
                this.elements.backtestPicks.innerHTML = "";
            }
        }
    }

    renderBacktest() {
        const data = this.backtestData || {};
        const summary = data.summary || {};
        const picks = Array.isArray(data.picks) ? data.picks : [];
        const dates = Array.isArray(data.dates_scanned) ? data.dates_scanned : [];
        const dateRange = dates.length
            ? `${this.formatRunStamp(dates[0])} to ${this.formatRunStamp(dates[dates.length - 1])}`
            : "n/a";

        const overviewItems = [
            ["Real Picks", this.formatInt(summary.picks)],
            ["Settled", this.formatInt(summary.settled)],
            ["Wins", this.formatInt(summary.wins)],
            ["Real Hit Rate", this.formatPct(summary.hit_rate)],
            ["Real ROI", this.formatSignedPct(summary.roi)],
            ["Dates Covered", dateRange],
        ];
        if (this.elements.backtestOverview) {
            this.elements.backtestOverview.innerHTML = overviewItems.map(([label, value]) => `
                <article class="prediction-about-metric-card">
                    <span>${this.escapeHtml(label)}</span>
                    <strong>${this.escapeHtml(value)}</strong>
                </article>
            `).join("");
        }

        if (!this.elements.backtestPicks) return;
        if (!picks.length) {
            this.elements.backtestPicks.innerHTML = '<div class="prediction-about-empty">No backtest picks available yet.</div>';
            return;
        }
        const rows = picks.map((pick) => `
            <tr>
                <td>${this.escapeHtml(pick.date)}</td>
                <td>${this.escapeHtml(pick.player)}</td>
                <td>${this.escapeHtml(pick.target)} ${this.escapeHtml(pick.direction)} ${this.escapeHtml(this.formatNum(pick.market_line))}</td>
                <td>${this.escapeHtml(this.formatPct(pick.final_hit_probability))}</td>
                <td>${this.escapeHtml(pick.result ? pick.result.toUpperCase() : "unsettled")}</td>
            </tr>
        `).join("");
        this.elements.backtestPicks.innerHTML = `
            <table class="prediction-about-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Player</th>
                        <th>Market</th>
                        <th>Real Prob.</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }

    mountShell() {
        if (!window.CardVaultShell) return;

        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics",
            brandHref: "/",
            sportSlug: "mlb",
            sportAccent: "#087f5b",
            navLinks: [
                { label: "Board", href: "/mlb/predictions/", active: false },
                { label: "Method", href: "/mlb/prediction-about/", active: true },
            ],
            showDisclaimer: true,
        });
    }

    renderRunFacts() {
        const runDate = this.data?.run_date || "n/a";
        const throughDate = this.data?.through_date || "n/a";
        const modelRun = this.data?.model_run_id || "n/a";
        const policy = this.data?.policy_profile || "n/a";
        this.elements.runFacts.innerHTML = `
            <span>Run ${this.escapeHtml(runDate)}</span>
            <span>Data through ${this.escapeHtml(throughDate)}</span>
            <span>Model ${this.escapeHtml(modelRun)}</span>
            <span>Policy ${this.escapeHtml(policy)}</span>
        `;
    }

    renderOverview() {
        const summary = this.data?.summary || {};
        const dailyParlay = this.data?.daily_parlay || {};
        const ticket = dailyParlay?.selected_ticket || {};
        const overviewItems = [
            ["Board Size", this.formatInt(summary.play_count)],
            ["Avg Hit Rate", this.formatPct(summary.avg_expected_hit_rate)],
            ["Avg Graded Hit Rate", this.formatPct(summary.avg_graded_hit_rate)],
            ["Avg Edge", this.formatSignedNum(summary.avg_edge)],
            ["Avg Abs Edge", this.formatNum(summary.avg_abs_edge)],
            ["Avg Value Score", this.formatNum(summary.avg_value_score)],
            ["Avg Precision Score", this.formatNum(summary.avg_precision_score)],
            ["Daily Parlay Legs", this.formatInt(ticket.leg_count)],
            ["Parlay Hit Rate", this.formatPct(ticket.projected_probability)],
            ["Supported Rows", this.formatInt(summary.supported_rows)],
            ["Rows After Filters", this.formatInt(summary.rows_after_filters)],
            ["Rejected Rows", this.formatInt(summary.rejected_rows)],
        ];

        this.elements.overview.innerHTML = overviewItems.map(([label, value]) => `
            <article class="prediction-about-metric-card">
                <span>${this.escapeHtml(label)}</span>
                <strong>${this.escapeHtml(value)}</strong>
            </article>
        `).join("");

        const targetRows = Object.entries(this.data?.by_target || {});
        this.elements.byTarget.innerHTML = this.renderSplitTable(targetRows, "target");

        const directionRows = Object.entries(this.data?.by_direction || {});
        this.elements.byDirection.innerHTML = this.renderSplitTable(directionRows, "direction");
    }

    renderSplitTable(entries, labelKind) {
        if (!entries.length) {
            return '<div class="prediction-about-empty">No split metrics available.</div>';
        }
        const rows = entries.map(([label, bucket]) => `
            <tr>
                <td>${this.escapeHtml(labelKind === "direction" ? label.toUpperCase() : label)}</td>
                <td>${this.escapeHtml(this.formatInt(bucket.count))}</td>
                <td>${this.escapeHtml(this.formatPct(bucket.share))}</td>
            </tr>
        `).join("");
        return `
            <table class="prediction-about-table">
                <thead>
                    <tr>
                        <th>${labelKind === "direction" ? "Direction" : "Target"}</th>
                        <th>Count</th>
                        <th>Share</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }

    renderBoardSummary() {
        const selection = this.data?.selection || {};
        const rejected = this.data?.filter_rejections || {};
        const dailyParlay = this.data?.daily_parlay || {};
        const ticket = dailyParlay?.selected_ticket || {};
        const legValidation = dailyParlay?.validation?.by_leg_count || [];
        const selectedValidation = legValidation.find((row) => Number(row?.leg_count) === Number(ticket?.leg_count));
        const rejectionText = Object.entries(rejected)
            .sort((a, b) => Number(b[1]) - Number(a[1]))
            .slice(0, 4)
            .map(([label, count]) => `${label.replaceAll("_", " ")}: ${count}`)
            .join(", ");

        this.elements.boardSummary.innerHTML = `
            <p>
                <strong>Current board profile:</strong> max ${this.formatInt(selection.top_n)} plays,
                minimum ${this.formatNum(selection.min_abs_edge)} absolute edge,
                minimum ${this.formatPct(selection.min_hit_probability)} estimated hit rate,
                and minimum ${this.formatInt(selection.min_history_rows)} history rows.
            </p>
            <p>
                <strong>Concentration limits:</strong> max ${this.formatInt(selection.max_per_player)} per player,
                ${this.formatInt(selection.max_per_game)} per game, and ${this.formatInt(selection.max_per_team)} per team.
            </p>
            <p>
                <strong>Main filter rejections:</strong> ${this.escapeHtml(rejectionText || "n/a")}.
            </p>
            <p>
                <strong>Daily parlay:</strong> ${this.formatInt(ticket.leg_count)} OVER-only legs at
                ${this.escapeHtml(ticket.sportsbook || "n/a")}, with projected ticket hit rate
                ${this.formatPct(ticket.projected_probability)} and expected return
                ${this.formatPct(ticket.expected_return_per_unit)}.
            </p>
            <p>
                <strong>Synthetic event holdout:</strong> ${selectedValidation
                    ? `${this.formatPct(selectedValidation.fixed_recent_holdout?.hit_rate)} hit rate across ${this.formatInt(selectedValidation.fixed_recent_holdout?.tickets)} recent graded dates; historical book-level prices were unavailable, so this does not claim ROI`
                    : "not available for the selected leg count"}.
            </p>
        `;
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a";
    }

    formatNum(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a";
    }

    formatSignedNum(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}` : "n/a";
    }

    formatRunStamp(value) {
        const match = /^(\d{4})(\d{2})(\d{2})$/.exec(String(value ?? ""));
        return match ? `${match[1]}-${match[2]}-${match[3]}` : String(value ?? "n/a");
    }

    formatSignedPct(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "n/a";
    }

    formatInt(value) {
        return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a";
    }

    escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new PredictionAboutPage();
});
