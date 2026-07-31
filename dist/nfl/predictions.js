class NflModelReport {
    constructor() {
        this.data = null;
        this.elements = {
            runFacts: document.getElementById("runFacts"),
            gate: document.getElementById("gateSummary"),
            overall: document.getElementById("overallMetrics"),
            targets: document.getElementById("targetMetrics"),
            rows: document.getElementById("holdoutRows"),
            marketStatus: document.getElementById("marketReplayStatus"),
            marketMetrics: document.getElementById("marketReplayMetrics"),
            marketBaselines: document.getElementById("marketBaselines"),
            marketWeekly: document.getElementById("marketWeekly"),
        };
        this.marketEvidence = null;
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const [response, marketResponse] = await Promise.all([
                fetch(`data/daily_predictions.json?v=${Date.now()}`),
                fetch(`data/market_validation_summary.json?v=${Date.now()}`),
            ]);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.marketEvidence = marketResponse.ok ? await marketResponse.json() : null;
            this.render();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load NFL model report: ${error.message}`;
        }
    }

    mountShell() {
        if (!window.CardVaultShell) return;
        window.CardVaultShell.mount({
            brandTitle: "Prediction Bounties",
            brandHref: "/",
            sportSlug: "nfl",
            sportAccent: "#7c3aed",
            navLinks: [
                { label: "Model Report", href: "/nfl/predictions/", active: true },
                { label: "Method", href: "/nfl/prediction-about/", active: false },
            ],
            showDisclaimer: true,
        });
    }

    render() {
        const design = this.data.methodology || {};
        const overall = this.data.overall || {};
        const gate = this.data.promotion_gate || {};
        const market = this.marketEvidence || this.data.market_validation || {};
        this.elements.runFacts.innerHTML = `
            <span>Evaluated ${this.escape(design.holdout_season ?? "n/a")}</span>
            <span>${this.escape(this.formatInt(overall.rows))} holdout rows</span>
            <span>Generated ${this.escape(this.data.run_date || "n/a")}</span>
            <span>Mode ${this.escape(this.data.mode || "n/a")}</span>
            <span>Architecture ${this.escape(this.data.architecture?.name || "n/a")}</span>
            <span>Market validation ${this.escape(market.status || "not evaluated")}</span>
        `;
        const passed = gate.status === "passed";
        this.elements.gate.innerHTML = `<p><strong>${passed ? "Passed" : "Research only"}:</strong> ${this.escape(gate.reason || (passed
            ? "every target cleared the repository-defined sample, R², baseline-improvement, and residual-direction checks."
            : "at least one target missed a required validation threshold; the model should remain in research."))}</p>`;
        const cards = [
            ["Holdout Rows", this.formatInt(overall.rows)],
            ["Weighted MAE", `${this.formatNum(overall.weighted_mae, 1)} yd`],
            ["Rolling Baseline MAE", `${this.formatNum(overall.weighted_baseline_mae, 1)} yd`],
            ["MAE Improvement", this.formatPct(overall.weighted_mae_improvement_vs_rolling_baseline)],
            ["Within Tolerance", this.formatPct(overall.weighted_within_tolerance_accuracy)],
        ];
        this.elements.overall.innerHTML = cards.map(([label, value]) => `
            <article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>
        `).join("");
        this.renderTargets();
        this.renderMarketReplay();
        this.renderRows();
    }

    renderTargets() {
        const rows = (this.data.targets || []).map((target) => {
            const m = target.metrics || {};
            return `<tr>
                <td>${this.escape(target.label)}</td>
                <td>${this.escape(this.formatInt(m.rows))}</td>
                <td>${this.escape(this.formatNum(m.mae, 1))}</td>
                <td>${this.escape(this.formatNum(m.rmse, 1))}</td>
                <td>${this.escape(this.formatNum(m.r2, 3))}</td>
                <td>${this.escape(this.formatPct(m.within_tolerance_accuracy))}</td>
                <td>${this.escape(this.formatPct(m.mae_improvement_vs_rolling_baseline))}</td>
            </tr>`;
        }).join("");
        this.elements.targets.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Target</th><th>N</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Within tolerance</th><th>vs baseline</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
    }

    renderMarketReplay() {
        const evidence = this.marketEvidence;
        if (!evidence) {
            this.elements.marketStatus.innerHTML = "<p>Market replay evidence is unavailable.</p>";
            this.elements.marketMetrics.innerHTML = "";
            this.elements.marketBaselines.innerHTML = "";
            this.elements.marketWeekly.innerHTML = "";
            return;
        }
        const final = evidence.final_test || {};
        const policy = evidence.locked_policy || {};
        const deployment = evidence.gates?.deployment || {};
        const stats = evidence.statistical_evidence || {};
        this.elements.marketStatus.innerHTML = `<p><strong>Historical effectiveness passed; deployment ${this.escape(deployment.status || "blocked")}.</strong> ${this.escape(deployment.reason || "The source-timing gate remains unresolved.")}</p>`;
        const cards = [
            ["Validated Market", (evidence.validated_targets || []).join(", ") || "n/a"],
            ["Weekly Cap", this.formatInt(policy.weekly_top_n)],
            ["Record", `${this.formatInt(final.wins)}–${this.formatInt(final.losses)}`],
            ["Hit Rate", this.formatPct(final.hit_rate)],
            ["ROI", this.formatSignedPct(final.roi)],
            ["Profit", `${this.formatSignedNum(final.profit_units, 2)}u`],
            ["Clustered Hit 95%", this.formatRange(stats.week_cluster_hit_rate_95, false)],
            ["Clustered ROI 95%", this.formatRange(stats.week_cluster_roi_95, true)],
        ];
        this.elements.marketMetrics.innerHTML = cards.map(([label, value]) => `
            <article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>
        `).join("");

        const baselines = evidence.baselines || {};
        const baselineRows = [
            ["Production selector", final],
            ["Always under", baselines.always_under || {}],
            ["Point-projection side", baselines.point_projection_side || {}],
        ].map(([label, row]) => `<tr>
            <td>${this.escape(label)}</td>
            <td>${this.escape(this.formatInt(row.graded_decisions))}</td>
            <td>${this.escape(`${this.formatInt(row.wins)}–${this.formatInt(row.losses)}`)}</td>
            <td>${this.escape(this.formatPct(row.hit_rate))}</td>
            <td>${this.escape(this.formatSignedPct(row.roi))}</td>
        </tr>`).join("");
        this.elements.marketBaselines.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Policy</th><th>N</th><th>Record</th><th>Hit rate</th><th>ROI</th></tr></thead>
            <tbody>${baselineRows}</tbody>
        </table>`;

        const weeklyRows = (evidence.weekly || []).map((row) => `<tr>
            <td>W${this.escape(this.formatInt(row.week))}</td>
            <td>${this.escape(this.formatInt(row.picks))}</td>
            <td>${this.escape(`${this.formatInt(row.wins)}–${this.formatInt(row.losses)}`)}</td>
            <td>${this.escape(this.formatPct(row.hit_rate))}</td>
            <td>${this.escape(this.formatSignedPct(row.roi))}</td>
            <td>${this.escape(`${this.formatSignedNum(row.profit_units, 2)}u`)}</td>
        </tr>`).join("");
        this.elements.marketWeekly.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Week</th><th>Picks</th><th>Record</th><th>Hit rate</th><th>ROI</th><th>Units</th></tr></thead>
            <tbody>${weeklyRows}</tbody>
        </table>`;
    }

    renderRows() {
        const rows = (this.data.plays || []).map((row) => `<tr>
            <td>${this.escape(row.player)}</td>
            <td>${this.escape(String(row.target || "").toUpperCase())}</td>
            <td>${this.escape(`${row.team || "?"} vs ${row.opponent || "?"}`)}</td>
            <td>${this.escape(`${row.season} W${row.week}`)}</td>
            <td>${this.escape(this.formatNum(row.prediction, 1))}</td>
            <td>${this.escape(this.formatNum(row.actual, 1))}</td>
            <td>${this.escape(this.formatNum(row.absolute_error, 1))}</td>
        </tr>`).join("");
        this.elements.rows.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Player</th><th>Target</th><th>Matchup</th><th>Game</th><th>Prediction</th><th>Actual</th><th>Abs error</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
    }

    formatPct(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedPct(value) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedNum(value, places = 2) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(places)}` : "n/a"; }
    formatRange(values, signed = false) {
        if (!Array.isArray(values) || values.length !== 2) return "n/a";
        const formatter = signed ? this.formatSignedPct.bind(this) : this.formatPct.bind(this);
        return `${formatter(values[0])}–${formatter(values[1])}`;
    }
    formatNum(value, places = 2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "n/a"; }
    formatInt(value) { return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a"; }
    escape(value) {
        return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => new NflModelReport());
