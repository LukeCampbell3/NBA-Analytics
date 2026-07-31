class NflModelReport {
    constructor() {
        this.data = null;
        this.elements = {
            runFacts: document.getElementById("runFacts"),
            gate: document.getElementById("gateSummary"),
            overall: document.getElementById("overallMetrics"),
            targets: document.getElementById("targetMetrics"),
            rows: document.getElementById("holdoutRows"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
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
        this.elements.runFacts.innerHTML = `
            <span>Evaluated ${this.escape(design.holdout_season ?? "n/a")}</span>
            <span>${this.escape(this.formatInt(overall.rows))} holdout rows</span>
            <span>Generated ${this.escape(this.data.run_date || "n/a")}</span>
            <span>Mode ${this.escape(this.data.mode || "n/a")}</span>
        `;
        const passed = gate.status === "passed";
        this.elements.gate.innerHTML = `<p><strong>${passed ? "Passed" : "Not passed"}:</strong> ${passed
            ? "every target cleared the repository-defined sample, R², baseline-improvement, and residual-direction checks."
            : "at least one target missed a required validation threshold; the model should remain in research."}</p>`;
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
    formatNum(value, places = 2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "n/a"; }
    formatInt(value) { return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a"; }
    escape(value) {
        return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => new NflModelReport());
