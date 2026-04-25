class PredictionAboutPage {
    constructor() {
        this.data = null;
        this.elements = {
            runFacts: document.getElementById("aboutRunFacts"),
            overview: document.getElementById("accuracyOverview"),
            byTarget: document.getElementById("accuracyByTarget"),
            byDirection: document.getElementById("accuracyByDirection"),
            boardSummary: document.getElementById("boardSummary"),
        };
        this.init();
    }

    async init() {
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
    }

    renderRunFacts() {
        const runDate = this.data?.run_date || "n/a";
        const throughDate = this.data?.through_date || "n/a";
        const modelRun = this.data?.model_run_id || "n/a";
        const policy = this.data?.policy_profile || "n/a";
        this.elements.runFacts.textContent = `Run ${runDate} | Data through ${throughDate} | Model ${modelRun} | Policy ${policy}`;
    }

    renderOverview() {
        const summary = this.data?.summary || {};
        const overviewItems = [
            ["Board Size", this.formatInt(summary.play_count)],
            ["Avg Hit Rate", this.formatPct(summary.avg_expected_hit_rate)],
            ["Avg Graded Hit Rate", this.formatPct(summary.avg_graded_hit_rate)],
            ["Avg Abs Edge", this.formatNum(summary.avg_abs_edge)],
            ["Avg Precision Score", this.formatNum(summary.avg_precision_score)],
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
        const rejectionText = Object.entries(rejected)
            .sort((a, b) => Number(b[1]) - Number(a[1]))
            .slice(0, 4)
            .map(([label, count]) => `${label.replaceAll("_", " ")}: ${count}`)
            .join(" | ");

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
        `;
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a";
    }

    formatNum(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a";
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
