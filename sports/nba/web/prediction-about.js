class PredictionAboutPage {
    constructor() {
        this.data = null;
        this.elements = {
            runFacts: document.getElementById('aboutRunFacts'),
            overview: document.getElementById('accuracyOverview'),
            byTarget: document.getElementById('accuracyByTarget'),
            byDirection: document.getElementById('accuracyByDirection'),
            boardSummary: document.getElementById('boardSummary'),
        };
        this.init();
    }

    async init() {
        try {
            const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.renderRunFacts();
            this.renderAccuracy();
            this.renderBoardSummary();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load prediction metadata: ${error.message}`;
            this.elements.overview.innerHTML = '<div class="prediction-about-empty">Accuracy metrics unavailable.</div>';
            this.elements.byTarget.innerHTML = '<div class="prediction-about-empty">No target metrics available.</div>';
            this.elements.byDirection.innerHTML = '<div class="prediction-about-empty">No direction metrics available.</div>';
        }
    }

    renderRunFacts() {
        const runDate = this.data?.run_date || 'n/a';
        const throughDate = this.data?.through_date || 'n/a';
        const modelRun = this.data?.model_run_id || 'n/a';
        const policy = this.data?.policy_profile || 'n/a';
        this.elements.runFacts.textContent = `Run ${runDate} | Data through ${throughDate} | Model ${modelRun} | Policy ${policy}`;
    }

    renderAccuracy() {
        const metrics = this.data?.accuracy_metrics || {};
        if (!metrics.available || !metrics.overall) {
            this.elements.overview.innerHTML = `
                <div class="prediction-about-empty">
                    Accuracy metrics are not available in this export.
                </div>
            `;
            this.elements.byTarget.innerHTML = '<div class="prediction-about-empty">No target metrics available.</div>';
            this.elements.byDirection.innerHTML = '<div class="prediction-about-empty">No direction metrics available.</div>';
            return;
        }

        const overall = metrics.overall || {};
        const parlaySummary = this.data?.parlay_summary || {};
        const overviewItems = [
            ['Win Rate', this.formatPct(overall.win_rate)],
            ['ROI / Graded Play', this.formatSignedPct(overall.roi_per_graded_play)],
            ['Signal Count', this.formatInt(overall.signal_count)],
            ['Graded Plays', this.formatInt(overall.graded_count)],
            ['Push Rate', this.formatPct(overall.push_rate)],
            ['Parlay Tagged Plays', this.formatInt(parlaySummary.tagged_play_count)],
            ['Parlay Pairs', this.formatInt(parlaySummary.selected_pair_count)],
            ['Mean Abs Edge', this.formatNum(overall.mean_abs_edge)],
            ['MAE', this.formatNum(overall.mean_abs_error)],
            ['RMSE', this.formatNum(overall.rmse)],
        ];

        this.elements.overview.innerHTML = overviewItems.map(([label, value]) => `
            <article class="prediction-about-metric-card">
                <span>${this.escapeHtml(label)}</span>
                <strong>${this.escapeHtml(value)}</strong>
            </article>
        `).join('');

        const targetRows = Object.entries(metrics.by_target || {});
        this.elements.byTarget.innerHTML = this.renderSplitTable(targetRows, 'target');

        const directionRows = Object.entries(metrics.by_direction || {});
        this.elements.byDirection.innerHTML = this.renderSplitTable(directionRows, 'direction');
    }

    renderSplitTable(entries, labelKind) {
        if (!entries.length) {
            return '<div class="prediction-about-empty">No split metrics available.</div>';
        }
        const rows = entries.map(([label, bucket]) => `
            <tr>
                <td>${this.escapeHtml(labelKind === 'direction' ? label.toUpperCase() : label)}</td>
                <td>${this.escapeHtml(this.formatPct(bucket.win_rate))}</td>
                <td>${this.escapeHtml(this.formatSignedPct(bucket.roi_per_graded_play))}</td>
                <td>${this.escapeHtml(this.formatInt(bucket.graded_count))}</td>
                <td>${this.escapeHtml(this.formatPct(bucket.push_rate))}</td>
            </tr>
        `).join('');
        return `
            <table class="prediction-about-table">
                <thead>
                    <tr>
                        <th>${labelKind === 'direction' ? 'Direction' : 'Target'}</th>
                        <th>Win Rate</th>
                        <th>ROI/Play</th>
                        <th>Graded</th>
                        <th>Push Rate</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;
    }

    renderBoardSummary() {
        const summary = this.data?.summary || {};
        const policy = this.data?.policy || {};
        const asOf = this.data?.accuracy_metrics?.as_of_market_date || 'n/a';
        const parlaySummary = this.data?.parlay_summary || {};
        const parlayValidation = this.data?.parlay_validation || {};
        this.elements.boardSummary.innerHTML = `
            <p>
                <strong>Current board profile:</strong> ${this.formatInt(summary.play_count)} published plays,
                average expected win rate ${this.formatPct(summary.avg_expected_win_rate)},
                average EV ${this.formatSignedPct(summary.avg_ev)},
                and average edge ${this.formatNum(summary.avg_edge)}.
            </p>
            <p>
                <strong>Policy context:</strong> minimum EV gate ${this.formatSignedPct(policy.min_ev)},
                minimum confidence ${this.formatPct(policy.min_final_confidence)},
                max total plays ${this.formatInt(policy.max_total_plays)}.
            </p>
            <p>
                <strong>Historical snapshot as of ${this.escapeHtml(asOf)}:</strong>
                metrics are recomputed during web export from the pipeline history CSV so this page always reflects the latest available run context.
            </p>
            <p>
                <strong>Parlay screen:</strong> ${this.formatInt(parlaySummary.tagged_play_count)} plays are tagged into
                ${this.formatInt(parlaySummary.selected_pair_count)} suggested 2-leg combos, averaging
                ${this.formatPct(parlaySummary.avg_projected_pair_hit_rate)} projected pair hit rate.
            </p>
            <p>
                <strong>Historical pair validation:</strong> ${parlayValidation.available
                    ? `${this.formatPct(parlayValidation.selected?.pair_hit_rate)} hit rate across ${this.formatInt(parlayValidation.selected?.graded_pair_count)} graded tagged pairs`
                    : this.escapeHtml(parlayValidation.reason || 'not available for this export')}.
            </p>
        `;
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : 'n/a';
    }

    formatSignedPct(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(1)}%` : 'n/a';
    }

    formatNum(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : 'n/a';
    }

    formatInt(value) {
        return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : 'n/a';
    }

    escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new PredictionAboutPage();
});
