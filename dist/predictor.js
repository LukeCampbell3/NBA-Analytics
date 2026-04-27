class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.filtered = [];
        this.elements = {
            status: document.getElementById('predictionStatus'),
            meta: document.getElementById('predictionsMeta'),
            summary: document.getElementById('predictionsSummary'),
            source: document.getElementById('predictionSource'),
            cards: document.getElementById('predictionCards'),
            empty: document.getElementById('predictionEmpty'),
            search: document.getElementById('predictionSearch'),
            target: document.getElementById('predictionTargetFilter'),
            recommendation: document.getElementById('predictionRecommendationFilter'),
        };
        this.init();
    }

    async init() {
        try {
            await this.load();
            this.populateFilters();
            this.bind();
            this.applyFilters();
        } catch (error) {
            console.error(error);
            this.elements.status.textContent = `Unable to load daily predictions: ${error.message}`;
        }
    }

    async load() {
        const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.data = await response.json();
        this.plays = Array.isArray(this.data.plays) ? this.data.plays.slice() : [];
        this.plays.sort((a, b) => {
            const evDiff = (b.ev || 0) - (a.ev || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (b.abs_edge || 0) - (a.abs_edge || 0);
        });
        this.elements.status.textContent = `${this.plays.length} published plays loaded`;
        this.renderMeta();
        this.renderSummary();
        this.renderSource();
    }

    bind() {
        this.elements.search.addEventListener('input', () => this.applyFilters());
        this.elements.target.addEventListener('change', () => this.applyFilters());
        this.elements.recommendation.addEventListener('change', () => this.applyFilters());
    }

    populateFilters() {
        const targets = [...new Set(this.plays.map(play => play.target).filter(Boolean))];
        const recommendations = [...new Set(this.plays.map(play => play.recommendation).filter(Boolean))];
        for (const target of targets) {
            const option = document.createElement('option');
            option.value = target;
            option.textContent = target;
            this.elements.target.appendChild(option);
        }
        for (const recommendation of recommendations) {
            const option = document.createElement('option');
            option.value = recommendation;
            option.textContent = recommendation.toUpperCase();
            this.elements.recommendation.appendChild(option);
        }
    }

    applyFilters() {
        const q = (this.elements.search.value || '').trim().toLowerCase();
        const target = this.elements.target.value;
        const recommendation = this.elements.recommendation.value;
        this.filtered = this.plays.filter(play => {
            const matchesQuery = !q || `${play.player} ${play.target} ${play.direction}`.toLowerCase().includes(q);
            const matchesTarget = !target || play.target === target;
            const matchesRecommendation = !recommendation || play.recommendation === recommendation;
            return matchesQuery && matchesTarget && matchesRecommendation;
        });
        this.renderCards();
    }

    renderMeta() {
        const meta = [];
        if (this.data?.run_date) meta.push(`Run Date ${this.data.run_date}`);
        if (this.data?.through_date) meta.push(`Data Through ${this.data.through_date}`);
        if (this.data?.policy_profile) meta.push(`Policy ${this.data.policy_profile}`);
        if (this.data?.model_run_id) meta.push(`Model ${this.data.model_run_id}`);
        this.elements.meta.innerHTML = meta.map(item => `<span class="prediction-meta-pill">${item}</span>`).join('');
    }

    renderSummary() {
        const summary = this.data?.summary || {};
        const items = [
            ['Published Plays', summary.play_count],
            ['Avg Expected Win Rate', this.formatPct(summary.avg_expected_win_rate)],
            ['Avg EV', this.formatSignedPct(summary.avg_ev)],
            ['Avg Edge', this.formatNumber(summary.avg_edge)],
        ];
        this.elements.summary.innerHTML = items.map(([label, value]) => `
            <article class="prediction-summary-card">
                <span class="prediction-summary-label">${label}</span>
                <strong class="prediction-summary-value">${value ?? 'n/a'}</strong>
            </article>
        `).join('');
    }

    renderSource() {
        const snapshot = this.data?.current_market_snapshot_meta || {};
        const validation = this.data?.input_validation || {};
        const market = validation.market_lines || {};
        const prior = validation.prior_game_data || {};
        const skipped = validation.skipped_rows || {};
        this.elements.source.innerHTML = `
            <div class="prediction-source-grid">
                <div>
                    <h3>Market Snapshot</h3>
                    <p>${snapshot.mode || 'unknown'} • ${snapshot.selected_market_date || snapshot.selected_market_date_min || 'n/a'}</p>
                    <p>${market.market_rows || 0} market rows • PTS ${market.market_pts_lines || 0} • TRB ${market.market_trb_lines || 0} • AST ${market.market_ast_lines || 0}</p>
                </div>
                <div>
                    <h3>Prior Game Data</h3>
                    <p>${prior.slate_rows || 0} player histories • median ${prior.history_rows_median || 0} games</p>
                    <p>${prior.history_before_market_violations || 0} history violations • ${skipped.count || 0} skipped rows</p>
                </div>
            </div>
        `;
    }

    renderCards() {
        this.elements.empty.style.display = this.filtered.length ? 'none' : 'block';
        this.elements.cards.innerHTML = this.filtered.map(play => {
            const tier = play.parlay_candidate ? 'parlay' : (play.recommendation || 'consider').toLowerCase();
            const edgeClass = (play.direction || '').toLowerCase() === 'over' ? 'prediction-pill-over' : 'prediction-pill-under';
            const primaryBadge = play.parlay_candidate ? 'PARLAY' : (play.recommendation || 'consider').toUpperCase();
            return `
                <article class="prediction-card prediction-card-${tier}">
                    <div class="prediction-card-top">
                        <div>
                            <div class="prediction-rank">#${play.rank}</div>
                            <h2>${play.player}</h2>
                            <p class="prediction-card-sub">${play.target} • ${play.market_date || 'n/a'}</p>
                        </div>
                        <div class="prediction-card-badges">
                            <span class="prediction-badge prediction-badge-${tier}">${primaryBadge}</span>
                            <span class="prediction-badge ${edgeClass}">${play.direction}</span>
                        </div>
                    </div>
                    <div class="prediction-main-metric">
                        <div>
                            <span class="prediction-metric-label">Prediction</span>
                            <strong>${this.formatNumber(play.prediction)}</strong>
                        </div>
                        <div>
                            <span class="prediction-metric-label">Market</span>
                            <strong>${this.formatNumber(play.market_line)}</strong>
                        </div>
                        <div>
                            <span class="prediction-metric-label">Edge</span>
                            <strong>${this.formatSignedNumber(play.edge)}</strong>
                        </div>
                    </div>
                    <div class="prediction-stat-grid">
                        ${this.statCell('Expected Win Rate', this.formatPct(play.expected_win_rate))}
                        ${this.statCell('Raw Expected Win', this.formatPct(play.raw_expected_win_rate))}
                        ${this.statCell('EV', this.formatSignedPct(play.ev))}
                        ${this.statCell('Confidence', this.formatPct(play.final_confidence))}
                        ${this.statCell('Percentile', this.formatPct(play.gap_percentile))}
                        ${this.statCell('History Rows', play.history_rows ?? 'n/a')}
                    </div>
                    <div class="prediction-footer">
                        <span>Last history: ${play.last_history_date || 'n/a'}</span>
                        <span>Books: ${play.market_books ?? 'n/a'}</span>
                    </div>
                </article>
            `;
        }).join('');
    }

    statCell(label, value) {
        return `
            <div class="prediction-stat-cell">
                <span>${label}</span>
                <strong>${value}</strong>
            </div>
        `;
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : 'n/a';
    }

    formatSignedPct(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(1)}%` : 'n/a';
    }

    formatNumber(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : 'n/a';
    }

    formatSignedNumber(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(2)}` : 'n/a';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DailyPredictionsPage();
});
