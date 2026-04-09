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
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || 0) - (Number(a.abs_edge) || 0);
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
        const targets = [...new Set(this.plays.map((play) => play.target).filter(Boolean))];
        const recommendations = [...new Set(this.plays.map((play) => play.recommendation).filter(Boolean))];
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
        this.filtered = this.plays.filter((play) => {
            const displayName = this.getPlayDisplayName(play);
            const matchesQuery = !q || `${displayName} ${play.target || ''} ${play.direction || ''}`.toLowerCase().includes(q);
            const matchesTarget = !target || play.target === target;
            const matchesRecommendation = !recommendation || play.recommendation === recommendation;
            return matchesQuery && matchesTarget && matchesRecommendation;
        });
        this.renderCards();
    }

    getPlayDisplayName(play) {
        const fromPayload = String(play.player_display_name || '').trim();
        if (fromPayload) return fromPayload;
        const fromPlayer = String(play.player || '').replaceAll('_', ' ').trim();
        return fromPlayer || 'Unknown Player';
    }

    getPlayHeadshotUrl(play) {
        const explicitUrl = String(play.player_headshot_url || '').trim();
        if (explicitUrl) return explicitUrl;
        const id = Number(play.player_id);
        if (Number.isFinite(id) && id > 0) {
            return `https://cdn.nba.com/headshots/nba/latest/1040x760/${id}.png`;
        }
        return '';
    }

    getMonogram(name) {
        const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return 'NA';
        if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
        return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
    }

    renderMeta() {
        const meta = [];
        if (this.data?.run_date) meta.push(`Run Date ${this.data.run_date}`);
        if (this.data?.through_date) meta.push(`Data Through ${this.data.through_date}`);
        if (this.data?.policy_profile) meta.push(`Policy ${this.data.policy_profile}`);
        if (this.data?.model_run_id) meta.push(`Model ${this.data.model_run_id}`);
        this.elements.meta.innerHTML = meta.map((item) => `<span class="prediction-meta-pill">${this.escapeHtml(item)}</span>`).join('');
    }

    renderSummary() {
        const summary = this.buildSummaryFromPlays();
        const items = [
            ['Published Plays', summary.play_count],
            ['Avg Expected Win Rate', this.formatPct(summary.avg_expected_win_rate)],
            ['Avg EV', this.formatSignedPct(summary.avg_ev)],
            ['Avg Edge', this.formatNumber(summary.avg_edge)],
        ];
        this.elements.summary.innerHTML = items.map(([label, value]) => `
            <article class="prediction-summary-card">
                <span class="prediction-summary-label">${this.escapeHtml(label)}</span>
                <strong class="prediction-summary-value">${this.escapeHtml(String(value ?? 'n/a'))}</strong>
            </article>
        `).join('');
    }

    buildSummaryFromPlays() {
        const plays = Array.isArray(this.plays) ? this.plays : [];
        if (!plays.length) {
            return this.data?.summary || {
                play_count: 0,
                avg_expected_win_rate: null,
                avg_ev: null,
                avg_edge: null,
            };
        }
        const toNum = (value) => {
            const num = Number(value);
            return Number.isFinite(num) ? num : null;
        };
        const avg = (values) => {
            const nums = values.map(toNum).filter((value) => value !== null);
            if (!nums.length) return null;
            return nums.reduce((acc, value) => acc + value, 0) / nums.length;
        };
        return {
            play_count: plays.length,
            avg_expected_win_rate: avg(plays.map((play) => play.expected_win_rate)),
            avg_ev: avg(plays.map((play) => play.ev)),
            avg_edge: avg(plays.map((play) => play.abs_edge ?? play.edge)),
        };
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
                    <p>${this.escapeHtml(snapshot.mode || 'unknown')} &bull; ${this.escapeHtml(snapshot.selected_market_date || snapshot.selected_market_date_min || 'n/a')}</p>
                    <p>${this.escapeHtml(String(market.market_rows || 0))} market rows &bull; PTS ${this.escapeHtml(String(market.market_pts_lines || 0))} &bull; TRB ${this.escapeHtml(String(market.market_trb_lines || 0))} &bull; AST ${this.escapeHtml(String(market.market_ast_lines || 0))}</p>
                </div>
                <div>
                    <h3>Prior Game Data</h3>
                    <p>${this.escapeHtml(String(prior.slate_rows || 0))} player histories &bull; median ${this.escapeHtml(String(prior.history_rows_median || 0))} games</p>
                    <p>${this.escapeHtml(String(prior.history_before_market_violations || 0))} history violations &bull; ${this.escapeHtml(String(skipped.count || 0))} skipped rows</p>
                </div>
            </div>
        `;
    }

    renderCards() {
        this.elements.empty.style.display = this.filtered.length ? 'none' : 'block';
        this.elements.cards.innerHTML = this.filtered.map((play) => this.renderWantedCard(play)).join('');
    }

    renderWantedCard(play) {
        const tierRaw = String(play.recommendation || 'consider').toLowerCase();
        const tier = ['elite', 'strong', 'consider', 'pass'].includes(tierRaw) ? tierRaw : 'consider';
        const directionRaw = String(play.direction || '').toUpperCase();
        const direction = directionRaw === 'UNDER' ? 'UNDER' : 'OVER';
        const displayName = this.getPlayDisplayName(play);
        const escapedName = this.escapeHtml(displayName);
        const headshotUrl = this.getPlayHeadshotUrl(play);
        const monogram = this.escapeHtml(this.getMonogram(displayName));
        const lineText = this.formatNumber(play.market_line);
        const predictionText = this.formatNumber(play.prediction);
        const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(' @ ');
        const footerParts = [play.target || '', play.market_date || '', gameText].filter(Boolean);
        return `
            <article class="prediction-card wanted-card wanted-card-${tier}" data-direction="${this.escapeAttr(direction)}">
                <div class="wanted-rank">#${this.escapeHtml(String(play.rank || '-'))}</div>
                <div class="wanted-title">WANTED</div>
                <div class="wanted-photo-frame ${headshotUrl ? '' : 'is-fallback-visible'}">
                    ${headshotUrl ? `
                        <img
                            class="wanted-photo"
                            src="${this.escapeAttr(headshotUrl)}"
                            alt="${this.escapeAttr(displayName)} headshot"
                            loading="lazy"
                            onerror="this.remove(); this.parentElement.classList.add('is-fallback-visible');"
                        />
                    ` : ''}
                    <div class="wanted-photo-fallback">${monogram}</div>
                </div>
                <div class="wanted-reward-label">REWARD</div>
                <div class="wanted-reward-value">${this.escapeHtml(this.formatReward(play.ev))}</div>
                <div class="wanted-name">${escapedName}</div>
                <div class="wanted-direction">${this.escapeHtml(direction)}</div>
                <div class="wanted-prop-line">LINE ${this.escapeHtml(lineText)} | PRED ${this.escapeHtml(predictionText)}</div>
                <div class="wanted-footer">${this.escapeHtml(footerParts.join(' | '))}</div>
            </article>
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

    formatReward(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(1)}% EV` : 'n/a EV';
    }

    escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    escapeAttr(value) {
        return this.escapeHtml(value).replaceAll('`', '&#96;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DailyPredictionsPage();
});
