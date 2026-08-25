class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.openingPool = null;
        this.plays = [];
        this.availableDates = [];
        this.activeDate = null;
        this.currentDate = null;
        this.openingTarget = 'PTS';
        this.elements = {
            cards: document.getElementById('predictionCards'),
            empty: document.getElementById('predictionEmpty'),
            runMeta: document.getElementById('predictionRunMeta'),
            dateNav: document.getElementById('predictionDateNav'),
            openingRunFacts: document.getElementById('openingRunFacts'),
            openingPoolStatus: document.getElementById('openingPoolStatus'),
            openingPoolMetrics: document.getElementById('openingPoolMetrics'),
            openingGames: document.getElementById('openingGames'),
            openingTargetFilters: document.getElementById('openingTargetFilters'),
            openingProjectionPool: document.getElementById('openingProjectionPool'),
            openingWatchlistStatus: document.getElementById('openingWatchlistStatus'),
            openingWatchlists: document.getElementById('openingWatchlists'),
        };
        this.init();
    }

    init() {
        this.mountShell();
        if (window.CardVault && this.elements.cards) {
            this.elements.cards.innerHTML = window.CardVault.renderSkeletonCard(6);
        }
        this.bindOpeningFilters();
        this.loadDatesAndRender();
    }

    mountShell() {
        if (!window.CardVaultShell) return;

        window.CardVaultShell.mount({
            brandTitle: 'Prediction Bounties',
            brandHref: '/',
            sportSlug: 'nba',
            sportAccent: '#c02c3a',
            navLinks: [
                { label: 'Board', href: '/nba/predictions/', active: true },
                { label: 'Stats', href: '/nba/stats/', active: false },
                { label: 'Drive-Pass', href: '/nba/drive-pass/', active: false },
                { label: 'Post-Pass', href: '/nba/post-pass/', active: false },
                { label: 'Method', href: '/nba/prediction-about/', active: false },
            ],
            showDisclaimer: true,
        });
    }

    async loadDatesAndRender() {
        await Promise.all([this.loadDateIndex(), this.loadOpeningPool()]);
        const currentLoaded = await this.loadAndRender(null);
        if (!currentLoaded && this.availableDates.length) {
            await this.loadAndRender(this.availableDates[0]);
        }
        this.renderDateNav();
    }

    bindOpeningFilters() {
        this.elements.openingTargetFilters?.addEventListener('click', (event) => {
            const button = event.target.closest('button[data-target]');
            if (!button) return;
            this.openingTarget = button.dataset.target || 'PTS';
            this.elements.openingTargetFilters.querySelectorAll('button').forEach((item) => {
                item.classList.toggle('is-active', item === button);
            });
            this.renderOpeningProjectionPool();
        });
    }

    async loadOpeningPool() {
        try {
            const response = await fetch(`data/opening_night_pool.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`Opening pool HTTP ${response.status}`);
            this.openingPool = await response.json();
            this.renderOpeningPool();
        } catch (error) {
            console.error(error);
            if (this.elements.openingPoolStatus) {
                this.elements.openingPoolStatus.innerHTML = `<p><strong>Opening-night pool unavailable.</strong> ${this.escapeHtml(error.message)}</p>`;
            }
        }
    }

    renderOpeningPool() {
        const data = this.openingPool || {};
        const validation = data.validation || {};
        const watchlistPolicy = data.watchlist_policy || {};
        const cutoffDates = data.data_quality?.simulation_cutoff_dates || [];
        this.elements.openingRunFacts.innerHTML = [
            `${this.escapeHtml(data.season || 'NBA')} opener`,
            this.formatPoolDate(data.opening_date),
            `${this.formatInt(data.game_count)} games`,
            `${this.formatInt(data.player_count)} players`,
            `${this.formatInt(data.projection_count)} projections`,
        ].map((item) => `<span>${item}</span>`).join('');

        this.elements.openingPoolStatus.innerHTML = `<p><strong>Projection pool ready; bets withheld.</strong> ${this.escapeHtml(data.scope || '')} ${this.escapeHtml(data.data_quality?.roster_warning || '')}</p>`;
        const metrics = [
            ['Games', this.formatInt(data.game_count)],
            ['Players', this.formatInt(data.player_count)],
            ['Markets', this.formatInt(data.market_observations)],
            ['Model state', validation.frontend_label || 'research only'],
            ['Data cutoff', cutoffDates.join(', ') || 'n/a'],
        ];
        this.elements.openingPoolMetrics.innerHTML = metrics.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${this.escapeHtml(label)}</span><strong>${this.escapeHtml(value)}</strong></article>`).join('');
        this.renderOpeningGames();
        this.renderOpeningProjectionPool();
        this.elements.openingWatchlistStatus.innerHTML = `<p><strong>Templates only — not bets.</strong> ${this.escapeHtml(watchlistPolicy.reason || 'Current two-sided lines are required before evaluation.')}</p>`;
        this.renderOpeningWatchlists();
    }

    renderOpeningGames() {
        const games = Array.isArray(this.openingPool?.games) ? this.openingPool.games : [];
        this.elements.openingGames.innerHTML = games.map((game) => `<article class="opening-game-card">
            <p class="opening-game-card__time">${this.escapeHtml(this.formatTipoff(game.tipoff_utc))}</p>
            <p class="opening-game-card__matchup">${this.escapeHtml(`${game.away_team} at ${game.home_team}`)}</p>
            <p class="opening-game-card__time">${this.escapeHtml(game.network || '')}</p>
        </article>`).join('');
    }

    renderOpeningProjectionPool() {
        const pool = Array.isArray(this.openingPool?.pool) ? this.openingPool.pool : [];
        const rows = pool.filter((row) => row.target === this.openingTarget);
        if (!rows.length) {
            this.elements.openingProjectionPool.innerHTML = '<p>No opening-night projections are available.</p>';
            return;
        }
        const body = rows.map((row) => `<tr>
            <td>${this.escapeHtml(this.formatInt(row.projection_rank))}</td>
            <td><strong>${this.escapeHtml(row.player)}</strong><br><small>${this.escapeHtml(`${row.team} ${row.venue === 'home' ? 'vs' : 'at'} ${row.opponent}`)}</small></td>
            <td><strong>${this.escapeHtml(this.formatNum(row.projection, 1))}</strong></td>
            <td>${this.escapeHtml(`${this.formatNum(row.p10, 1)}–${this.formatNum(row.p90, 1)}`)}</td>
            <td>${this.escapeHtml(this.formatConfidence(row.confidence_tier))}</td>
            <td>${row.market_line == null ? 'Awaiting line' : this.escapeHtml(this.formatNum(row.market_line, 1))}</td>
        </tr>`).join('');
        this.elements.openingProjectionPool.innerHTML = `<table class="prediction-about-table"><thead><tr><th>RK</th><th>Player / Matchup</th><th>${this.escapeHtml(this.openingTarget)}</th><th>P10–P90</th><th>Confidence</th><th>Market</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    renderOpeningWatchlists() {
        const watchlists = Array.isArray(this.openingPool?.watchlists) ? this.openingPool.watchlists : [];
        this.elements.openingWatchlists.innerHTML = watchlists.map((watchlist) => `<article class="opening-watchlist-card">
            <h3>${this.escapeHtml(watchlist.name)}</h3>
            <p class="opening-watchlist-card__note">${this.escapeHtml(watchlist.note || '')}</p>
            <div class="opening-watchlist-legs">${(watchlist.legs || []).map((leg) => `<div class="opening-watchlist-leg">
                <div><strong>${this.escapeHtml(leg.player)}</strong><small>${this.escapeHtml(`${leg.team} vs ${leg.opponent}`)}</small></div>
                <div class="opening-watchlist-leg__projection"><strong>${this.escapeHtml(this.formatNum(leg.projection, 1))}</strong><small>${this.escapeHtml(leg.target)}</small></div>
            </div>`).join('')}</div>
        </article>`).join('');
    }

    async loadDateIndex() {
        try {
            const idx = await fetch(`data/history/index.json?v=${Date.now()}`);
            if (idx.ok) {
                const idxData = await idx.json();
                this.availableDates = Array.isArray(idxData.dates)
                    ? idxData.dates
                        .map((date) => String(date))
                        .filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date))
                        .sort()
                        .reverse()
                    : [];
            }
        } catch (_) { /* no history available */ }
    }

    async loadAndRender(date) {
        try {
            await this.load(date);
            this.renderCards();
            return true;
        } catch (error) {
            console.error(error);
            if (window.CardVault && this.elements.cards) {
                this.elements.cards.innerHTML = window.CardVault.renderEmptyState(
                    'Board unavailable',
                    `Unable to load daily predictions: ${error.message}`,
                    'Check that data/daily_predictions.json exists for this build.'
                );
            }
            return false;
        }
    }

    async load(date) {
        const url = date
            ? `data/history/${date}.json?v=${Date.now()}`
            : `data/daily_predictions.json?v=${Date.now()}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.data = await response.json();
        this.activeDate = this.data?.run_date || date || null;
        if (!date) this.currentDate = this.activeDate;
        const publicationStatus = String(this.data?.publication_status || 'ready').toLowerCase();
        this.plays = Array.isArray(this.data.plays)
            ? this.data.plays.map((play) => ({ ...play, board_publication_status: publicationStatus }))
            : [];
        this.plays.sort((a, b) => {
            const parlayDiff = Number(Boolean(b.parlay_candidate)) - Number(Boolean(a.parlay_candidate));
            if (parlayDiff !== 0) return parlayDiff;
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || 0) - (Number(a.abs_edge) || 0);
        });
        this.renderRunMeta();
    }

    renderDateNav() {
        const nav = this.elements.dateNav;
        if (!nav) return;

        const dates = [this.currentDate, ...this.availableDates]
            .filter((date, index, values) => date && values.indexOf(date) === index);
        if (dates.length < 2) {
            nav.innerHTML = '';
            return;
        }

        const buttons = dates.map((date) => {
            const isActive = date === this.activeDate;
            const label = this.formatDateLabel(date);
            return `<button type="button" class="date-nav__btn${isActive ? ' is-active' : ''}" data-date="${this.escapeHtml(date)}" aria-pressed="${isActive}">${this.escapeHtml(label)}</button>`;
        }).join('');

        nav.innerHTML = `<div class="date-nav__scroll">${buttons}</div>`;

        nav.querySelectorAll('.date-nav__btn').forEach((btn) => {
            btn.addEventListener('click', async () => {
                const date = btn.dataset.date;
                if (date === this.activeDate) return;

                if (this.elements.cards) {
                    this.elements.cards.innerHTML = window.CardVault
                        ? window.CardVault.renderSkeletonCard(4)
                        : '';
                }
                await this.loadAndRender(date === this.currentDate ? null : date);
                this.renderDateNav();
            });
        });
    }

    formatDateLabel(dateStr) {
        try {
            const d = new Date(dateStr + 'T12:00:00');
            const today = new Date();
            const todayStr = today.toISOString().slice(0, 10);
            if (dateStr === this.currentDate) return dateStr === todayStr ? 'Today' : 'Current';
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        } catch (_) {
            return dateStr;
        }
    }

    formatPoolDate(value) {
        if (!value) return 'Date n/a';
        const parsed = new Date(`${value}T12:00:00`);
        return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
    }

    formatTipoff(value) {
        const parsed = new Date(value);
        return Number.isNaN(parsed.valueOf()) ? 'Tipoff n/a' : parsed.toLocaleString('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
    }

    formatInt(value) {
        return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : 'n/a';
    }

    formatNum(value, places = 1) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(places) : 'n/a';
    }

    formatConfidence(value) {
        return String(value || 'unrated').replaceAll('_', ' ').toLowerCase();
    }

    renderRunMeta() {
        const runDate = this.data?.run_date || 'n/a';
        const throughDate = this.data?.through_date || 'n/a';
        const policy = this.data?.policy_profile || 'n/a';
        const publicationStatus = String(this.data?.publication_status || 'ready').toLowerCase();
        const publicationLabel = publicationStatus === 'ready' ? 'Published' : publicationStatus === 'review' ? 'Review' : 'Withheld';
        const publicationTone = publicationStatus === 'ready' ? 'active' : publicationStatus === 'review' ? 'stale' : 'withheld';
        const quality = this.data?.data_quality || {};
        const lagText = Number.isFinite(Number(quality.lag_days)) ? `${Number(quality.lag_days)}d` : 'n/a';

        if (this.elements.runMeta && window.CardVault) {
            this.elements.runMeta.innerHTML = `
                ${window.CardVault.renderStatusPill(publicationTone, publicationLabel)}
                <span class="prediction-run-meta__item">Run <strong>${this.escapeHtml(runDate)}</strong></span>
                <span class="prediction-run-meta__item">Data through <strong>${this.escapeHtml(throughDate)}</strong></span>
                <span class="prediction-run-meta__item">Lag <strong>${this.escapeHtml(lagText)}</strong></span>
                <span class="prediction-run-meta__item">Signals <strong>${this.plays.length}</strong></span>
                <span class="prediction-run-meta__item">Policy <strong>${this.escapeHtml(policy)}</strong></span>
            `;
        } else if (this.elements.runMeta) {
            this.elements.runMeta.textContent = `Run ${runDate} | Data through ${throughDate} | Policy ${policy} | ${publicationLabel}`;
        }
    }

    renderCards() {
        const cv = window.CardVault;
        if (!cv) {
            console.error('CardVault not loaded');
            return;
        }

        if (!this.plays.length) {
            const message = String(this.data?.publication_message || 'No analytical signals are available for this run.').trim();
            const emptyEl = this.elements.empty;
            if (emptyEl) {
                emptyEl.style.display = 'block';
                const msgP = emptyEl.querySelector('p');
                if (msgP) msgP.textContent = message || 'No analytical signals are available for this run.';
            }
            this.elements.cards.innerHTML = '';
            return;
        }

        if (this.elements.empty) {
            this.elements.empty.style.display = 'none';
        }

        this.elements.cards.innerHTML = this.plays
            .map((play, index) => cv.renderPredictionCard(play, index))
            .join('');
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
    new DailyPredictionsPage();
});
