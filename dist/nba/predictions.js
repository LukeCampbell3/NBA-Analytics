class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.availableDates = [];
        this.activeDate = null;
        this.currentDate = null;
        this.elements = {
            cards: document.getElementById('predictionCards'),
            empty: document.getElementById('predictionEmpty'),
            runMeta: document.getElementById('predictionRunMeta'),
            dateNav: document.getElementById('predictionDateNav'),
        };
        this.init();
    }

    init() {
        this.mountShell();
        if (window.CardVault && this.elements.cards) {
            this.elements.cards.innerHTML = window.CardVault.renderSkeletonCard(6);
        }
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
                { label: 'Method', href: '/nba/prediction-about/', active: false },
            ],
            showDisclaimer: true,
        });
    }

    async loadDatesAndRender() {
        await this.loadDateIndex();
        const currentLoaded = await this.loadAndRender(null);
        if (!currentLoaded && this.availableDates.length) {
            await this.loadAndRender(this.availableDates[0]);
        }
        this.renderDateNav();
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
