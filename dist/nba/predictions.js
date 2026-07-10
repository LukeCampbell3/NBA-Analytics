class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.elements = {
            cards: document.getElementById('predictionCards'),
            empty: document.getElementById('predictionEmpty'),
            runMeta: document.getElementById('predictionRunMeta'),
        };
        this.init();
    }

    init() {
        this.mountShell();
        if (window.CardVault && this.elements.cards) {
            this.elements.cards.innerHTML = window.CardVault.renderSkeletonCard(6);
        }
        this.loadAndRender();
    }

    mountShell() {
        if (!window.CardVaultShell) return;

        window.CardVaultShell.mount({
            brandTitle: 'NBA Analytics',
            brandHref: '/',
            workspaceLabel: 'NBA',
            workspaceHref: '/nba/',
            sportSlug: 'nba',
            sportAccent: '#f59e0b',
            breadcrumbs: [{ label: 'Prediction Board', href: 'predictions.html' }],
            navLinks: [
                { label: 'Dashboard', href: 'index.html', active: false },
                { label: 'PAR Records', href: 'par.html', active: false },
                { label: 'Board', href: 'predictions.html', active: true },
                { label: 'Safe-State', href: 'safe-state.html', active: false },
                { label: 'Method', href: 'prediction-about.html', active: false },
                { label: 'Metrics', href: 'about.html', active: false },
            ],
            showDisclaimer: true,
        });
    }

    async loadAndRender() {
        try {
            await this.load();
            this.renderCards();
        } catch (error) {
            console.error(error);
            if (window.CardVault && this.elements.cards) {
                this.elements.cards.innerHTML = window.CardVault.renderEmptyState(
                    'Board unavailable',
                    `Unable to load daily predictions: ${error.message}`,
                    'Check that data/daily_predictions.json exists for this build.'
                );
            }
        }
    }

    async load() {
        const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.data = await response.json();
        this.plays = Array.isArray(this.data.plays) ? this.data.plays.slice() : [];
        this.plays.sort((a, b) => {
            const parlayDiff = Number(Boolean(b.parlay_candidate)) - Number(Boolean(a.parlay_candidate));
            if (parlayDiff !== 0) return parlayDiff;
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || 0) - (Number(a.abs_edge) || 0);
        });
        this.renderRunMeta();
    }

    renderRunMeta() {
        const runDate = this.data?.run_date || 'n/a';
        const throughDate = this.data?.through_date || 'n/a';
        const policy = this.data?.policy_profile || 'n/a';
        const publicationStatus = String(this.data?.publication_status || 'ready').toLowerCase();
        const publicationLabel = publicationStatus === 'ready' ? 'Published' : publicationStatus === 'review' ? 'Review' : 'Withheld';
        const publicationTone = publicationStatus === 'ready' ? 'active' : publicationStatus === 'review' ? 'stale' : 'withheld';
        const stale = publicationStatus !== 'ready';
        const quality = this.data?.data_quality || {};
        const lagText = Number.isFinite(Number(quality.lag_days)) ? `Lag ${Number(quality.lag_days)}d / ` : '';

        if (this.elements.runMeta && window.CardVault) {
            this.elements.runMeta.innerHTML = [
                window.CardVault.renderDataFreshness(`Run ${runDate} · Data through ${throughDate}`, stale),
                ` · Policy ${this.escapeHtml(policy)} · `,
                this.escapeHtml(lagText),
                window.CardVault.renderStatusPill(publicationTone, publicationLabel),
                ` · ${this.plays.length} board card${this.plays.length === 1 ? '' : 's'}`,
            ].join('');
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
