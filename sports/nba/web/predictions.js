class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.elements = {
            parlayTickets: document.getElementById('parlayTickets'),
            cards: document.getElementById('predictionCards'),
            empty: document.getElementById('predictionEmpty'),
            runMeta: document.getElementById('predictionRunMeta'),
        };
        this.init();
    }

    async init() {
        try {
            await this.load();
            this.renderParlayTickets();
            this.renderCards();
        } catch (error) {
            console.error(error);
            this.elements.cards.innerHTML = `<div class="prediction-about-empty">Unable to load daily predictions: ${this.escapeHtml(error.message)}</div>`;
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
        const publicationLabel = publicationStatus === 'ready' ? 'Published' : 'Withheld';
        this.elements.runMeta.textContent = `Run ${runDate} | Data through ${throughDate} | Policy ${policy} | ${publicationLabel}`;
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

    renderParlayTickets() {
        const container = this.elements.parlayTickets;
        if (!container) return;
        const parlays = Array.isArray(this.data?.parlay_board?.parlays) ? this.data.parlay_board.parlays : [];
        if (!parlays.length) {
            container.innerHTML = '';
            return;
        }
        container.innerHTML = parlays.map((parlay, idx) => {
            const legs = Array.isArray(parlay.legs) ? parlay.legs : [];
            const n = legs.length;
            const jointPct = this.formatPct(parlay.joint_probability || parlay.adjusted_probability);
            const type = String(parlay.type || 'primary').toUpperCase();
            const payout = Math.pow(1 + 100 / 110, n);
            const payoutText = `${payout.toFixed(1)}x`;
            const legsHtml = legs.map((leg, li) => {
                const name = this.escapeHtml(String(leg.player_display_name || leg.player || '').replace(/_/g, ' '));
                const target = this.escapeHtml(String(leg.target || ''));
                const dir = this.escapeHtml(String(leg.direction || '').toUpperCase());
                const line = this.formatNumber(leg.market_line);
                const wr = this.formatPct(leg.win_rate || leg.hit_probability || leg.expected_win_rate);
                return `<div class="parlay-leg">
                    <span class="parlay-leg-num">${li + 1}</span>
                    <span class="parlay-leg-name">${name}</span>
                    <span class="parlay-leg-prop">${target} ${dir} ${this.escapeHtml(line)}</span>
                    <span class="parlay-leg-prob">${this.escapeHtml(wr)}</span>
                </div>`;
            }).join('');
            return `<article class="parlay-ticket">
                <div class="parlay-ticket-header">
                    <span class="parlay-ticket-badge">${this.escapeHtml(type)} PARLAY</span>
                    <span class="parlay-ticket-legs">${n}-LEG</span>
                    <span class="parlay-ticket-payout">${this.escapeHtml(payoutText)} PAYOUT</span>
                </div>
                <div class="parlay-ticket-prob">
                    <span class="parlay-ticket-prob-label">Combined hit probability</span>
                    <span class="parlay-ticket-prob-value">${this.escapeHtml(jointPct)}</span>
                </div>
                <div class="parlay-ticket-legs-list">${legsHtml}</div>
            </article>`;
        }).join('');
    }

    renderCards() {
        if (!this.plays.length) {
            const message = String(this.data?.publication_message || 'No prediction bounties available right now.').trim();
            this.elements.empty.innerHTML = `<p>${this.escapeHtml(message || 'No prediction bounties available right now.')}</p>`;
        }
        this.elements.empty.style.display = this.plays.length ? 'none' : 'block';
        this.elements.cards.innerHTML = this.plays.map((play) => this.renderWantedCard(play)).join('');
    }

    renderWantedCard(play) {
        const directionRaw = String(play.direction || '').toUpperCase();
        const direction = directionRaw === 'UNDER' ? 'UNDER' : 'OVER';
        const displayName = this.getPlayDisplayName(play);
        const target = this.escapeHtml(String(play.target || ''));
        const lineText = this.formatNumber(play.market_line);
        const ev = Number(play.ev) || 0;
        const evText = Number.isFinite(ev) ? `${ev >= 0 ? '+' : ''}${(ev * 100).toFixed(1)}%` : '';
        const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(' @ ');
        const headshotUrl = this.getPlayHeadshotUrl(play);
        const monogram = this.escapeHtml(this.getMonogram(displayName));

        return `
            <article class="bounty-card" data-direction="${this.escapeAttr(direction)}">
                <div class="bounty-top">
                    <span class="bounty-rank">#${this.escapeHtml(String(play.rank || '-'))}</span>
                    <span class="bounty-ev ${ev >= 0 ? 'positive' : 'negative'}">${this.escapeHtml(evText)} EV</span>
                </div>
                <div class="bounty-headshot ${headshotUrl ? '' : 'is-fallback'}">
                    ${headshotUrl ? `<img src="${this.escapeAttr(headshotUrl)}" alt="${this.escapeAttr(displayName)}" loading="lazy" onerror="this.remove(); this.parentElement.classList.add('is-fallback');" />` : ''}
                    <span class="bounty-headshot-fallback">${monogram}</span>
                </div>
                <div class="bounty-name">${this.escapeHtml(displayName)}</div>
                <div class="bounty-pick">
                    <span class="bounty-target">${target}</span>
                    <span class="bounty-direction">${this.escapeHtml(direction)}</span>
                    <span class="bounty-line">${this.escapeHtml(lineText)}</span>
                </div>
                <div class="bounty-meta">${this.escapeHtml(gameText)}</div>
            </article>
        `;
    }

    formatNumber(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : 'n/a';
    }

    formatReward(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(1)}% EV` : 'n/a EV';
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : 'n/a';
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
