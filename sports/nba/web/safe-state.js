class DistributionRangeBar {
    static render(stat, label) {
        const p10 = Number(stat?.p10);
        const p50 = Number(stat?.p50 ?? stat?.median);
        const p90 = Number(stat?.p90);
        const max = Math.max(p90, p50, p10, 1);
        const left = Number.isFinite(p10) ? Math.max(0, Math.min(100, (p10 / max) * 100)) : 0;
        const mid = Number.isFinite(p50) ? Math.max(0, Math.min(100, (p50 / max) * 100)) : 50;
        const right = Number.isFinite(p90) ? Math.max(0, Math.min(100, (p90 / max) * 100)) : 100;
        return `
            <div class="dist-row">
                <div class="dist-label">${SafeStatePage.escapeHtml(label)}</div>
                <div class="dist-track" aria-label="${SafeStatePage.escapeAttr(label)} p10 to p90 range">
                    <span class="dist-band" style="left:${left}%; width:${Math.max(2, right - left)}%"></span>
                    <span class="dist-dot" style="left:${mid}%"></span>
                </div>
                <div class="dist-values">${SafeStatePage.formatNumber(p10)} / ${SafeStatePage.formatNumber(p50)} / ${SafeStatePage.formatNumber(p90)}</div>
            </div>
        `;
    }
}

class ConfidenceBadge {
    static render(value) {
        const text = String(value || 'INSUFFICIENT_DATA');
        return `<span class="confidence-badge confidence-${SafeStatePage.escapeAttr(text.toLowerCase())}">${SafeStatePage.escapeHtml(text.replaceAll('_', ' '))}</span>`;
    }
}

class RiskBadge {
    static render(value) {
        return `<span class="risk-badge">${SafeStatePage.escapeHtml(String(value || '').replaceAll('_', ' '))}</span>`;
    }
}

class ScenarioSummary {
    static render(upside, downside) {
        return `
            <div class="scenario-summary">
                <p><strong>Upside:</strong> ${SafeStatePage.escapeHtml(upside || 'More stable role and minutes can lift the upper range.')}</p>
                <p><strong>Downside:</strong> ${SafeStatePage.escapeHtml(downside || 'Role or availability volatility can pull results toward the floor.')}</p>
            </div>
        `;
    }
}

class SafeStateCard {
    static render(card, locked = false) {
        const badges = Array.isArray(card.warning_badges) ? card.warning_badges : [];
        return `
            <article class="safe-state-card ${locked ? 'player-card--locked' : ''}">
                <div class="safe-state-card-top">
                    <span class="shadow-pill">SHADOW</span>
                    <span class="settlement-pill">${SafeStatePage.escapeHtml(card.settlement_status || 'PENDING')}</span>
                    ${locked ? '<span class="vault-status-pill vault-status-pill--withheld">Preview</span>' : ''}
                </div>
                <h3>${SafeStatePage.escapeHtml(card.player || 'Unknown Player')}</h3>
                <div class="safe-state-market">${SafeStatePage.escapeHtml(card.market_type || '')} ${SafeStatePage.escapeHtml(card.side || '')} ${SafeStatePage.formatNumber(card.line)}</div>
                <div class="safe-state-metrics">
                    <div><span>Price</span><strong>${SafeStatePage.escapeHtml(card.price ?? 'n/a')}</strong></div>
                    <div><span>Break-even</span><strong>${SafeStatePage.formatPct(card.break_even_probability)}</strong></div>
                    <div><span>LCB edge</span><strong>${SafeStatePage.formatPct(card.lcb_edge)}</strong></div>
                </div>
                <div class="safe-state-tier-stack">
                    ${RiskBadge.render(card.edge_defendability_tier || 'EDGE_UNKNOWN')}
                    ${RiskBadge.render(card.forecastability_tier || 'FORECASTABILITY_UNKNOWN')}
                    ${RiskBadge.render(card.safe_state_tier || 'SAFE_STATE_UNKNOWN')}
                </div>
                <p class="safe-state-blocker"><strong>Blocker:</strong> ${SafeStatePage.escapeHtml(card.primary_blocker || card.root_cause || 'none recorded')}</p>
                <p class="safe-state-explanation">${SafeStatePage.escapeHtml(card.explanation || 'Shadow-only evidence. Production picks are unchanged.')}</p>
                <div class="safe-state-badges">${badges.map(RiskBadge.render).join('')}</div>
            </article>
        `;
    }
}

class PlayerSimulationCard {
    static render(card, credibility) {
        const risks = Array.isArray(card.main_risk_factors) ? card.main_risk_factors : [];
        const warnings = Array.isArray(card.missing_data_warnings) ? card.missing_data_warnings : [];
        const label = credibility?.labels?.frontend_label || 'research projection / uncalibrated';
        const gateStatus = credibility?.status || 'BACKTEST_REQUIRED';
        return `
            <article class="simulation-card">
                <div class="safe-state-card-top">
                    ${ConfidenceBadge.render(card.confidence_tier)}
                    <span class="settlement-pill">Cutoff ${SafeStatePage.escapeHtml(card.data_cutoff_date || 'n/a')}</span>
                </div>
                <div class="simulation-credibility ${SafeStatePage.escapeAttr(gateStatus.toLowerCase())}">${SafeStatePage.escapeHtml(label)}</div>
                <h3>${SafeStatePage.escapeHtml(card.player || 'Unknown Player')}</h3>
                <div class="simulation-subtitle">${SafeStatePage.escapeHtml(card.team || '')} ${SafeStatePage.escapeHtml(card.archetype || '')}</div>
                <div class="simulation-score-row">
                    <div><span>Forecastability</span><strong>${SafeStatePage.formatPct(card.forecastability_score)}</strong></div>
                    <div><span>Volatility</span><strong>${SafeStatePage.formatPct(card.volatility_score)}</strong></div>
                    <div><span>Minutes</span><strong>${SafeStatePage.formatNumber(card.projected_minutes_per_game)}</strong></div>
                </div>
                ${DistributionRangeBar.render(card.pts, 'PTS p10/p50/p90')}
                ${DistributionRangeBar.render(card.reb, 'REB p10/p50/p90')}
                ${DistributionRangeBar.render(card.ast, 'AST p10/p50/p90')}
                ${DistributionRangeBar.render(card.pra, 'PRA p10/p50/p90')}
                <p class="simulation-summary">${SafeStatePage.escapeHtml(card.best_projection_summary || '')}</p>
                <p class="simulation-summary">${SafeStatePage.escapeHtml(card.uncertainty_summary || '')}</p>
                ${ScenarioSummary.render(card.primary_upside_path, card.primary_downside_path)}
                <div class="safe-state-badges">${risks.map(RiskBadge.render).join('')}${warnings.map(w => RiskBadge.render(`missing ${w}`)).join('')}</div>
            </article>
        `;
    }
}

class SafeStatePage {
    constructor() {
        this.elements = {
            meta: document.getElementById('safeStateMeta'),
            entitlementMeta: document.getElementById('entitlementMeta'),
            safeCards: document.getElementById('safeStateCards'),
            simCards: document.getElementById('simulationCards'),
            safeEmpty: document.getElementById('safeStateEmpty'),
            simEmpty: document.getElementById('simulationEmpty'),
            safeLocked: document.getElementById('safeStateLocked'),
        };
        this.init();
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
            breadcrumbs: [{ label: 'Safe-State Lab', href: 'safe-state.html' }],
            navLinks: [
                { label: 'Dashboard', href: 'index.html' },
                { label: 'PAR Records', href: 'par.html' },
                { label: 'Board', href: 'predictions.html' },
                { label: 'Safe-State', href: 'safe-state.html', active: true },
                { label: 'Pricing', href: 'pricing.html' },
                { label: 'Account', href: 'account.html' },
                { label: 'API', href: 'api.html' },
            ],
            showDisclaimer: true,
        });
    }

    async init() {
        this.mountShell();
        const [safeState, simulations, manifest, credibility, entitlements] = await Promise.all([
            this.fetchJson('data/safe_state_latest.json', { cards: [] }),
            this.fetchJson('data/player_simulation_cards.json', []),
            this.fetchJson('data/site_manifest.json', {}),
            this.fetchJson('data/simulation_credibility_gate.json', { status: 'BACKTEST_REQUIRED', labels: { frontend_label: 'research projection / uncalibrated' } }),
            this.loadEntitlements(),
        ]);

        let safeCards = Array.isArray(safeState.cards) ? safeState.cards : [];
        let simCards = Array.isArray(simulations) ? simulations : [];
        let lockedCount = 0;

        if (window.NbaAuthClient?.getToken?.()) {
            try {
                const paid = await window.NbaAuthClient.apiFetch('/api/safe-state/cards');
                if (Array.isArray(paid.cards) && paid.cards.length) {
                    lockedCount = Math.max(0, Number(paid.total_available || 0) - paid.cards.length);
                    safeCards = paid.cards;
                }
                const simPaid = await window.NbaAuthClient.apiFetch('/api/players/simulations');
                if (Array.isArray(simPaid.cards) && simPaid.cards.length) {
                    simCards = simPaid.cards;
                }
            } catch (error) {
                console.warn('Paid safe-state fetch failed, using public preview', error);
            }
        }

        const caps = entitlements?.capabilities || {};
        const previewLimit = caps.can_view_full_safe_state ? safeCards.length : (caps.max_cards_per_day || 3);
        if (!caps.can_view_full_safe_state && safeCards.length > previewLimit) {
            lockedCount = safeCards.length - previewLimit;
            safeCards = safeCards.slice(0, previewLimit);
        }
        if (!caps.can_view_simulation_filters) {
            simCards = simCards.slice(0, 3);
        }

        this.elements.meta.textContent = `Run ${safeState.run_date || manifest.run_date || 'n/a'} | Cutoff ${safeState.data_cutoff_date || manifest.data_cutoff_date || 'n/a'} | Shadow-only evidence`;
        this.elements.entitlementMeta.textContent = caps.can_view_full_safe_state
            ? 'Full safe-state board unlocked for your plan.'
            : `Free preview: ${previewLimit} safe-state cards with ${caps.settlement_delay_hours || 24}h settlement delay. Upgrade for full analytics depth.`;

        this.elements.safeEmpty.style.display = safeCards.length ? 'none' : 'block';
        this.elements.simEmpty.style.display = simCards.length ? 'none' : 'block';
        this.elements.safeLocked.style.display = lockedCount > 0 ? 'block' : 'none';
        this.elements.safeCards.innerHTML = safeCards.map((card) => SafeStateCard.render(card, !caps.can_view_full_safe_state)).join('');
        this.elements.simCards.innerHTML = simCards.slice(0, 60).map((card) => PlayerSimulationCard.render(card, credibility)).join('');
    }

    async loadEntitlements() {
        if (!window.NbaAuthClient) return { capabilities: { max_cards_per_day: 3, can_view_full_safe_state: false, settlement_delay_hours: 24 } };
        try {
            return await window.NbaAuthClient.fetchEntitlements();
        } catch (_error) {
            return { capabilities: { max_cards_per_day: 3, can_view_full_safe_state: false, settlement_delay_hours: 24 } };
        }
    }

    async fetchJson(path, fallback) {
        try {
            const response = await fetch(`${path}?v=${Date.now()}`);
            if (!response.ok) return fallback;
            return await response.json();
        } catch (_error) {
            return fallback;
        }
    }

    static formatNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number.toFixed(1) : 'n/a';
    }

    static formatPct(value) {
        const number = Number(value);
        return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : 'n/a';
    }

    static escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    static escapeAttr(value) {
        return SafeStatePage.escapeHtml(value).replaceAll('`', '&#96;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SafeStatePage();
});
