const state = {
    leaderboard: [],
    players: [],
    atoms: [],
    forecasts: [],
    model: {},
    validation: {},
};

const categoryOrder = [
    ['SCORING', 'scoring_par', 'Scoring'],
    ['CREATION', 'creation_par', 'Creation'],
    ['BALL_SECURITY', 'ball_security_par', 'Ball Security'],
    ['PLAYTYPE_PNR', 'playtype_pnr_par', 'Pick & Roll'],
    ['SPACING', 'spacing_par', 'Spacing'],
    ['REBOUNDING', 'rebounding_par', 'Rebounding'],
    ['PERIMETER_DISRUPTION', 'perimeter_disruption_par', 'Perimeter Disruption'],
    ['RIM_DEFENSE', 'rim_defense_par', 'Rim Defense'],
    ['CONTEST_DEFENSE', 'contest_defense_par', 'Contest Defense'],
    ['HUSTLE', 'hustle_par', 'Hustle'],
    ['RESIDUAL', 'residual_par', 'Residual'],
];

const fmt = (value, digits = 1) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Limited evidence';
    const n = Number(value);
    return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`;
};
const pct = (value) => value === null || value === undefined ? 'Limited evidence' : `${(Number(value) * 100).toFixed(0)}%`;
const byId = (id) => document.getElementById(id);

function mountShell() {
    if (!window.CardVaultShell) return;
    window.CardVaultShell.mount({
        brandTitle: 'NBA Analytics',
        brandHref: '/',
        workspaceLabel: 'NBA',
        workspaceHref: '/nba/',
        sportSlug: 'nba',
        sportAccent: '#f59e0b',
        breadcrumbs: [{ label: 'PAR Records', href: 'par.html' }],
        navLinks: [
            { label: 'Dashboard', href: 'index.html' },
            { label: 'PAR Records', href: 'par.html', active: true },
            { label: 'Board', href: 'predictions.html' },
            { label: 'Safe-State', href: 'safe-state.html' },
            { label: 'Method', href: 'prediction-about.html' },
            { label: 'Metrics', href: 'about.html' },
        ],
        showDisclaimer: true,
    });
}

async function readJson(path, fallback) {
    try {
        const response = await fetch(path, { cache: 'no-store' });
        if (!response.ok) return fallback;
        return await response.json();
    } catch {
        return fallback;
    }
}

async function readJsonl(path) {
    try {
        const response = await fetch(path, { cache: 'no-store' });
        if (!response.ok) return [];
        const text = await response.text();
        return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    } catch {
        return [];
    }
}

async function loadData() {
    const [leaderboard, players, atoms, forecasts, model, validation, manifest] = await Promise.all([
        readJson('data/par_leaderboard.json', []),
        readJson('data/player_par_components.json', []),
        readJson('data/player_par_atom_summary.json', []),
        readJson('data/player_par_forecasts.json', []),
        readJson('data/par_model.json', {}),
        readJson('data/par_validation.json', {}),
        readJson('data/par_build_manifest.json', {}),
    ]);
    Object.assign(state, { leaderboard, players, atoms, forecasts, model, validation, manifest });
    renderMeta();
    populateFilters();
    route();
}

function renderMeta() {
    const meta = byId('parModelMeta');
    const proof = byId('parProofBadge');
    meta.textContent = `${state.model.par_model_version || 'par_pvg_v0_5'} / ${state.model.parf_model_version || 'parf_v0_6'} | points per win ${state.model.points_per_win || 30.4}`;
    const allowed = state.manifest && state.manifest.production_publish_allowed;
    proof.textContent = allowed ? 'Production proof passed' : 'Production proof blocked';
    proof.classList.toggle('is-blocked', !allowed);
}

function populateFilters() {
    const seasons = [...new Set(state.leaderboard.map((r) => r.season).filter(Boolean))].sort();
    const teams = [...new Set(state.leaderboard.map((r) => r.team).filter(Boolean))].sort();
    const roles = [...new Set(state.leaderboard.map((r) => r.role).filter(Boolean))].sort();
    fillSelect(byId('parSeasonFilter'), seasons, 'All seasons');
    fillSelect(byId('parTeamFilter'), teams, 'All teams');
    fillSelect(byId('parRoleFilter'), roles, 'All roles');
    ['parSearch', 'parSeasonFilter', 'parTeamFilter', 'parRoleFilter', 'parMinMinutes', 'parSort'].forEach((id) => {
        byId(id).addEventListener('input', renderLeaderboard);
        byId(id).addEventListener('change', renderLeaderboard);
    });
    byId('parBackButton').addEventListener('click', () => {
        history.pushState({}, '', 'par.html');
        route();
    });
}

function fillSelect(select, values, allLabel) {
    select.innerHTML = `<option value="">${allLabel}</option>` + values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
}

function renderLeaderboard() {
    byId('leaderboardView').hidden = false;
    byId('playerView').hidden = true;
    const query = byId('parSearch').value.trim().toLowerCase();
    const season = byId('parSeasonFilter').value;
    const team = byId('parTeamFilter').value;
    const role = byId('parRoleFilter').value;
    const minMinutes = Number(byId('parMinMinutes').value || 0);
    const sort = byId('parSort').value;
    const rows = state.leaderboard
        .filter((r) => !query || String(r.player_name).toLowerCase().includes(query))
        .filter((r) => !season || r.season === season)
        .filter((r) => !team || r.team === team)
        .filter((r) => !role || r.role === role)
        .filter((r) => Number(r.minutes || 0) >= minMinutes)
        .sort((a, b) => Number(b[sort] ?? -Infinity) - Number(a[sort] ?? -Infinity));
    byId('parLeaderboardBody').innerHTML = rows.map((row, idx) => `
        <tr>
            <td>${idx + 1}</td>
            <td><a href="par.html?player=${encodeURIComponent(row.player_id)}" data-player-link="${escapeHtml(row.player_id)}">${escapeHtml(row.player_name)}</a></td>
            <td>${escapeHtml(row.team || '')}</td>
            <td>${escapeHtml(row.role || '')}</td>
            <td>${Number(row.minutes || 0).toFixed(0)}</td>
            <td class="${signedClass(row.par)}">${fmt(row.par)}</td>
            <td class="${signedClass(row.par_1000)}">${fmt(row.par_1000)}</td>
            <td>${Number(row.pvg_score || 0).toFixed(1)}</td>
            <td class="${signedClass(row.projected_parf)}">${fmt(row.projected_parf)}</td>
            <td>${pct(row.continuation_score)}</td>
        </tr>
    `).join('');
    document.querySelectorAll('[data-player-link]').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            history.pushState({}, '', `par.html?player=${encodeURIComponent(link.dataset.playerLink)}`);
            route();
        });
    });
}

function route() {
    const url = new URL(window.location.href);
    const pathMatch = window.location.pathname.match(/\/par\/player\/([^/]+)/);
    const playerId = pathMatch ? decodeURIComponent(pathMatch[1]) : url.searchParams.get('player');
    if (playerId) renderPlayer(playerId);
    else renderLeaderboard();
}

function renderPlayer(playerId) {
    const player = state.players.find((row) => String(row.player_id) === String(playerId));
    if (!player) {
        renderLeaderboard();
        return;
    }
    byId('leaderboardView').hidden = true;
    byId('playerView').hidden = false;
    const forecast = state.forecasts.find((row) => String(row.player_id) === String(playerId)) || {};
    const atoms = state.atoms.filter((row) => String(row.player_id) === String(playerId));
    const maxAbs = Math.max(1, ...categoryOrder.map(([, field]) => Math.abs(Number(player[field] || 0))));
    const splitRows = categoryOrder.map(([category, field, label]) => {
        const status = player.category_statuses && player.category_statuses[category];
        const value = status && status.status === 'insufficient_evidence' ? null : player[field];
        return { category, field, label, value, status: status ? status.status : 'measured' };
    });
    byId('playerDetail').innerHTML = `
        <section class="par-player-summary">
            <div>
                <h2>${escapeHtml(player.player_name)}</h2>
                <p>${escapeHtml(player.team || '')} | ${escapeHtml(player.role || '')} | ${escapeHtml(player.season || '')}</p>
            </div>
            <div class="par-summary-grid">
                ${metric('PAR', fmt(player.total_par))}
                ${metric('PAR/1000', fmt(player.par_1000))}
                ${metric('PVG Score', Number(player.pvg_score || 0).toFixed(1))}
                ${metric('WAR Equivalent', fmt(player.war_equivalent, 2))}
                ${metric('PAR-F Experimental', fmt(forecast.projected_par))}
                ${metric('PAR-F Exp. CI', `${fmt(forecast.confidence_interval_low)} to ${fmt(forecast.confidence_interval_high)}`)}
                ${metric('Continuation', pct(forecast.continuation_score))}
                ${metric('Role Portability', pct(forecast.role_portability_score))}
            </div>
        </section>
        <section class="par-panel">
            <h3>Where The Player Adds Value</h3>
            <div class="par-splits">
                ${splitRows.map((row) => splitRow(row, maxAbs, atoms)).join('')}
                <div class="par-total-line"><span>Total PAR = sum of displayed value splits</span><strong>${fmt(player.total_par)}</strong></div>
            </div>
        </section>
        <section class="par-panel">
            <h3>Next-Season Value Outlook</h3>
            <div class="par-outlook-grid">
                ${metric('Current PAR', fmt(forecast.current_par))}
                ${metric('PAR-F Experimental', fmt(forecast.projected_par))}
                ${metric('Confidence Interval', `${fmt(forecast.confidence_interval_low)} to ${fmt(forecast.confidence_interval_high)}`)}
                ${metric('Stable PAR Share', pct(forecast.stable_par_share))}
                ${metric('Volatile PAR Share', pct(forecast.volatile_par_share))}
                ${metric('Forecast Confidence', pct(forecast.forecast_confidence))}
            </div>
            ${forecastBridge(forecast.forecast_bridge || {})}
            ${categoryForecastTable(player, forecast, atoms)}
        </section>
        <section class="par-panel">
            <h3>Evidence And Data Quality</h3>
            <div class="par-evidence-grid">
                ${metric('PAR Evidence Coverage', pct(player.par_evidence_coverage))}
                ${metric('Direct Value Share', share(player.direct_par, player.total_par))}
                ${metric('Tracking-Backed Share', share(player.tracking_backed_par, player.total_par))}
                ${metric('Proxy Share', share(player.proxy_par, player.total_par))}
                ${metric('Residual Share', share(player.residual_par, player.total_par))}
                ${metric('Supported Atoms', player.supported_atom_count)}
                ${metric('Unsupported Atoms', player.unsupported_atom_count)}
            </div>
        </section>
    `;
}

function splitRow(row, maxAbs, atoms) {
    const width = row.value === null ? 0 : Math.max(2, Math.abs(Number(row.value)) / maxAbs * 100);
    const negative = Number(row.value) < 0;
    const categoryAtoms = atoms.filter((atom) => atom.category === row.category);
    const drilldown = categoryAtoms.length ? categoryAtoms.map((atom) => `
        <tr>
            <td>${escapeHtml(labelForAtom(atom.primary_value_label))}</td>
            <td class="${signedClass(atom.par_value)}">${fmt(atom.par_value)}</td>
            <td>${Number(atom.replacement_baseline || 0).toFixed(2)}</td>
            <td>${escapeHtml(atom.source_tier || '')}</td>
            <td>${Number(atom.reliability_weight || 0).toFixed(2)}</td>
            <td>${escapeHtml(atom.confidence_tier || '')}</td>
        </tr>
    `).join('') : `<tr><td colspan="6">Limited evidence</td></tr>`;
    return `
        <details class="par-split-row">
            <summary>
                <span class="par-split-label">${escapeHtml(row.label)}</span>
                <span class="par-diverging-bar ${negative ? 'is-negative' : 'is-positive'}">
                    <span style="width:${width}%"></span>
                </span>
                <strong class="${signedClass(row.value)}">${fmt(row.value)}</strong>
            </summary>
            <table class="par-atom-table">
                <thead><tr><th>Atom</th><th>PAR</th><th>Replacement</th><th>Source Tier</th><th>Reliability</th><th>Confidence</th></tr></thead>
                <tbody>${drilldown}</tbody>
            </table>
        </details>
    `;
}

function forecastBridge(bridge) {
    const rows = [
        ['Current PAR', bridge.current_par],
        ['Persistence adjustment', bridge.persistence_adjustment],
        ['Trend', bridge.trend],
        ['Role continuity', bridge.role_continuity],
        ['Minutes adjustment', bridge.minutes_adjustment],
        ['Health adjustment', bridge.health_adjustment],
        ['Age curve', bridge.age_curve],
        ['Fit lift', bridge.fit_lift],
        ['PAR-F Experimental', bridge.projected_par_f],
    ];
    return `<div class="par-bridge">${rows.map(([label, value]) => `<div><span>${label}</span><strong class="${signedClass(value)}">${fmt(value)}</strong></div>`).join('')}</div>`;
}

function categoryForecastTable(player, forecast, atoms) {
    const byCategory = {};
    atoms.forEach((atom) => {
        const persistenceKey = state.model.atom_registry && state.model.atom_registry[atom.primary_value_label] && state.model.atom_registry[atom.primary_value_label].persistence_key;
        const persistence = state.model.persistence_values && persistenceKey ? state.model.persistence_values[persistenceKey] : 0;
        byCategory[atom.category] = (byCategory[atom.category] || 0) + Number(atom.par_value || 0) * persistence * Number(atom.reliability_weight || 0);
    });
    return `<table class="par-forecast-table"><thead><tr><th>Category</th><th>Current</th><th>PAR-F Exp.</th></tr></thead><tbody>
        ${categoryOrder.map(([category, field, label]) => `<tr><td>${label}</td><td class="${signedClass(player[field])}">${fmt(player[field])}</td><td class="${signedClass(byCategory[category])}">${fmt(byCategory[category] || null)}</td></tr>`).join('')}
    </tbody></table>`;
}

function metric(label, value) {
    return `<div class="par-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function share(value, total) {
    const denom = Math.abs(Number(total || 0));
    if (!denom) return '0%';
    return `${(Math.abs(Number(value || 0)) / denom * 100).toFixed(0)}%`;
}

function labelForAtom(atomType) {
    const cfg = state.model.atom_registry && state.model.atom_registry[atomType];
    return cfg ? cfg.label : atomType;
}

function signedClass(value) {
    if (value === null || value === undefined) return 'is-limited';
    return Number(value) < 0 ? 'is-negative-value' : 'is-positive-value';
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[ch]));
}

window.addEventListener('popstate', route);
mountShell();
loadData();
