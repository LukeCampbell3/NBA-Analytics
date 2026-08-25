class NflFantasyDraftBoard {
    constructor() {
        this.data = null;
        this.position = "ALL";
        this.query = "";
        this.selectedId = null;
        this.elements = {
            season: document.getElementById("seasonLabel"),
            badge: document.getElementById("validationBadge"),
            runFacts: document.getElementById("runFacts"),
            design: document.getElementById("validationDesign"),
            metrics: document.getElementById("validationMetrics"),
            confidence: document.getElementById("confidenceMetrics"),
            filters: document.getElementById("positionFilters"),
            search: document.getElementById("playerSearch"),
            table: document.getElementById("rankingTable"),
            detail: document.getElementById("playerDetail"),
            note: document.getElementById("methodNote"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        this.bindControls();
        try {
            const response = await fetch(`data/fantasy_draft_rankings.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.selectedId = this.data.rankings?.[0]?.player_id || null;
            this.render();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load fantasy rankings: ${error.message}`;
            this.elements.badge.textContent = "Unavailable";
            this.elements.badge.classList.add("is-failed");
        }
    }

    mountShell() {
        if (!window.CardVaultShell) return;
        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: "nfl", sportAccent: "#b42318",
            navLinks: [
                { label: "Picks", href: "/nfl/predictions/", active: false },
                { label: "Fantasy Draft", href: "/nfl/fantasy/", active: true },
                { label: "Method", href: "/nfl/prediction-about/", active: false },
            ],
            showDisclaimer: true,
        });
    }

    bindControls() {
        this.elements.filters.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-position]");
            if (!button) return;
            this.position = button.dataset.position;
            this.elements.filters.querySelectorAll("button").forEach((item) => item.classList.toggle("is-active", item === button));
            this.renderRankings();
        });
        this.elements.search.addEventListener("input", () => {
            this.query = this.elements.search.value.trim().toLowerCase();
            this.renderRankings();
        });
        this.elements.table.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-player-id]");
            if (!button) return;
            this.selectedId = button.dataset.playerId;
            this.renderRankings();
            this.renderDetail();
            if (window.matchMedia("(max-width: 980px)").matches) {
                this.elements.detail.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
    }

    render() {
        const validation = this.data.validation || {};
        const metrics = validation.metrics || {};
        const seen = validation.seen_weeks || {};
        const unseen = validation.unseen_weeks || metrics;
        this.elements.season.textContent = this.data.season || "NFL";
        const passed = validation.status === "passed";
        this.elements.badge.textContent = passed ? "Holdout passed" : "Holdout failed";
        this.elements.badge.classList.toggle("is-passed", passed);
        this.elements.badge.classList.toggle("is-failed", !passed);
        this.elements.runFacts.innerHTML = [
            this.data.format,
            `${this.formatInt(this.data.model?.simulations)} simulations`,
            `${this.formatInt(this.data.players_published)} ranked players`,
            `Updated ${this.formatDate(this.data.generated_at_utc)}`,
        ].map((item) => `<span>${this.escape(item)}</span>`).join("");
        this.elements.design.textContent = validation.design || "Forward-only validation unavailable.";
        const cards = [
            ["Seen rows", this.formatInt(seen.rows)],
            ["Seen MAE", this.formatNum(seen.mae_fantasy_points, 2)],
            ["Unseen MAE", this.formatNum(unseen.mae_fantasy_points, 2)],
            ["Overfit gap", this.formatSignedPct(validation.overfit_gap)],
            ["vs. recency", this.formatSignedPct(unseen.mae_improvement_vs_recency)],
            ["P10–P90 coverage", this.formatPct(unseen.central_80_interval_coverage)],
        ];
        this.elements.metrics.innerHTML = cards.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>`).join("");
        const tiers = validation.confidence_calibration?.tiers || {};
        this.elements.confidence.innerHTML = ["high", "medium", "low"].map((label) => {
            const tier = tiers[label] || {};
            return `<article><span>${this.escape(label)} confidence</span><strong>${this.formatNum(tier.mae, 2)} MAE</strong><small>${this.formatInt(tier.rows)} unseen rows · ${this.formatPct(tier.coverage)} covered · ±${this.formatNum(tier.mean_interval_half_width, 1)} pts</small></article>`;
        }).join("");
        this.elements.note.innerHTML = `<strong>Research boundary:</strong> ${this.escape(this.data.method_note || "These rankings are model projections, not guarantees.")}`;
        this.renderRankings();
        this.renderDetail();
    }

    visiblePlayers() {
        const players = Array.isArray(this.data?.rankings) ? this.data.rankings : [];
        return players.filter((player) => {
            const positionMatch = this.position === "ALL" || player.position === this.position;
            const text = `${player.player} ${player.team} ${player.position}`.toLowerCase();
            return positionMatch && (!this.query || text.includes(this.query));
        });
    }

    renderRankings() {
        if (!this.data) return;
        const players = this.visiblePlayers();
        if (!players.length) {
            this.elements.table.innerHTML = "<p class=\"fantasy-empty\">No players match this view.</p>";
            return;
        }
        const rows = players.map((player) => {
            const points = player.fantasy_points || {};
            const selected = String(player.player_id) === String(this.selectedId);
            return `<tr class="${selected ? "is-selected" : ""}">
                <td><span class="fantasy-rank">${this.formatInt(player.rank)}</span></td>
                <td><button type="button" class="fantasy-player-button" data-player-id="${this.escape(player.player_id)}"><strong>${this.escape(player.player)}</strong><span>${this.escape(player.team)} · ${this.escape(player.position)}${this.formatInt(player.position_rank)}</span></button></td>
                <td><span class="fantasy-tier">T${this.formatInt(player.tier)}</span></td>
                <td><strong>${this.formatNum(points.per_game, 1)}</strong></td>
                <td>${this.formatNum(points.season_mean, 1)}</td>
                <td>${this.formatNum(points.season_p10, 0)}–${this.formatNum(points.season_p90, 0)}</td>
                <td>${this.formatSignedNum(player.value_over_replacement, 1)}</td>
            </tr>`;
        }).join("");
        this.elements.table.innerHTML = `<table class="prediction-about-table fantasy-table"><thead><tr><th>RK</th><th>Player</th><th>Tier</th><th>PPR/G</th><th>Total</th><th>P10–P90</th><th>VORP</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    renderDetail() {
        if (!this.data) return;
        const player = this.data.rankings.find((item) => String(item.player_id) === String(this.selectedId));
        if (!player) return;
        const points = player.fantasy_points || {};
        const perGame = player.projected_stats?.per_game || {};
        const total = player.projected_stats?.season_total || {};
        const lineup = player.lineup || {};
        const positionNames = { QB: "Quarterback", RB: "Running back", WR: "Wide receiver", TE: "Tight end" };
        const statRows = this.statsFor(player.position).map(([key, label]) => `<tr><td>${this.escape(label)}</td><td>${this.formatNum(perGame[key], this.statPlaces(key))}</td><td>${this.formatNum(total[key], this.statPlaces(key))}</td></tr>`).join("");
        const roleLabel = Number.isFinite(Number(lineup.depth_rank)) ? `${player.position}${this.formatInt(lineup.depth_rank)} on current depth chart` : "Depth role unconfirmed";
        const teamChange = lineup.changed_team ? " / new-team uncertainty applied" : "";
        const distribution = this.renderDistribution(points);
        this.elements.detail.innerHTML = `
            <div class="fantasy-detail-top"><div><span class="fantasy-detail-rank">#${this.formatInt(player.rank)} overall</span><h3>${this.escape(player.player)}</h3><p>${this.escape(player.team)} · ${this.escape(positionNames[player.position] || player.position)} · ${this.escape(player.position)}${this.formatInt(player.position_rank)}</p></div><span class="fantasy-tier fantasy-tier-large">Tier ${this.formatInt(player.tier)}</span></div>
            <p class="fantasy-assessment">${this.escape(player.assessment)}</p>
            ${distribution}
            <div class="fantasy-range"><div><span>Floor</span><strong>${this.formatNum(points.season_p10, 1)}</strong></div><div><span>Median</span><strong>${this.formatNum(points.season_median, 1)}</strong></div><div><span>Ceiling</span><strong>${this.formatNum(points.season_p90, 1)}</strong></div></div>
            <div class="fantasy-detail-metrics"><div><span>PPR / game</span><strong>${this.formatNum(points.per_game, 2)}</strong></div><div><span>Season mean</span><strong>${this.formatNum(points.season_mean, 1)}</strong></div><div><span>VORP</span><strong>${this.formatSignedNum(player.value_over_replacement, 1)}</strong></div><div><span>Draft score</span><strong>${this.formatNum(player.draft_score, 1)}</strong></div></div>
            <p class="fantasy-confidence"><strong>${this.escape(roleLabel)}</strong> / ${this.formatNum(player.games, 1)} expected active or start games of ${this.formatInt(player.schedule_games)}${teamChange}</p>
            <table class="fantasy-stat-table"><thead><tr><th>Stat</th><th>Per game</th><th>Season</th></tr></thead><tbody>${statRows}</tbody></table>
            <p class="fantasy-confidence"><strong>${this.escape(player.projection_confidence)}</strong> confidence / ${this.formatInt(player.history_games)} recent games</p>`;
    }

    renderDistribution(points) {
        const curve = Array.isArray(points?.distribution)
            ? points.distribution.filter((point) => Number.isFinite(Number(point.value)) && Number.isFinite(Number(point.density)))
            : [];
        if (curve.length < 3) return "";
        const width = 620;
        const left = 34;
        const right = 594;
        const top = 48;
        const baseline = 166;
        const low = Number(curve[0].value);
        const high = Number(curve[curve.length - 1].value);
        const span = Math.max(high - low, 1);
        const x = (value) => left + ((Number(value) - low) / span) * (right - left);
        const y = (density) => baseline - Math.max(0, Math.min(1, Number(density))) * (baseline - top);
        const coordinates = curve.map((point) => ({ x: x(point.value), y: y(point.density), density: Number(point.density) }));
        let curvePath = `M ${coordinates[0].x.toFixed(1)} ${coordinates[0].y.toFixed(1)}`;
        for (let index = 1; index < coordinates.length - 1; index += 1) {
            const current = coordinates[index];
            const next = coordinates[index + 1];
            const midX = (current.x + next.x) / 2;
            const midY = (current.y + next.y) / 2;
            curvePath += ` Q ${current.x.toFixed(1)} ${current.y.toFixed(1)} ${midX.toFixed(1)} ${midY.toFixed(1)}`;
        }
        const last = coordinates[coordinates.length - 1];
        curvePath += ` L ${last.x.toFixed(1)} ${last.y.toFixed(1)}`;
        const areaPath = `${curvePath} L ${last.x.toFixed(1)} ${baseline} L ${coordinates[0].x.toFixed(1)} ${baseline} Z`;
        const densityAt = (value) => {
            const target = Number(value);
            for (let index = 1; index < curve.length; index += 1) {
                const previous = curve[index - 1];
                const current = curve[index];
                if (target <= Number(current.value)) {
                    const range = Math.max(Number(current.value) - Number(previous.value), 0.001);
                    const ratio = Math.max(0, Math.min(1, (target - Number(previous.value)) / range));
                    return Number(previous.density) + ratio * (Number(current.density) - Number(previous.density));
                }
            }
            return Number(curve[curve.length - 1].density);
        };
        const meanX = x(points.season_mean);
        const medianX = x(points.season_median);
        const closeMarkers = Math.abs(meanX - medianX) < 74;
        const marker = (kind, label, value, labelY) => {
            const markerX = x(value);
            const markerTop = Math.max(y(densityAt(value)), top);
            return `<g class="fantasy-distribution-marker is-${kind}"><line x1="${markerX.toFixed(1)}" y1="${markerTop.toFixed(1)}" x2="${markerX.toFixed(1)}" y2="${baseline}"/><text x="${markerX.toFixed(1)}" y="${labelY}" text-anchor="middle">${label} ${this.formatNum(value, 1)}</text></g>`;
        };
        const bound = (kind, label, value) => {
            const markerX = x(value);
            return `<g class="fantasy-distribution-bound is-${kind}"><line x1="${markerX.toFixed(1)}" y1="${top}" x2="${markerX.toFixed(1)}" y2="${baseline}"/><text x="${markerX.toFixed(1)}" y="190" text-anchor="middle">${label}</text><text x="${markerX.toFixed(1)}" y="207" text-anchor="middle">${this.formatNum(value, 1)}</text></g>`;
        };
        const skewDifference = Number(points.season_mean) - Number(points.season_median);
        const skew = Math.abs(skewDifference) < span * 0.012 ? "balanced" : skewDifference < 0 ? "left-skewed" : "right-skewed";
        return `<figure class="fantasy-distribution"><div class="fantasy-distribution-heading"><strong>Simulated season distribution</strong><span>${this.escape(skew)}</span></div><svg viewBox="0 0 ${width} 220" role="img" aria-label="Fantasy point distribution. Floor ${this.formatNum(points.season_p10, 1)}, mean ${this.formatNum(points.season_mean, 1)}, median ${this.formatNum(points.season_median, 1)}, and ceiling ${this.formatNum(points.season_p90, 1)}."><path class="fantasy-distribution-area" d="${areaPath}"/><path class="fantasy-distribution-curve" d="${curvePath}"/><line class="fantasy-distribution-axis" x1="${left}" y1="${baseline}" x2="${right}" y2="${baseline}"/>${bound("floor", "P10 FLOOR", points.season_p10)}${bound("ceiling", "P90 CEILING", points.season_p90)}${marker("mean", "MEAN", points.season_mean, closeMarkers ? 22 : 30)}${marker("median", "MEDIAN", points.season_median, closeMarkers ? 40 : 30)}</svg><figcaption>Each curve uses this player's 2,000 simulated season totals.</figcaption></figure>`;
    }

    statsFor(position) {
        if (position === "QB") return [["passing_yards", "Pass yards"], ["passing_tds", "Pass TD"], ["interceptions", "INT"], ["rushing_yards", "Rush yards"], ["rushing_tds", "Rush TD"]];
        return [["receptions", "Receptions"], ["receiving_yards", "Rec yards"], ["receiving_tds", "Rec TD"], ["rushing_yards", "Rush yards"], ["rushing_tds", "Rush TD"]];
    }

    statPlaces(key) { return key.includes("yards") ? 1 : 2; }
    formatDate(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? "n/a" : date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }); }
    formatPct(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedPct(value) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedNum(value, places = 1) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(places)}` : "n/a"; }
    formatNum(value, places = 1) { return Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "n/a"; }
    formatInt(value) { return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a"; }
    escape(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }
}

document.addEventListener("DOMContentLoaded", () => new NflFantasyDraftBoard());
