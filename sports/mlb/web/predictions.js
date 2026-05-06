class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.elements = {
            parlayTickets: document.getElementById("parlayTickets"),
            cards: document.getElementById("predictionCards"),
            empty: document.getElementById("predictionEmpty"),
            runMeta: document.getElementById("predictionRunMeta"),
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
            this.elements.cards.innerHTML = `<div class="prediction-about-empty">Unable to load MLB predictions: ${this.escapeHtml(error.message)}</div>`;
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
            const scoreDiff = (Number(b.value_score) || 0) - (Number(a.value_score) || 0);
            if (Math.abs(scoreDiff) > 1e-9) return scoreDiff;
            return (Number(b.abs_edge) || 0) - (Number(a.abs_edge) || 0);
        });
        this.renderRunMeta();
    }

    renderRunMeta() {
        const runDate = this.data?.run_date || "n/a";
        const throughDate = this.data?.through_date || "n/a";
        const policy = this.data?.policy_profile || "n/a";
        this.elements.runMeta.textContent = `Run ${runDate} | Data through ${throughDate} | Profile ${policy}`;
    }

    getPlayDisplayName(play) {
        const value = String(play.player_display_name || play.player || "").trim();
        return value || "Unknown Player";
    }

    getPlayHeadshotUrl(play) {
        const explicitUrl = String(play.player_headshot_url || "").trim();
        if (explicitUrl) return explicitUrl;
        const id = Number(play.player_mlbam_id);
        if (Number.isFinite(id) && id > 0) {
            return `https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/${id}/headshot/67/current`;
        }
        return "";
    }

    getPlayHeadshotFallbackUrl(play, primaryUrl = "") {
        const explicitUrl = String(play.player_headshot_fallback_url || "").trim();
        if (explicitUrl && explicitUrl !== primaryUrl) return explicitUrl;
        const id = Number(play.player_mlbam_id);
        if (Number.isFinite(id) && id > 0) {
            const midfieldUrl = `https://midfield.mlbstatic.com/v1/people/${id}/headshot/67/current`;
            if (midfieldUrl !== primaryUrl) return midfieldUrl;
        }
        return "";
    }

    getMonogram(name) {
        const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return "NA";
        if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
        return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
    }

    renderParlayTickets() {
        const container = this.elements.parlayTickets;
        if (!container) return;
        const parlays = Array.isArray(this.data?.parlay_board?.parlays) ? this.data.parlay_board.parlays : [];
        if (!parlays.length) {
            container.innerHTML = "";
            return;
        }
        container.innerHTML = parlays.map((parlay) => {
            const legs = Array.isArray(parlay.legs) ? parlay.legs : [];
            const n = legs.length;
            const jointPct = this.formatPct(parlay.joint_probability || parlay.adjusted_probability);
            const type = String(parlay.type || "primary").toUpperCase();
            const parlayOdds = parlay.odds_american || `+${Math.round((parlay.odds_decimal - 1) * 100)}`;
            const parlayPayout = parlay.payout_per_dollar ? `$${(parlay.payout_per_dollar * 10).toFixed(0)} on $10` : "";
            const legsHtml = legs.map((leg, li) => {
                const name = this.escapeHtml(String(leg.player || ""));
                const target = this.escapeHtml(this.formatTarget(leg.target));
                const dir = this.escapeHtml(String(leg.direction || "").toUpperCase());
                const line = this.formatNumber(leg.market_line);
                const odds = leg.odds_american || "-110";
                const team = this.escapeHtml(String(leg.team || ""));
                return `<div class="parlay-leg">
                    <span class="parlay-leg-num">${li + 1}</span>
                    <span class="parlay-leg-name">${name} <small>(${team})</small></span>
                    <span class="parlay-leg-prop">${target} ${dir} ${this.escapeHtml(line)}</span>
                    <span class="parlay-leg-prob">${this.escapeHtml(String(odds))}</span>
                </div>`;
            }).join("");
            return `<article class="parlay-ticket">
                <div class="parlay-ticket-header">
                    <span class="parlay-ticket-badge">${this.escapeHtml(type)} PARLAY</span>
                    <span class="parlay-ticket-legs">${n}-LEG</span>
                    <span class="parlay-ticket-payout">${this.escapeHtml(String(parlayOdds))}</span>
                </div>
                <div class="parlay-ticket-prob">
                    <span class="parlay-ticket-prob-label">${this.escapeHtml(parlayPayout)}</span>
                    <span class="parlay-ticket-prob-value">${this.escapeHtml(jointPct)} hit rate</span>
                </div>
                <div class="parlay-ticket-legs-list">${legsHtml}</div>
                <div class="parlay-ticket-actions">
                    <button class="bet-action-btn" onclick="navigator.clipboard.writeText(this.closest('.parlay-ticket').querySelector('.parlay-ticket-legs-list').innerText.replace(/\\d+/g,'').trim())">Copy Slip</button>
                    <a class="bet-action-btn" href="https://sportsbook.draftkings.com/mlb-player-props" target="_blank" rel="noopener">DraftKings</a>
                    <a class="bet-action-btn" href="https://sportsbook.fanduel.com/navigation/mlb" target="_blank" rel="noopener">FanDuel</a>
                    <a class="bet-action-btn" href="https://www.bet365.com/#/AS/B63/" target="_blank" rel="noopener">bet365</a>
                </div>
            </article>`;
        }).join("");
    }

    renderCards() {
        this.elements.empty.style.display = this.plays.length ? "none" : "block";
        this.elements.cards.innerHTML = this.plays.map((play) => this.renderWantedCard(play)).join("");
    }

    formatTarget(target) {
        const lookup = {
            H: "HITS",
            TB: "TOTAL BASES",
            R: "RUNS",
            K: "PITCHER K",
            HR: "HOME RUNS",
            RBI: "RBIS",
            ER: "EARNED RUNS",
        };
        return lookup[String(target || "").toUpperCase()] || String(target || "").toUpperCase();
    }

    formatSource(source) {
        return String(source || "").toLowerCase() === "real" ? "BOOK LINE" : "BENCHMARK LINE";
    }

    renderWantedCard(play) {
        const directionRaw = String(play.direction || "").toUpperCase();
        const direction = directionRaw === "UNDER" ? "UNDER" : "OVER";
        const displayName = this.getPlayDisplayName(play);
        const target = this.escapeHtml(this.formatTarget(play.target));
        const lineText = this.formatNumber(play.market_line);
        const hitRate = this.formatPct(play.estimated_graded_hit_rate);
        const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(" @ ");
        const team = this.escapeHtml(String(play.team || ""));
        const headshotUrl = this.getPlayHeadshotUrl(play);
        const monogram = this.escapeHtml(this.getMonogram(displayName));
        const odds = play.odds_american || -110;

        return `
            <article class="bounty-card" data-direction="${this.escapeAttr(direction)}">
                <div class="bounty-top">
                    <span class="bounty-wanted">WANTED</span>
                    <span class="bounty-odds">${this.escapeHtml(String(odds))}</span>
                </div>
                <div class="bounty-headshot ${headshotUrl ? "" : "is-fallback"}">
                    ${headshotUrl ? `<img src="${this.escapeAttr(headshotUrl)}" alt="${this.escapeAttr(displayName)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove(); this.parentElement.classList.add('is-fallback');" />` : ""}
                    <span class="bounty-headshot-fallback">${monogram}</span>
                </div>
                <div class="bounty-name">${this.escapeHtml(displayName)}</div>
                <div class="bounty-pick">
                    <span class="bounty-target">${target}</span>
                    <span class="bounty-direction">${this.escapeHtml(direction)}</span>
                    <span class="bounty-line">${this.escapeHtml(lineText)}</span>
                </div>
                <div class="bounty-meta">${team ? this.escapeHtml(team) + " | " : ""}${this.escapeHtml(gameText)}${hitRate ? " | " + this.escapeHtml(hitRate) + " HIT" : ""}</div>
                ${play.betslip_link ? `<a class="bet-action-btn bounty-betslip" href="${this.escapeAttr(play.betslip_link)}" target="_blank" rel="noopener">Place on FanDuel</a>` : ""}
            </article>
        `;
    }

    formatNumber(value) {
        return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "n/a";
    }

    formatSignedNumber(value) {
        return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}` : "n/a";
    }

    formatPct(value) {
        return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a";
    }

    escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    escapeAttr(value) {
        return this.escapeHtml(value).replaceAll("`", "&#96;");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new DailyPredictionsPage();
});
