class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.elements = {
            cards: document.getElementById("predictionCards"),
            empty: document.getElementById("predictionEmpty"),
            runMeta: document.getElementById("predictionRunMeta"),
        };
        this.init();
    }

    async init() {
        try {
            await this.load();
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

    getMonogram(name) {
        const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return "NA";
        if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
        return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
    }

    renderCards() {
        this.elements.empty.style.display = this.plays.length ? "none" : "block";
        this.elements.cards.innerHTML = this.plays.map((play) => this.renderWantedCard(play)).join("");
    }

    renderWantedCard(play) {
        const tierRaw = String(play.confidence_tier || "consider").toLowerCase();
        const tier = play.parlay_candidate ? "parlay" : (["elite", "strong", "consider", "pass"].includes(tierRaw) ? tierRaw : "consider");
        const directionRaw = String(play.direction || "").toUpperCase();
        const direction = directionRaw === "UNDER" ? "UNDER" : "OVER";
        const displayName = this.getPlayDisplayName(play);
        const headshotUrl = this.getPlayHeadshotUrl(play);
        const monogram = this.escapeHtml(this.getMonogram(displayName));
        const edge = Number(play.edge);
        const lineText = this.formatNumber(play.market_line);
        const predictionText = this.formatNumber(play.prediction);
        const edgeText = this.formatSignedNumber(play.edge);
        const valueText = this.formatNumber(play.value_score);
        const hitRateText = this.formatPct(play.estimated_graded_hit_rate);
        const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(" @ ");
        const parlayPartner = String(play.parlay_partner_name || "").trim();
        const parlayRate = this.formatPct(play.parlay_projected_hit_rate);

        return `
            <article class="prediction-card wanted-card wanted-card-${tier}" data-direction="${this.escapeAttr(direction)}">
                <div class="wanted-rank">#${this.escapeHtml(String(play.rank || "-"))}</div>
                <div class="wanted-title">WANTED</div>
                <div class="wanted-photo-frame ${headshotUrl ? "" : "is-fallback-visible"}">
                    ${headshotUrl ? `
                        <img
                            class="wanted-photo"
                            src="${this.escapeAttr(headshotUrl)}"
                            alt="${this.escapeAttr(displayName)} headshot"
                            loading="lazy"
                            onerror="this.remove(); this.parentElement.classList.add('is-fallback-visible');"
                        />
                    ` : ""}
                    <div class="wanted-photo-fallback">${monogram}</div>
                </div>
                ${play.parlay_candidate ? `
                    <div class="wanted-tag-row">
                        <span class="wanted-tag wanted-tag-parlay">PARLAY</span>
                        <span class="wanted-tag wanted-tag-support">PAIR ${this.escapeHtml(parlayRate)}</span>
                    </div>
                ` : ""}
                <div class="wanted-reward-label">EDGE</div>
                <div class="wanted-reward-value">${this.escapeHtml(this.formatSignedNumber(edge))}</div>
                <div class="wanted-name">${this.escapeHtml(displayName)}</div>
                <div class="wanted-direction">${this.escapeHtml(direction)}</div>
                <div class="wanted-prop-line">${this.escapeHtml(play.target || "")} | LINE ${this.escapeHtml(lineText)} | PRED ${this.escapeHtml(predictionText)}</div>
                ${play.parlay_candidate ? `<div class="wanted-parlay-note">Best paired with ${this.escapeHtml(parlayPartner || "another tagged leg")}</div>` : ""}
                <div class="wanted-footer">${this.escapeHtml(`${hitRateText} HIT | VALUE ${valueText} | ${edgeText} EDGE | ${gameText}`)}</div>
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
